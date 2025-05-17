import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import pickle
import collections
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Shift

# Focal Loss implementation
def focal_loss(alpha=0.25, gamma=1.5):
    """
    Focal loss for one-hot encoded targets.
    """
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return loss_fn

# Settings
DATASET_PATH = "VERİ SETİ YOLU(DATASET_PATH)"
LABELS = ["belly_pain", "burping", "discomfort", "hungry", "Background_Noise", "tired"]
SAMPLE_RATE = 22050
FIXED_DURATION = 2.0
MFCC_FEATURES = 40
EPOCHS = 100

# Data augmentation pipeline
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    Shift(min_shift=-0.2, max_shift=0.2, shift_unit="fraction", p=0.3)
])

# Validate dataset
def validate_dataset():
    print("🔍 Validating dataset...")
    counts = {}
    files_by_class = {}
    invalid_files = []
    
    for label in LABELS:
        folder_path = os.path.join(DATASET_PATH, label)
        wav_files = []
        if os.path.exists(folder_path):
            wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        counts[label] = len(wav_files)
        files_by_class[label] = []
        for file in wav_files:
            file_path = os.path.join(folder_path, file)
            try:
                _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=FIXED_DURATION)
                files_by_class[label].append(file)
            except Exception as e:
                print(f"Invalid file: {file_path} | Error: {e}")
                invalid_files.append(file_path)
    
    print("\n🔢 Valid files per label:")
    for label, count in counts.items():
        print(f"{label}: {count}")
    
    if invalid_files:
        print(f"\n⚠️ {len(invalid_files)} invalid files found:")
        for file in invalid_files:
            print(file)
    
    return counts, files_by_class, invalid_files

# Feature extraction
def extract_features(file_path, scaler=None, top_features=None, augment_data=False, attempt=0):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=FIXED_DURATION)
        target_length = int(FIXED_DURATION * SAMPLE_RATE)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        if augment_data:
            audio = augment(audio, sample_rate=sr)
        
        audio = audio / (np.max(np.abs(audio)) + 1e-7)
        
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_FEATURES)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.vstack([mfcc, delta, delta2])
        mfccs_scaled = np.mean(combined.T, axis=0)
        
        if top_features is not None:
            mfccs_scaled = mfccs_scaled[top_features]
        
        if scaler is not None:
            mfccs_scaled = scaler.transform(mfccs_scaled.reshape(1, -1)).flatten()
        
        return mfccs_scaled
    except Exception as e:
        print(f"Error for {file_path}: {e} | Attempt: {attempt}")
        return None

# Load and prepare data
def load_data():
    counts, files_by_class, invalid_files = validate_dataset()
    
    features, targets, file_paths = [], [], []
    for idx, label in enumerate(LABELS):
        for file in files_by_class[label]:
            path = os.path.join(DATASET_PATH, label, file)
            feat = extract_features(path)
            if feat is not None:
                features.append(feat)
                targets.append(idx)
                file_paths.append(path)
            # Data augmentation
            for attempt in range(5):
                feat_aug = extract_features(path, augment_data=True, attempt=attempt+1)
                if feat_aug is not None:
                    features.append(feat_aug)
                    targets.append(idx)
                    file_paths.append(f"{path}_aug_{attempt+1}")
    
    features = np.array(features)
    targets = np.array(targets)
    
    print(f"\n✅ Extracted features from {len(features)} samples.")
    print("📊 Label distribution:", collections.Counter(targets))
    
    # Feature importance
    clf = RandomForestClassifier(random_state=42)
    clf.fit(features, targets)
    importances = clf.feature_importances_
    top_features = np.argsort(importances)[::-1][:80]
    
    # Save top features
    os.makedirs("saved_models", exist_ok=True)
    with open("saved_models/top_features.pkl", 'wb') as f:
        pickle.dump(top_features, f)

    # Reduce feature set
    features = features[:, top_features]
    
    # Scale
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    with open("saved_models/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    return features, targets, file_paths, scaler, top_features

# Build model
def build_model(input_len, num_classes):
    model = Sequential([
        Input(shape=(input_len, 1)),
        Conv1D(32, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu', padding='same'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss=focal_loss(alpha=0.25, gamma=1.5),
        metrics=['accuracy']
    )
    return model

# Main
def main():
    X, y, paths, scaler, top_features = load_data()
    y = to_categorical(y, num_classes=len(LABELS))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    model = build_model(X_train.shape[1], len(LABELS))
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=8,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    # Evaluation
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=LABELS))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=LABELS, yticklabels=LABELS, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('saved_models/confusion_matrix.png')
    plt.show()

    # Save weights
    os.makedirs('saved_models', exist_ok=True)
    model.save_weights('saved_models/baby_cry_weights.h5')

    # TFLite conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('saved_models/baby_cry_model.tflite', 'wb') as f:
        f.write(tflite_model)

    # Verify TFLite
    interpreter = tf.lite.Interpreter(model_path='saved_models/baby_cry_model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    correct = 0
    confidences = []
    for i in range(len(X_test)):
        inp = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details['index'], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details['index'])
        pred = np.argmax(out)
        confidences.append(np.max(out))
        if pred == y_true[i]: correct += 1
    print(f"TFLite accuracy: {correct/len(X_test):.4f}")
    print(f"Average confidence: {np.mean(confidences):.4f}")

if __name__ == '__main__':
    main()
