# 1. Gerekli kütüphanelerin kurulumu
!pip install librosa tensorflow scikit-learn audiomentations seaborn matplotlib

import os
import numpy as np
import librosa
import tensorflow as tf
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Dosyaların yolunu ayarlayalım
# Colab'a yüklediyseniz: /content/scaler.pkl, /content/top_features.pkl, /content/baby_cry_model.tflite
SCALER_PATH = '/content/scaler.pkl'
TOP_FEATURES_PATH = '/content/top_features.pkl'
TFLITE_MODEL_PATH = '/content/baby_cry_model.tflite'
DATASET_PATH = '/content/verisetim/'  # Colab'da ses klasörünüz
LABELS = ["belly_pain", "burping", "discomfort", "hungry", "Background_Noise", "tired"]
SAMPLE_RATE = 22050
FIXED_DURATION = 2.0

# 3. Önceden kaydedilmiş scaler ve top_features yükleyelim
with open(SCALER_PATH, 'rb') as f:
    scaler: MinMaxScaler = pickle.load(f)
with open(TOP_FEATURES_PATH, 'rb') as f:
    top_features: np.ndarray = pickle.load(f)
print("🔄 Scaler ve top_features yüklendi.")

# 4. Özellik çıkarma fonksiyonu (sabit uzunluk + MFCC+delta+delta2)
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=FIXED_DURATION)
    target_len = int(FIXED_DURATION * SAMPLE_RATE)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    audio = audio / (np.max(np.abs(audio)) + 1e-7)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])
    feat = np.mean(combined.T, axis=0)

    # Top özellikleri seç
    feat = feat[top_features]
    # Ölçekle
    feat = scaler.transform(feat.reshape(1, -1)).flatten()
    return feat

# 5. Veri setini yükleme ve etiketleme
features = []
labels = []
for idx, label in enumerate(LABELS):
    class_dir = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_dir): continue
    for file in os.listdir(class_dir):
        if file.endswith('.wav'):
            path = os.path.join(class_dir, file)
            features.append(extract_features(path))
            labels.append(idx)

features = np.array(features, dtype=np.float32)
labels = np.array(labels)
print(f"✅ Toplam örnek: {len(labels)} | Sınıf dağılımı: {np.bincount(labels)}")

# 6. TFLite modelini yükleyelim
tflite_model = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
tflite_model.allocate_tensors()
input_details = tflite_model.get_input_details()[0]
output_details = tflite_model.get_output_details()[0]

# 7. Tüm örnekler için tahmin yapalım
preds = []
confs = []
for feat in features:
    inp = feat.reshape(1, -1, 1)  # (1, features, 1)
    tflite_model.set_tensor(input_details['index'], inp.astype(np.float32))
    tflite_model.invoke()
    out = tflite_model.get_tensor(output_details['index'])
    pred = np.argmax(out)
    conf = np.max(out)
    preds.append(pred)
    confs.append(conf)

# 8. Sonuçları değerlendirelim
print("\nClassification Report:\n", classification_report(labels, preds, target_names=LABELS))
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=LABELS, yticklabels=LABELS, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Ortalama güven skorunu yazdır
print(f"Average Confidence: {np.mean(confs):.4f}")
