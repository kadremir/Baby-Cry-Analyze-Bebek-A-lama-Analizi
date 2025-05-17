**👶 Bebek Ağlama Sebebi Tahmin Modeli (TensorFlow)**

KULLANILAN DATA: https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus
Bu proje, TensorFlow kullanılarak geliştirilmiş ve bir bebeğin ağlama sesini analiz ederek altı farklı kategoriye ayırmayı amaçlayan bir makine öğrenimi modelini içerir:

* belly\_pain
* burping
* discomfort
* hungry
* tired
* Background\_Noise

Model, ses dosyalarından MFCC (Mel Frekans Kepstral Katsayıları) ve türevleri ile özellik çıkarımı yaparak eğitilmiştir.

---

**⚠️ Önemli Bilgilendirme**

Modelin altyapısı tamamen hazırdır, ancak şu an kullanılan eğitim veri setinde sınıflar arasında ciddi dengesizlik vardır. Özellikle `hungry` sınıfına ait örneklerin fazlalığı, modelin bu sınıfa aşırı odaklanmasına ve diğer sınıfları doğru şekilde ayırt edememesine neden olmaktadır.

Gözlemlerimize göre:

* hungry verileri eksiltilmeden model eğitildiğinde, test sonuçları büyük oranda bu sınıfa yöneliyor.
* Diğer sınıfların tahmin değerleri sıfıra kadar düşüyor.
* Uygun sayıda ve dengeli veri içeren bir set ile eğitildiğinde modelin çok daha yüksek doğrulukla çalışacağı öngörülmektedir.

---

**📂 Proje Yapısı**

bebek\_aglama\_tensorflow\.py → Model eğitimi, veri artırma ve TFLite dönüştürme
bebek\_aglama\_tensorflow\_test\_script.py → Eğitilen modelin test edilmesi ve raporlanması
saved\_models/ → Model ağırlıkları, scaler ve özellik seçim dosyaları
tensor eğitim verisi/ → Eğitim sırasında kullanılan ses verileri
tensor test verisi/ → Test sırasında kullanılan ses verileri
verisetim/ → Sınıflandırmaya göre yapılandırılmış ses klasörleri

---

**🚀 Kullanım Talimatları**

1. Gereksinimler:
   pip install librosa tensorflow scikit-learn audiomentations seaborn matplotlib

2. Modeli Eğitme:
   python bebek\_aglama\_tensorflow\.py

* MFCC tabanlı özellik çıkarımı yapılır
* Veri artırma uygulanır
* Model eğitilir ve saved\_models klasörüne kaydedilir

3. Modeli Test Etme:
   python bebek\_aglama\_tensorflow\_test\_script.py

* Sınıflandırma raporu ve confusion matrix görselleştirilir

---

**📈 Geliştirme Önerileri**

* Veri dengesi sağlanmalı
* Gelişmiş augmentasyon teknikleri uygulanabilir
* LSTM, GRU, Attention gibi modeller denenebilir
* Hiperparametre ayarlamaları yapılmalıdır

---

**📥 Veri Seti Talebi**

Veri seti talebi üzerine paylaşılabilir. Dengeli veriyle model çok daha iyi performans göstermektedir.

---

**📝 Lisans**

Bu proje MIT Lisansı ile lisanslanmıştır.

---

---

**👶 Baby Cry Reason Classification Model (TensorFlow)**

USED DATA: https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus
This project features a machine learning model developed with TensorFlow, aiming to classify a baby's cry sound into six different categories:

* belly\_pain
* burping
* discomfort
* hungry
* tired
* Background\_Noise

The model is trained using MFCC (Mel Frequency Cepstral Coefficients) and its derivatives for feature extraction from audio files.

---

**⚠️ Important Notice**

The model infrastructure is fully operational. However, the current training dataset suffers from significant class imbalance. Especially, the abundance of samples in the `hungry` class causes the model to overfit on it and fail to distinguish other classes accurately.

Our observations:

* When trained without balancing, predictions mostly lean toward the `hungry` class.
* Other classes’ prediction rates drop nearly to zero.
* A more balanced dataset is expected to significantly improve model performance.

---

**📂 Project Structure**

bebek\_aglama\_tensorflow\.py → Training the model, augmentation, and TFLite conversion
bebek\_aglama\_tensorflow\_test\_script.py → Testing and evaluating the trained model
saved\_models/ → Model weights, scaler, and feature selection files
tensor eğitim verisi/ → Training audio files
tensor test verisi/ → Testing audio files
verisetim/ → Sound files organized by class folders

---

**🚀 Instructions**

1. Requirements:
   pip install librosa tensorflow scikit-learn audiomentations seaborn matplotlib

2. Training the Model:
   python bebek\_aglama\_tensorflow\.py

* Extracts MFCC features
* Applies audio augmentation
* Trains and saves the model into saved\_models/

3. Testing the Model:
   python bebek\_aglama\_tensorflow\_test\_script.py

* Generates classification report and visualizes confusion matrix

---

**📈 Suggestions for Improvement**

* Balance the dataset
* Apply advanced augmentation techniques
* Experiment with LSTM, GRU, or Attention-based models
* Optimize hyperparameters

---

**📥 Dataset Request**

The dataset used for training and testing can be shared upon request. With balanced data, this infrastructure provides significantly better accuracy.

---

**📝 License**

This project is licensed under the MIT License.

---
