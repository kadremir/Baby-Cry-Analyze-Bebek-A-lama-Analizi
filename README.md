# 👶 Bebek Ağlama Sebebi Tahmin Modeli (TensorFlow)

<div align="center">
  
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

</div>

## 📋 Proje Hakkında
Bebeklerin ağlama seslerini yapay zeka ile analiz ederek **neden ağladıklarını tahmin eden** bir derin öğrenme modeli.

### Kullanılan Veri Seti
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus)

### Tahmin Edilen Ağlama Sebepleri
- 🤕 **Karın Ağrısı** (belly_pain)
- 🫧 **Gaz Çıkarma İhtiyacı** (burping)
- 😣 **Rahatsızlık** (discomfort)
- 🍼 **Açlık** (hungry)
- 😴 **Yorgunluk** (tired)
- 🔇 **Arka Plan Gürültüsü** (Background_Noise)

## 🔍 Teknik Detaylar
Model, ses dosyalarından **MFCC (Mel Frekans Kepstral Katsayıları)** ve türevlerini kullanarak özellik çıkarımı yapar ve derin öğrenme ile sınıflandırma gerçekleştirir.

## ⚠️ Önemli Bilgilendirme
## (Veriler Eşitlenmeden Önce/Before the data is synchronized)
![WhatsApp Görsel 2025-05-18 saat 01 45 22_4bc74d6f](https://github.com/user-attachments/assets/bc4caf3f-d02f-45e0-bd23-9c99ff1982a2)

## (Veriler Eşitlendikten Sonra/After the data is synchronized)
![WhatsApp Görsel 2025-05-18 saat 01 13 57_8ef48887](https://github.com/user-attachments/assets/d3a71861-d608-4564-afe8-f54bab57d484)

<table>
<tr>
<td>
Model altyapısı tamamen hazır olmasına rağmen, şu an kullanılan eğitim veri setinde sınıflar arasında ciddi dengesizlik bulunmaktadır. Özellikle <code>hungry</code> <code>discomfort</code> sınıfındaki örneklerin fazlalığı, modelin bu sınıfa aşırı odaklanmasına neden olmaktadır.

**Gözlemlerimiz:**
- Hungry ve Discomfort verileri dengelenmeden eğitildiğinde, tahminler bu sınıfa yönelmekte
- Diğer sınıfların tahmin oranları düşük kalmakta
- Dengeli veri setiyle eğitildiğinde doğruluk oranı artmaktadır
</td>
</tr>
</table>

## 📁 Proje Yapısı

```
├── bebek_aglama_tensorflow.py         # Model eğitimi ve veri artırma
├── bebek_aglama_tensorflow_test_script.py  # Test ve değerlendirme 
├── saved_models/                      # Model dosyaları
├── tensor_egitim_verisi/              # Eğitim ses dosyaları
├── tensor_test_verisi/                # Test ses dosyaları  
└── verisetim/                         # Sınıflandırılmış ses dosyaları
```

## 🚀 Kullanım

### 1. Gereksinimlerin Kurulumu

```bash
pip install librosa tensorflow scikit-learn audiomentations seaborn matplotlib
```

### 2. Model Eğitimi

```bash
python bebek_aglama_tensorflow.py
```

- MFCC tabanlı özellik çıkarımı yapılır
- Ses artırma teknikleri uygulanır
- Model eğitilir ve kaydedilir

### 3. Model Testi

```bash
python bebek_aglama_tensorflow_test_script.py
```

- Sınıflandırma raporunu gösterir
- Karışıklık matrisini görselleştirir

## 📊 Performans İyileştirme Önerileri

| İyileştirme | Açıklama |
|------------|-----------|
| 📊 **Veri Dengesi** | Sınıflar arasındaki örnek sayısı dengelenmeli |
| 🔄 **Gelişmiş Veri Artırma** | Çeşitli ses dönüşüm teknikleri uygulanabilir |
| 🧠 **Model Mimarisi** | LSTM, GRU, Attention gibi yapılar denenebilir |
| ⚙️ **Hiperparametreler** | Grid Search veya Bayesian optimizasyon uygulanabilir |

## 📥 Veri Seti

Veri seti ve eğitilmiş model dosyaları talep üzerine paylaşılabilir. Dengeli veri setiyle model performansı önemli ölçüde artmaktadır.

## 📄 Lisans

Bu proje [MIT Lisansı](https://opensource.org/licenses/MIT) ile lisanslanmıştır.

---

<details>
<summary>English Version</summary>

# 👶 Baby Cry Reason Classification Model (TensorFlow)

<div align="center">
  
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

</div>

## 📋 About the Project
A deep learning model that analyzes baby cry sounds to **predict why they're crying**.

### Dataset Used
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus)

### Predicted Crying Reasons
- 🤕 **Belly Pain** (belly_pain)
- 🫧 **Burping Needed** (burping)
- 😣 **Discomfort** (discomfort)
- 🍼 **Hunger** (hungry)
- 😴 **Tiredness** (tired)
- 🔇 **Background Noise** (Background_Noise)

## 🔍 Technical Details
The model extracts features using **MFCC (Mel Frequency Cepstral Coefficients)** and its derivatives from audio files and performs classification with deep learning.

## ⚠️ Important Notice

<table>
<tr>
<td>
While the model infrastructure is fully operational, the current training dataset has significant class imbalance. Especially the abundance of samples in the <code>hungry</code> <code>discomfort</code> class causes the model to focus excessively on this class.

**Our observations:**
- When trained without balancing, predictions tend toward the hungry and discomfort class
- Other classes' prediction rates remain low
- Accuracy increases significantly with a balanced dataset
</td>
</tr>
</table>

## 📁 Project Structure

```
├── bebek_aglama_tensorflow.py         # Model training and augmentation
├── bebek_aglama_tensorflow_test_script.py  # Testing and evaluation 
├── saved_models/                      # Model files
├── tensor_egitim_verisi/              # Training audio files
├── tensor_test_verisi/                # Test audio files  
└── verisetim/                         # Classified audio files
```

## 🚀 Usage

### 1. Install Requirements

```bash
pip install librosa tensorflow scikit-learn audiomentations seaborn matplotlib
```

### 2. Train the Model

```bash
python bebek_aglama_tensorflow.py
```

- Performs MFCC-based feature extraction
- Applies audio augmentation techniques
- Trains and saves the model

### 3. Test the Model

```bash
python bebek_aglama_tensorflow_test_script.py
```

- Shows classification report
- Visualizes confusion matrix

## 📊 Performance Improvement Suggestions

| Improvement | Description |
|------------|-----------|
| 📊 **Data Balance** | Balance sample counts between classes |
| 🔄 **Advanced Augmentation** | Apply various audio transformation techniques |
| 🧠 **Model Architecture** | Experiment with LSTM, GRU, Attention structures |
| ⚙️ **Hyperparameters** | Apply Grid Search or Bayesian optimization |

## 📥 Dataset

The dataset and trained model files can be shared upon request. Model performance significantly increases with a balanced dataset.

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
