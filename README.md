# Eye Disease Classification using Deep Learning

## 🔬 Project Overview

This project develops a machine learning solution for automated classification of eye diseases using deep learning techniques. By leveraging transfer learning and state-of-the-art convolutional neural network architectures, we aim to create a robust diagnostic support tool for detecting key eye conditions.

### 🎯 Project Goals
- Develop an accurate multi-class image classification model for eye diseases
- Compare performance across different pre-trained neural network architectures
- Create a reproducible and scalable machine learning workflow

### 🩺 Supported Disease Categories
- Cataract
- Glaucoma
- Diabetic Retinopathy
- Normal (Healthy) Eye Condition

## 📊 Technical Approach

### Methodology
- **Transfer Learning**: Utilize pre-trained neural network architectures
- **Multi-class Classification**: Classify images into four distinct categories
- **Model Architectures**:
  - VGG16
  - ResNet50
  - MobileNetV2

### Performance Metrics
- Accuracy
- F1-Score
- Precision
- Recall
- Confusion Matrix

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- NumPy
- Matplotlib (for visualization)

### Installation

1. Clone the repository
```bash
git clone https://github.com/Setsofia-lab/EyeDisease_Classifier_DeepLearning.git
cd EyeDisease_Classifier_DeepLearning
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required packages
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
eye-disease-classification/
│
├── dataset/                   # Raw image dataset
│   ├── cataract/
│   ├── glaucoma/
│   ├── diabetic_retinopathy/
│   └── normal/
│
├── output_directory/          # Processed dataset splits
│   ├── train/
│   ├── val/
│   └── test/
│
├── scripts/
│   ├── split_dataset.py       # Dataset splitting utility
│   ├── train_vgg16.py         # VGG16 training script
│   ├── train_resnet50.py      # ResNet50 training script
│   └── train_mobilenetv2.py   # MobileNetV2 training script
│
├── models/                    # Saved model weights
├── notebooks/                 # Jupyter notebooks for exploration
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🔧 Usage

### 1. Prepare Dataset
Organize your image dataset into the required folder structure:
```
dataset/
├── cataract/
├── glaucoma/
├── diabetic_retinopathy/
└── normal/
```

### 2. Split Dataset
```bash
python scripts/split_dataset.py
```

### 3. Train Models
```bash
python scripts/train_vgg16.py
python scripts/train_resnet50.py
python scripts/train_mobilenetv2.py
```

## 📈 Model Performance

### Comparative Results

(base) samuelsetsofia@Samuels-MBP-2 eyeDiseaseImage_classifier % python model.py
Found 2949 files belonging to 4 classes.
Found 633 files belonging to 4 classes.
Found 635 files belonging to 4 classes.
Epoch 1/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 233s 2s/step - accuracy: 0.6420 - loss: 8.3180 - val_accuracy: 0.8183 - val_loss: 0.5676
Epoch 2/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 228s 2s/step - accuracy: 0.7967 - loss: 0.5542 - val_accuracy: 0.8183 - val_loss: 0.5344
Epoch 3/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 235s 3s/step - accuracy: 0.8223 - loss: 0.4606 - val_accuracy: 0.8594 - val_loss: 0.3928
Epoch 4/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 251s 3s/step - accuracy: 0.8339 - loss: 0.4190 - val_accuracy: 0.8120 - val_loss: 0.6745
Epoch 5/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 250s 3s/step - accuracy: 0.8350 - loss: 0.4323 - val_accuracy: 0.8578 - val_loss: 0.4465
Epoch 1/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 80s 827ms/step - accuracy: 0.6358 - loss: 0.9529 - val_accuracy: 0.8515 - val_loss: 0.4054
Epoch 2/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 70s 755ms/step - accuracy: 0.8151 - loss: 0.4674 - val_accuracy: 0.8673 - val_loss: 0.3526
Epoch 3/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 75s 806ms/step - accuracy: 0.8531 - loss: 0.3816 - val_accuracy: 0.8815 - val_loss: 0.3154
Epoch 4/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 73s 781ms/step - accuracy: 0.8623 - loss: 0.3693 - val_accuracy: 0.8736 - val_loss: 0.3225
Epoch 5/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 71s 762ms/step - accuracy: 0.8733 - loss: 0.3263 - val_accuracy: 0.8973 - val_loss: 0.2766
Epoch 1/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 24s 241ms/step - accuracy: 0.6113 - loss: 0.9574 - val_accuracy: 0.8104 - val_loss: 0.5019
Epoch 2/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 22s 240ms/step - accuracy: 0.7710 - loss: 0.5798 - val_accuracy: 0.8215 - val_loss: 0.4556
Epoch 3/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 22s 237ms/step - accuracy: 0.7831 - loss: 0.5292 - val_accuracy: 0.8341 - val_loss: 0.4277
Epoch 4/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 22s 239ms/step - accuracy: 0.8042 - loss: 0.4996 - val_accuracy: 0.8246 - val_loss: 0.4161
Epoch 5/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 22s 231ms/step - accuracy: 0.8160 - loss: 0.4684 - val_accuracy: 0.8436 - val_loss: 0.4062

Model Performance Before Tuning
{'Model': 'VGG16', 'Val Accuracy': 0.859399676322937}
{'Model': 'ResNet50', 'Val Accuracy': 0.8973143696784973}
{'Model': 'MobileNetV2', 'Val Accuracy': 0.8436018824577332}
Epoch 1/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 226s 2s/step - accuracy: 0.6311 - loss: 3.9705 - val_accuracy: 0.8452 - val_loss: 0.5114
Epoch 2/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 226s 2s/step - accuracy: 0.8266 - loss: 0.4828 - val_accuracy: 0.8673 - val_loss: 0.3835
Epoch 3/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 225s 2s/step - accuracy: 0.8702 - loss: 0.3475 - val_accuracy: 0.8815 - val_loss: 0.3660
Epoch 4/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 225s 2s/step - accuracy: 0.9201 - loss: 0.2269 - val_accuracy: 0.8784 - val_loss: 0.3570
Epoch 5/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 225s 2s/step - accuracy: 0.9289 - loss: 0.2045 - val_accuracy: 0.8894 - val_loss: 0.3467
Epoch 1/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 74s 765ms/step - accuracy: 0.4992 - loss: 1.2277 - val_accuracy: 0.7946 - val_loss: 0.5652
Epoch 2/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 70s 755ms/step - accuracy: 0.7497 - loss: 0.6252 - val_accuracy: 0.8199 - val_loss: 0.4759
Epoch 3/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 70s 754ms/step - accuracy: 0.7939 - loss: 0.5179 - val_accuracy: 0.8404 - val_loss: 0.4308
Epoch 4/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 70s 751ms/step - accuracy: 0.8223 - loss: 0.4663 - val_accuracy: 0.8578 - val_loss: 0.3971
Epoch 5/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 70s 753ms/step - accuracy: 0.8415 - loss: 0.4314 - val_accuracy: 0.8594 - val_loss: 0.3820
Epoch 1/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 24s 236ms/step - accuracy: 0.4542 - loss: 1.2892 - val_accuracy: 0.7283 - val_loss: 0.6967
Epoch 2/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 21s 225ms/step - accuracy: 0.6901 - loss: 0.7902 - val_accuracy: 0.7709 - val_loss: 0.6152
Epoch 3/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 21s 229ms/step - accuracy: 0.7369 - loss: 0.6596 - val_accuracy: 0.8025 - val_loss: 0.5627
Epoch 4/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 21s 228ms/step - accuracy: 0.7711 - loss: 0.6110 - val_accuracy: 0.8088 - val_loss: 0.5300
Epoch 5/5
93/93 ━━━━━━━━━━━━━━━━━━━━ 22s 232ms/step - accuracy: 0.7833 - loss: 0.5658 - val_accuracy: 0.8073 - val_loss: 0.5096

Model Performance After Tuning
{'Model': 'VGG16', 'Val Accuracy': 0.8894155025482178}
{'Model': 'ResNet50', 'Val Accuracy': 0.859399676322937}
{'Model': 'MobileNetV2', 'Val Accuracy': 0.8088467717170715}

## 🧠 Transfer Learning Strategy
- Used pre-trained ImageNet weights
- Froze base model layers
- Added custom classification head
- Applied dropout for regularization


## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
