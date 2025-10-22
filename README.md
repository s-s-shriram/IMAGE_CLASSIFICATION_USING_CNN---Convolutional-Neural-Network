# IMAGE_CLASSIFICATION_USING_CNN---Convolutional-Neural-Network
CNN image classifier using TensorFlow &amp; MobileNetV2 to classify like (Cat, Dog, and Horse) images with transfer learning

# Image Classification using CNN

A deep learning project that classifies images of Animals (Lion, Leopard, Tiger) , Birds (Parrot, Peacock, Owl) using Convolutional Neural Networks (CNN) with Transfer Learning approach.

## 🎯 Project Overview

This project implements an image classification model using TensorFlow and Keras with MobileNetV2 as the base model. The model is trained to distinguish between the classes : (LIKE: Cat, Dog, Tiger, Lion, Birds, Horse, etc...)

## ✨ Features

- **Transfer Learning**: Uses pre-trained MobileNetV2 for efficient training
- **Data Augmentation**: Implements random flip, rotation, and zoom for robust training
- **Easy Dataset Upload**: Simple zip file upload mechanism
- **Interactive Testing**: Upload any image to get instant predictions
- **High Accuracy**: Achieves good classification accuracy with minimal training time

## 🛠️ Technologies Used

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Google Colab (optional)

## 📁 Dataset Structure

The dataset should be organized in the following structure:
```
dataset.zip
├── Classname-1/
│   ├── sampleimg1.jpg
│   ├── sampleimg2.jpg
│   └── ...
├── Classname-2/
│   ├── sampleimg1.jpg
│   ├── sampleimg2.jpg
│   └── ...
└── Classname-../
    ├── sampleimg1.jpg
    ├── sampleimg2.jpg
    └── ...
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow numpy matplotlib
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/animal-image-classifier-cnn.git
cd animal-image-classifier-cnn
```

2. Open the notebook in Google Colab or Jupyter Notebook

### Usage

1. **Prepare Dataset**: Create a zip file with folders named after each class containing respective images
2. **Upload Dataset**: Run the notebook and upload your `dataset.zip` when prompted
3. **Train Model**: The model will automatically train for 20 epochs
4. **Test Model**: Upload a test image to get predictions

## 📊 Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers**:
  - Global Average Pooling 2D
  - Dense Layer (128 neurons, ReLU activation)
  - Output Layer (3 neurons, Softmax activation)

## 🎓 Training Details

- **Image Size**: 224x224 pixels
- **Batch Size**: 4
- **Epochs**: 20
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Train/Test Split**: 80/20

## 📈 Results

The model achieves high accuracy in classifying the three animal classes with minimal training time due to transfer learning.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

Your Name
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- MobileNetV2 architecture from TensorFlow/Keras
- Pre-trained weights from ImageNet dataset
```

---

## GitHub Topics/Tags to Add:
```
deep-learning
machine-learning
image-classification
cnn
tensorflow
keras
transfer-learning
computer-vision
python
mobilenet
animal-classification
