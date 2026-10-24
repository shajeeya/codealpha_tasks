# CodeAlpha Machine Learning Internship

This repository contains machine learning projects developed as part of my CodeAlpha internship.

## Projects

### Task 2 — Speech Emotion Recognition

A machine learning project that classifies emotions from speech recordings using MFCC-based audio feature extraction and supervised learning.

**Highlights**
- RAVDESS speech dataset
- Audio feature extraction using Librosa
- SVM, Random Forest, and Extra Trees evaluation
- Actor-independent cross-validation
- WAV audio prediction script
- Evaluation metrics and confusion matrix

[View Task 2 documentation](Task2_Emotion_Recognition/README.md)

### Task 3 — Handwritten Digit Recognition Using CNN

A deep learning project that recognizes handwritten digits from 0 to 9 using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

**Highlights**
- MNIST dataset with 60,000 training images and 10,000 test images
- CNN model built using TensorFlow and Keras
- **98.94% test accuracy** on the MNIST test dataset
- Custom handwritten digit image prediction
- **Live air-writing digit recognition using a webcam**
- Hand tracking using MediaPipe and OpenCV
- Real-time digit prediction using the trained CNN

[View Task 3 documentation and source code](Task3_Handwritten_Recognition/)

## Technologies

- Python
- TensorFlow and Keras
- NumPy and Pandas
- scikit-learn
- Librosa
- OpenCV and MediaPipe
- Matplotlib
- Git and GitHub

## Repository Structure

```text
CodeAlpha_Tasks/
├── README.md
├── .gitignore
├── Task2_Emotion_Recognition/
│   ├── README.md
│   ├── src/
│   ├── models/
│   └── results/
└── Task3_Handwritten_Recognition/
    ├── README.md
    ├── requirements.txt
    ├── requirements-airwriting.txt
    ├── models/
    ├── samples/
    └── src/
        ├── train_model.py
        ├── predict_digit.py
        └── air_writing_recognition.py
```

## Notes

- Python virtual environments and raw datasets are excluded from version control.
- Refer to each project's README for setup instructions and dependencies.
- Model evaluation results are reported transparently, including known limitations.
- Task 3 includes an optional webcam-based air-writing enhancement in addition to image-based handwritten digit recognition.

## Author

Developed as part of my CodeAlpha Machine Learning Internship.
