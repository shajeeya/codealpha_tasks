# Handwritten Digit Recognition Using CNN

## 📌 Project Overview

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits from 0 to 9 using the MNIST dataset. It uses TensorFlow and Keras for model development, training, evaluation, and prediction on custom handwritten digit images.

The project also includes a **live air-writing digit recognition feature** that uses webcam-based hand tracking to draw digits in the air and predict them using the trained CNN model.

## 🎯 Objective

To develop a deep learning model that can identify handwritten digits using image preprocessing and a CNN architecture, with an interactive air-writing interface as an additional enhancement.

## 📂 Dataset

- **Dataset:** MNIST
- **Training images:** 60,000
- **Test images:** 10,000
- **Image dimensions:** 28 × 28 pixels
- **Number of classes:** 10 (digits 0–9)

The dataset is loaded using `tf.keras.datasets.mnist`.

## 🛠️ Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Pillow (PIL)
- Matplotlib
- OpenCV
- MediaPipe

## 🧠 Model Architecture

The CNN consists of:

1. Input layer — 28 × 28 grayscale image
2. Convolutional layer — 32 filters, ReLU activation
3. Max pooling layer
4. Convolutional layer — 64 filters, ReLU activation
5. Max pooling layer
6. Flatten layer
7. Dense layer — 128 neurons, ReLU activation
8. Dropout layer — 30%
9. Output layer — 10 neurons, softmax activation

The model uses the Adam optimizer and sparse categorical cross-entropy loss.

## 📊 Results

| Metric | Result |
|---|---:|
| Test accuracy | 98.94% |
| Test loss | 0.0326 |
| Epochs completed | 8 |
| Custom digit tested | 7 |
| Custom digit prediction | 7 |
| Custom prediction confidence | 63.57% |

The test accuracy is measured on the held-out MNIST test dataset. The custom prediction confidence applies only to the sample image `samples/digit7.png`.

The air-writing feature was also tested interactively with digits 0–9, and the predictions were correct during the author's tests. This is a manual functional test, not a separate benchmark accuracy measurement.

## ✍️ Live Air-Writing Digit Recognition

The air-writing feature allows users to draw digits in the air using their index finger in front of a webcam.

### How It Works

1. OpenCV captures live video from the webcam.
2. MediaPipe detects hand landmarks and tracks the index fingertip.
3. Finger movement is drawn on a digital canvas.
4. The drawing is preprocessed into a 28 × 28 grayscale image.
5. The trained CNN predicts the digit from 0 to 9.
6. The predicted digit and confidence score are displayed.

### Controls

| Key | Action |
|---|---|
| `C` | Clear the drawing canvas |
| `P` | Predict the drawn digit |
| `Q` | Quit the application |

### Run the Air-Writing Application

From the repository root, run the script using the dedicated air-writing environment:

```powershell
.\.venv_airwriting\Scripts\python.exe Task3_Handwritten_Recognition\src\air_writing_recognition.py
```

Ensure that your webcam is available and that the required libraries are installed in `.venv_airwriting`.

## 📁 Project Structure

```text
Task3_Handwritten_Recognition/
├── models/
│   └── handwritten_digit_cnn.keras
├── results/
│   ├── evaluation.json
│   └── training_history.png
├── samples/
│   └── digit7.png
├── src/
│   ├── air_writing_recognition.py
│   ├── predict_digit.py
│   └── train_model.py
└── README.md
```

The `.venv_airwriting` environment is located at the repository root and should not be committed to GitHub.

## ⚙️ Installation and Setup

Python 3.12 is used for this project.

### Original CNN Environment

Install the dependencies listed in the Task 3 requirements file:

```powershell
python -m pip install -r Task3_Handwritten_Recognition/requirements.txt
```

### Air-Writing Environment

The air-writing feature was tested with the following package versions:

- TensorFlow 2.18.0
- MediaPipe 0.10.21
- OpenCV 4.11.0
- NumPy 1.26.4

The environment was created separately to keep its dependencies isolated from the original training environment.

## ▶️ Train the Model

Run the training script from the repository root:

```powershell
python Task3_Handwritten_Recognition/src/train_model.py
```

The script downloads the MNIST dataset if needed, trains the CNN, evaluates the model, and saves the trained model and evaluation results.

## 🔍 Predict a Handwritten Digit

After training, provide the path to a PNG or other supported image file:

```powershell
python Task3_Handwritten_Recognition/src/predict_digit.py Task3_Handwritten_Recognition/samples/digit7.png
```

The prediction script preprocesses the image and displays the predicted digit, confidence, and three highest-scoring predictions.

## 📈 Training Visualization

The training and validation accuracy plot is saved to:

`results/training_history.png`

## 🚀 Future Enhancements

- Test the model on more handwritten digit samples.
- Improve image preprocessing for different handwriting styles.
- Build a simple graphical user interface for digit recognition.
- Extend the project to recognize handwritten letters using the EMNIST dataset.
- Explore recognition of complete words and sequences.

## 👩‍💻 Author

Developed as part of the CodeAlpha Machine Learning Internship.
