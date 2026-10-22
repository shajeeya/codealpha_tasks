
# Speech Emotion Recognition

A machine learning project that predicts emotions from speech recordings using audio feature extraction and supervised classification.

## Project Overview

This project uses the RAVDESS emotional speech dataset to classify speech into eight emotion categories. Audio recordings are converted into numerical features, and machine learning classifiers are evaluated using actor-independent cross-validation.

### Supported Emotions

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Technologies Used

- TensorFlow
- Keras
- Python
- NumPy
- Pandas
- Librosa
- scikit-learn
- Matplotlib
- SoundFile
- Joblib
- Pytest-compatible unit tests

## Project Structure

```text
Task2_Emotion_Recognition/
├── src/
│   ├── compare_classifiers.py
│   ├── compare_features.py
│   ├── extract_audio_features.py
│   ├── extract_features.py
│   ├── predict_emotion.py
│   ├── predict_emotion_cnn.py
│   ├── prepare_cnn_dataset.py
│   ├── prepare_cnn_dataset_resampled.py
│   ├── prepare_dataset.py
│   ├── prepare_enhanced_dataset.py
│   ├── test_features.py
│   ├── train_cnn_model.py
│   ├── train_cnn_model_resampled.py
│   ├── train_cnn_model_weighted.py
│   ├── train_model.py
│   └── tune_model.py
├── models/
│   ├── emotion_classifier.joblib
│   ├── emotion_cnn.keras
│   ├── emotion_cnn_normalization.npz
│   ├── emotion_cnn_resampled.keras
│   ├── emotion_cnn_resampled_normalization.npz
│   ├── emotion_cnn_weighted.keras
│   └── emotion_cnn_weighted_normalization.npz
├── results/
│   ├── classifier_comparison.json
│   ├── cnn_evaluation.json
│   ├── cnn_evaluation_resampled.json
│   ├── cnn_evaluation_weighted.json
│   ├── cnn_out_of_fold_predictions.npz
│   ├── cnn_out_of_fold_predictions_resampled.npz
│   ├── cnn_out_of_fold_predictions_weighted.npz
│   ├── confusion_matrix.png
│   ├── evaluation.json
│   ├── feature_comparison.json
│   ├── out_of_fold_predictions.npz
│   └── svm_tuning_results.json
├── data/
│   └── processed/
├── README.md
└── .gitignore
```

The raw audio dataset, virtual environments, and regenerable processed feature files are excluded from Git. Model files and evaluation artifacts are retained separately for prediction and analysis.

## Dataset

This project uses the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), specifically its speech audio recordings.

- 1,440 WAV recordings
- 24 actors
- 8 emotion categories
- 80 MFCC-based features per recording in the baseline model

Obtain the speech audio dataset from the official RAVDESS dataset source and extract the actor folders into this project directory.

Dataset source: https://zenodo.org/record/1188976

The dataset is not included in this repository. Please review its license and terms before using or redistributing it.

## Installation

Python 3.12 was used for this project.

From the repository root, create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install numpy pandas scikit-learn librosa scipy matplotlib soundfile joblib pytest
```

The packages above support the baseline SVM workflow. The CNN training and prediction workflows additionally require TensorFlow/Keras. Install TensorFlow in a compatible Python environment and verify the installation before running the CNN scripts.

The CNN predictor also requires Librosa and NumPy versions compatible with the saved model and the training environment.

## Feature Extraction and Dataset Preparation

Prepare the baseline MFCC dataset:

```powershell
python .\Task2_Emotion_Recognition\src\prepare_dataset.py
```

Prepare the enhanced feature dataset:

```powershell
python .\Task2_Emotion_Recognition\src\prepare_enhanced_dataset.py
```

These scripts generate processed feature data from the extracted audio recordings.

## Model Training and Evaluation

Train and evaluate the baseline classifiers:

```powershell
python .\Task2_Emotion_Recognition\src\train_model.py
```

Tune the SVM hyperparameters:

```powershell
python .\Task2_Emotion_Recognition\src\tune_model.py
```

Compare original and enhanced features:

```powershell
python .\Task2_Emotion_Recognition\src\compare_features.py
```

Compare classifiers:

```powershell
python .\Task2_Emotion_Recognition\src\compare_classifiers.py
```

The evaluation uses six-fold GroupKFold cross-validation, grouping recordings by actor to evaluate generalization to unseen actors.

## Predict an Emotion

Once the trained model is available, run:

```powershell
python .\Task2_Emotion_Recognition\src\predict_emotion.py ".\Task2_Emotion_Recognition\Actor_01\03-01-01-01-01-01-01.wav"
```

The script accepts a WAV file path and prints the predicted emotion.

Example output:

```text
Predicted emotion: Neutral
```

Actual predictions may differ depending on the input recording.

## CNN-Based Emotion Prediction

A separate CNN model is available for predicting emotions from WAV audio recordings. It uses 40 MFCC coefficients resampled to 174 frames, with normalization values saved during final model training.

The CNN model and its normalization parameters are stored in:

- `models/emotion_cnn_resampled.keras`
- `models/emotion_cnn_resampled_normalization.npz`

The CNN predictor requires TensorFlow/Keras and the audio-processing dependencies used during training. Run it using a Python environment in which these dependencies are installed and compatible with the saved model.

From the repository root, run:

```powershell
python .\Task2_Emotion_Recognition\src\predict_emotion_cnn.py ".\Task2_Emotion_Recognition\Actor_01\03-01-01-01-01-01-01.wav"
```

The script prints the predicted emotion and its model probability. This probability is not a calibrated confidence score.

Example output:

```text
Predicted emotion: Neutral
Model probability: 27.84%
Note: This probability is not a calibrated confidence score.
```

The example output is from one test recording; predictions can differ for other audio files.

## Evaluation Results

The baseline RBF Support Vector Machine (SVM) produced the following six-fold actor-grouped cross-validation results:

| Metric | Result |
|---|---:|
| Accuracy | 44.44% |
| Balanced accuracy | 43.36% |
| Mean fold macro F1-score | 0.4206 |
| Accuracy standard deviation | 5.58 percentage points |
| Balanced accuracy standard deviation | 4.58 percentage points |
| Macro F1 standard deviation | 0.0409 |

The RBF SVM outperformed the tested Random Forest, Extra Trees, Linear SVM, and Logistic Regression configurations in the recorded comparisons.

The enhanced 174-feature representation did not improve performance over the original 80-feature baseline in this experiment.

See the `results/` directory for detailed metrics, classifier comparisons, tuning results, out-of-fold predictions, and the confusion matrix.

## CNN Evaluation

The resampled CNN model was evaluated using five-fold GroupKFold cross-validation, with recordings grouped by actor.

| Metric | Resampled CNN |
|---|---:|
| Accuracy | 47.50% |
| Balanced accuracy | 44.53% |
| Macro F1-score | 0.4140 |
| Mean fold accuracy | 47.42% ± 6.84 percentage points |
| Mean fold macro F1-score | 0.4049 ± 0.0739 |

A separate experiment using class-weighted training achieved 40.63% accuracy, 40.04% balanced accuracy, and a macro F1-score of 0.3728. Although class weighting improved detection of the neutral class, it reduced overall performance in this experiment.

The unweighted resampled CNN performed poorly on the neutral class, achieving a neutral-class F1-score of 0.000 in its out-of-fold evaluation. This highlights the need for further improvement in class-specific performance.

**Comparison note:** The SVM uses six-fold actor-grouped cross-validation, while the CNN uses five-fold actor-grouped cross-validation. Therefore, their results are not a perfectly matched comparison.

## Testing

Run the feature extraction tests:

```powershell
python -m pytest .\Task2_Emotion_Recognition\src\test_features.py -v
```

The feature extraction tests cover output dimensions, custom MFCC counts, missing files, and invalid MFCC settings.

## Limitations

- Overall classification performance is modest and requires further improvement.
- Some emotions have similar acoustic characteristics and are frequently confused.
- Neutral emotion has fewer examples than the other classes in the dataset.
- Performance on acted RAVDESS recordings may not generalize to natural speech, different languages, microphones, or noisy environments.
- A predicted emotion is a model estimate, not a reliable determination of a person's actual emotional state.

## Future Improvements

- Evaluate additional acoustic representations and model architectures.
- Improve discrimination between acoustically similar emotions.
- Test on independent datasets and real-world recordings.
- Build a simple user interface for audio upload and prediction.

## Acknowledgements

The project uses the RAVDESS speech dataset for research and educational experimentation.
