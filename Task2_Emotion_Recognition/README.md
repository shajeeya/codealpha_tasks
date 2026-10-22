
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
│   ├── extract_features.py
│   ├── extract_audio_features.py
│   ├── prepare_dataset.py
│   ├── prepare_enhanced_dataset.py
│   ├── train_model.py
│   ├── tune_model.py
│   ├── compare_features.py
│   ├── compare_classifiers.py
│   ├── predict_emotion.py
│   └── test_features.py
├── models/
│   └── emotion_classifier.joblib
├── results/
│   ├── evaluation.json
│   ├── svm_tuning_results.json
│   ├── feature_comparison.json
│   ├── classifier_comparison.json
│   ├── confusion_matrix.png
│   └── out_of_fold_predictions.npz
├── data/
│   └── processed/
├── README.md
└── .gitignore
```

The raw audio dataset, virtual environment, and regenerable processed feature files are excluded from Git.

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

## Evaluation Results

The baseline RBF Support Vector Machine (SVM) produced the following six-fold actor-grouped cross-validation results:

| Metric | Result |
|---|---:|
| Accuracy | 44.44% |
| Balanced accuracy | 43.36% |
| Mean fold macro F1-score | 0.4206 |
| Accuracy standard deviation | 0.0558 |
| Balanced accuracy standard deviation | 0.0458 |
| Macro F1 standard deviation | 0.0409 |

The RBF SVM outperformed the tested Random Forest, Extra Trees, Linear SVM, and Logistic Regression configurations in the recorded comparisons.

The enhanced 174-feature representation did not improve performance over the original 80-feature baseline in this experiment.

See the `results/` directory for detailed metrics, classifier comparisons, tuning results, out-of-fold predictions, and the confusion matrix.

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
