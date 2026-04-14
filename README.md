# Voice Stress Detection Project

## Overview
This project aims to detect voice stress using Mel Frequency Cepstral Coefficients (MFCC) and Gated Recurrent Units (GRU). The objective is to analyze audio data and identify stress levels based on vocal characteristics.

## Key Technologies
- **Mel Frequency Cepstral Coefficients (MFCC)**: A feature extraction technique widely used in speech and audio processing.
- **Gated Recurrent Unit (GRU)**: A type of recurrent neural network architecture used for time series data analysis.

## Project Structure
- `data/`: Contains raw audio files and preprocessed data.
- `models/`: Holds the trained models and model training scripts.
- `notebooks/`: Jupyter notebooks for data exploration and visualization.
- `src/`: The main source code for training and inference.

## Installation
To set up the project, clone the repository and install the necessary packages:

```bash
git clone https://github.com/JoanneKoshy/Voice-stress-using-MFCC-and-GRU.git
cd Voice-stress-using-MFCC-and-GRU
pip install -r requirements.txt


---

## 📦 Tech Stack

* **Python**
* **Librosa** – Audio processing
* **NumPy** – Numerical computations
* **TensorFlow / Keras** – Deep Learning model
* **Streamlit** – Web app interface
* **Scikit-learn** – Data preprocessing

---

## 🎧 Dataset

Used:

* RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Contains:

* Speech with multiple emotions:

  * Neutral, Calm, Happy, Sad, Angry, Fear, Disgust, Surprise

---

## 🧩 Data Processing

### 1. Audio Loading

* Audio files are loaded using `librosa`

### 2. Feature Extraction

* MFCC (Mel-Frequency Cepstral Coefficients) extracted
* Shape standardized to `(40, 100)`

### 3. Label Mapping

Emotion → Stress:

| Emotion              | Stress Level |
| -------------------- | ------------ |
| Neutral, Calm, Happy | Calm         |
| Sad, Fear            | Stressed     |
| Angry, Disgust       | High Stress  |

---

## 🤖 Model Training (Kaggle)

Model trained using GPU on Kaggle.

### Steps:

1. Load dataset
2. Extract MFCC features
3. Encode labels
4. Train GRU model

### Model Architecture:

* GRU Layer (128 units)
* Dropout (0.3)
* Dense Layer (64 units)
* Output Layer (Softmax)

### Loss Function:

* Categorical Crossentropy

### Optimizer:

* Adam

### Output:

* Saved trained model as:
saved_model.h5


---

## 💻 Application Development (VS Code)

After training:

* Model downloaded from Kaggle
* Integrated into a Streamlit app

---

## 🎤 Features of the App

* Upload `.wav` audio file
* Audio playback
* Stress prediction output
* Confidence percentage display

---

## 📂 Project Structure

