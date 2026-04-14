import streamlit as st
import numpy as np
import librosa
import tempfile
from tensorflow.keras.models import load_model

# -------------------------------
# Load Model
# -------------------------------
model = load_model("saved_models/stress_model.h5")

# Label mapping (must match training)
labels = ["calm", "high_stress", "stressed"]

# -------------------------------
# Feature Extraction
# -------------------------------
def extract_features(file_path, max_pad_len=100):
    try:
        audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# -------------------------------
# UI
# -------------------------------
st.title("🎧 Voice Stress Detection App")
st.write("Upload a .wav audio file (3–10 sec) to analyze stress level")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    st.write("Processing audio... ⏳")

    features = extract_features(temp_path)

    if features is not None:
        features = np.expand_dims(features, axis=0)

        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_label = labels[predicted_index]
        confidence = np.max(prediction) * 100

        # -------------------------------
        # Display Results
        # -------------------------------
        st.subheader("🧠 Prediction Result")

        if predicted_label == "calm":
            st.success(f"😌 Calm ({confidence:.2f}%)")
        elif predicted_label == "stressed":
            st.warning(f"😟 Stressed ({confidence:.2f}%)")
        else:
            st.error(f"⚠️ High Stress ({confidence:.2f}%)")

        # Show raw probabilities
        st.write("### 🔍 Confidence Breakdown")
        for i, label in enumerate(labels):
            st.write(f"{label}: {prediction[0][i]*100:.2f}%")