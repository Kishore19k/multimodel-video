import streamlit as st
import numpy as np
import cv2
import os

from tensorflow.keras.models import load_model

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

import matplotlib.pyplot as plt

os.makedirs("temp", exist_ok=True)

SEQUENCE_LENGTH = 10
IMG_SIZE = 224

st.set_page_config(
    page_title="Video Accident Detection",
    layout="wide"
)

st.title(
    "Video Accident Detection System"
)

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():

    resnet = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224,224,3)
    )

    resnet.trainable = False

    video_model = load_model(
        "models/video_accident_model.h5"
    )

    return resnet, video_model


resnet, video_model = load_models()

# -----------------------------
# Frame Extraction
# -----------------------------
def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        return None

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH,
        dtype=int
    )

    idx = 0

    selected = set(frame_indices)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if idx in selected:

            frame = cv2.resize(
                frame,
                (IMG_SIZE, IMG_SIZE)
            )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(frame)

        idx += 1

    cap.release()

    if len(frames) != SEQUENCE_LENGTH:
        return None

    return np.array(frames)

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(frames):

    frames = preprocess_input(frames)

    features = resnet.predict(
        frames,
        verbose=0
    )

    return features

# -----------------------------
# Upload
# -----------------------------
video_file = st.file_uploader(
    "Upload Video",
    type=["mp4","avi","mov"]
)

# -----------------------------
# Predict
# -----------------------------
if st.button("Analyze Video"):

    if video_file is None:

        st.error(
            "Please upload a video."
        )

        st.stop()

    video_path = "temp/test_video.mp4"

    with open(video_path, "wb") as f:

        f.write(
            video_file.read()
        )

    with st.spinner(
        "Processing Video..."
    ):

        frames = extract_frames(
            video_path
        )

        if frames is None:

            st.error(
                "Could not extract frames."
            )

            st.stop()

        features = extract_features(
            frames
        )

        features = np.expand_dims(
            features,
            axis=0
        )

        prediction = video_model.predict(
            features,
            verbose=0
        )

        prob = float(
            prediction[0][0]
        )

        if prob > 0.5:

            label = "ACCIDENT"

        else:

            label = "NORMAL"

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Accident Probability",
            round(prob,4)
        )

    with col2:

        st.metric(
            "Prediction",
            label
        )

    st.subheader(
        "Frame Used For Prediction"
    )

    st.image(
        frames[4],
        caption=f"{label} | Probability: {prob:.4f}"
    )
