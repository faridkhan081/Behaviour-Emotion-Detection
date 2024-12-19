import cv2
import mediapipe as mp
import torch
import numpy as np
import streamlit as st
from PIL import Image
from utils import norm_coordinates, get_box, display_EMO_PRED
from models import ResNet50, LSTMPyTorch
from preprocessing import pth_processing
import os
import time

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Load models
name_backbone_model = 'FER_static_ResNet50_AffectNet.pt'
name_LSTM_model = 'FER_dinamic_LSTM_Aff-Wild2.pt'

if not os.path.exists(name_backbone_model) or not os.path.exists(name_LSTM_model):
    raise RuntimeError("Model files are missing!")

pth_backbone_model = ResNet50(7, channels=3)
pth_backbone_model.load_state_dict(torch.load(name_backbone_model, map_location=torch.device('cpu')))
pth_backbone_model.eval()

pth_LSTM_model = LSTMPyTorch()
pth_LSTM_model.load_state_dict(torch.load(name_LSTM_model, map_location=torch.device('cpu')))
pth_LSTM_model.eval()

# Emotion mapping
DICT_EMO = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}

def process_frame(frame):
    """Processes a single frame for emotion recognition."""
    try:
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(frame_rgb)
        lstm_features = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                startX, startY, endX, endY = get_box(face_landmarks, w, h)
                face_roi = frame[startY:endY, startX:endX]

                if face_roi.size == 0:
                    continue

                face_roi = pth_processing(Image.fromarray(face_roi))
                features = pth_backbone_model.extract_features(face_roi).detach().numpy()

                if not lstm_features:
                    lstm_features = [features] * 10
                else:
                    lstm_features = lstm_features[1:] + [features]

                lstm_input = torch.from_numpy(np.vstack(lstm_features)).unsqueeze(0)
                output = pth_LSTM_model(lstm_input).detach().numpy()
                emotion_label = DICT_EMO[np.argmax(output)]
                frame = display_EMO_PRED(frame, (startX, startY, endX, endY), f"{emotion_label}", 3)

        return frame
    except Exception as e:
        print(f"Error in process_frame: {e}")
        return frame

# Set up the Streamlit interface
st.title("Emotion Recognition App")

# Video processing mode
st.header("Process Uploaded Video")
uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov", "mkv"])

# Check if a video has been uploaded
if uploaded_file:
    # Save the uploaded file temporarily
    temp_video_path = f"temp_{uploaded_file.name}"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    # Start processing video
    st.info("Processing video. This might take some time...")
    cap = cv2.VideoCapture(temp_video_path)

    # Video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = f"processed_{uploaded_file.name}"
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize progress bar
    

    processed_frames = 0  # To track progress
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        out.write(processed_frame)

        # Update progress bar
       

    cap.release()
    out.release()

    # Clean up the temporary file after processing is done
    os.remove(temp_video_path)

    st.success("Video processing completed!")

    # Provide download link
    with open(output_path, "rb") as f:
        st.download_button("Download Processed Video", f, file_name=output_path)

    # Clean up the output video after download is provided
    os.remove(output_path)


    
    # Rerun the app to reset the state after video download
 
else:
    # If no video is uploaded, prompt the user to upload one
    st.info("Waiting for a video upload to begin processing.")
