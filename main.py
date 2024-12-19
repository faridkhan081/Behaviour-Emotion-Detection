import asyncio
import cv2
import mediapipe as mp
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import os
import numpy as np
import traceback
from utils import norm_coordinates, get_box, display_EMO_PRED
from models import ResNet50, LSTMPyTorch
from preprocessing import pth_processing

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="Emotion Recognition API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, you can restrict this to the Angular app's domain
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Load models (same as before)
name_backbone_model = 'FER_static_ResNet50_AffectNet.pt'
name_LSTM_model = 'FER_dinamic_LSTM_Aff-Wild2.pt'

if not os.path.exists(name_backbone_model) or not os.path.exists(name_LSTM_model):
    raise RuntimeError("Model files are missing!")

# Load models at startup (same as before)
pth_backbone_model = ResNet50(7, channels=3)
pth_backbone_model.load_state_dict(torch.load(name_backbone_model, map_location=torch.device('cpu')))
pth_backbone_model.eval()

pth_LSTM_model = LSTMPyTorch()
pth_LSTM_model.load_state_dict(torch.load(name_LSTM_model, map_location=torch.device('cpu')))
pth_LSTM_model.eval()

# Emotion mapping (same as before)
DICT_EMO = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}

async def process_frame(frame):
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

@app.get("/process_webcam")
async def process_webcam():
    """Streams processed video from the webcam in real-time."""
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise HTTPException(status_code=500, detail="Could not open webcam")

    async def generate_frames():
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Process the frame for emotion detection
            processed_frame = process_frame(frame)
            
            # Encode frame to JPEG format
            _, buffer = cv2.imencode(".jpg", processed_frame)
            # Yield the frame data to stream it back to the client
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            
            # Allow asyncio to run other tasks, improving responsiveness
            await asyncio.sleep(0)

    # Return StreamingResponse which continuously sends frames to the client
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")


@app.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    """Processes an uploaded video file and returns the processed video for download."""
    try:
        # Save the uploaded video to a temporary file
        temp_filename = f"/tmp/{file.filename}"
        processed_filename = f"/tmp/processed_{file.filename}"

        with open(temp_filename, "wb") as temp_file:
            temp_file.write(await file.read())

        
        video = cv2.VideoCapture(temp_filename)
        if not video.isOpened():
            raise HTTPException(status_code=400, detail="Invalid video file")

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Create a VideoWriter for the processed video
        out = cv2.VideoWriter(processed_filename, fourcc, fps, (width, height))

        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Process the frame
            processed_frame = process_frame(frame)
            out.write(processed_frame)

        # Release resources
        video.release()
        out.release()

        # Return the processed video as a downloadable file
        def iterfile():
            with open(processed_filename, "rb") as processed_file:
                yield from processed_file

        return StreamingResponse(iterfile(), media_type="video/mp4", headers={
            "Content-Disposition": f"attachment; filename=processed_{file.filename}"
        })

    except Exception as e:
        print(f"Error processing video: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running."""
    return {"status": "ok"}

#if __name__ == "__main__":
 #   import uvicorn
  #  print("Starting FastAPI server...")
   # uvicorn.run("main:app", host="127.0.0.1", port=8300, reload=True)
