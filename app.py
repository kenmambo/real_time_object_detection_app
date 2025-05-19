import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

st.title("📦 Real-Time Object Detection with YOLOv8")
st.subheader("Detect objects from webcam or video")

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# Sidebar options
source = st.sidebar.selectbox("Select source", ["Webcam", "Upload Video"])

if source == "Webcam":
    st.write("Starting webcam... (Use Ctrl+C to stop)")
    cap = cv2.VideoCapture(0)

elif source == "Upload Video":
    video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi"])
    if video_file:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture("temp_video.mp4")
    else:
        st.warning("Please upload a video file.")
        st.stop()

# Placeholder for frames
frame_placeholder = st.empty()
stop_button = st.button("Stop")

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Plot results on frame
    annotated_frame = results[0].plot()

    # Convert BGR to RGB
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Show frame
    frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

cap.release()
cv2.destroyAllWindows()
st.write("Detection stopped.")