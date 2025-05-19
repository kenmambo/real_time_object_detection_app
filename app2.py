import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import time

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    model_path = "yolov8s.pt"
    if not os.path.exists(model_path):
        model = YOLO("yolov8s")  # This will download the model if not present
    else:
        model = YOLO(model_path)
    return model

# Function to perform object detection on an image
def detect_objects(image, model, conf=0.25):
    # Perform inference
    results = model(image, conf=conf)
    
    # Process results
    processed_img = results[0].plot()
    
    return processed_img, results[0]

def main():
    st.title("Real-Time Object Detection App")
    
    # Sidebar for app options
    st.sidebar.title("Settings")
    
    # Model confidence
    confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
    
    # Load the YOLO model
    model = load_model()
    
    # App modes
    app_mode = st.sidebar.selectbox("Choose the app mode", ["About", "Image", "Video", "Webcam"])
    
    if app_mode == "About":
        st.markdown("## About")
        st.markdown("This application demonstrates real-time object detection using YOLOv8 and Streamlit.")
        st.markdown("### Instructions:")
        st.markdown("- Select 'Image' to upload and analyze a single image")
        st.markdown("- Select 'Video' to upload and analyze a video file")
        st.markdown("- Select 'Webcam' to use your webcam for real-time detection")
    
    elif app_mode == "Image":
        st.markdown("## Image Detection")
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if image_file is not None:
            # Read the image
            image = np.array(bytearray(image_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Display original image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Detect objects when user clicks the button
            if st.button("Detect Objects"):
                processed_image, results = detect_objects(image, model, confidence)
                
                # Display processed image
                st.image(processed_image, caption="Detection Results", use_column_width=True)
                
                # Display detection information
                boxes = results.boxes.cpu().numpy()
                if len(boxes) > 0:
                    st.write(f"Detected {len(boxes)} objects:")
                    
                    # Get class names
                    class_names = results.names
                    
                    # Create a table of results
                    data = []
                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        conf = box.conf[0]
                        label = class_names[cls_id]
                        data.append([i+1, label, f"{conf:.2f}"])
                    
                    st.table({"#": [d[0] for d in data], 
                              "Object": [d[1] for d in data], 
                              "Confidence": [d[2] for d in data]})
                else:
                    st.write("No objects detected")
    
    elif app_mode == "Video":
        st.markdown("## Video Detection")
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        
        if video_file is not None:
            # Save the uploaded video to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(video_file.read())
            
            # Process the video
            st.video(temp_file.name)
            
            if st.button("Process Video"):
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Open video capture
                cap = cv2.VideoCapture(temp_file.name)
                
                if not cap.isOpened():
                    st.error(f"Error: Could not open video file")
                else:
                    # Get video properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    st.write(f"Video properties: {width}x{height} at {fps} FPS, {total_frames} frames")
                    
                    # Create a temporary file for the processed video
                    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    output_path = output_file.name
                    output_file.close()
                    
                    # Create video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    # Process video frames
                    frame_count = 0
                    start_time = time.time()
                    
                    try:
                        while True:
                            # Read frame
                            ret, frame = cap.read()
                            
                            if not ret:
                                break
                            
                            # Perform object detection
                            processed_frame, _ = detect_objects(frame, model, confidence)
                            
                            # Write frame to output video
                            out.write(processed_frame)
                            
                            # Update progress
                            frame_count += 1
                            progress = int(frame_count / total_frames * 100)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress}%)")
                    
                    finally:
                        # Release resources
                        cap.release()
                        out.release()
                        
                        elapsed_time = time.time() - start_time
                        st.write(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
                        st.write(f"Average FPS: {frame_count / elapsed_time:.2f}")
                        
                        # Display the processed video
                        st.video(output_path)
                
                # Clean up temporary files
                os.unlink(temp_file.name)
    
    elif app_mode == "Webcam":
        st.markdown("## Webcam Detection")
        st.write("Click the button below to start the webcam feed")
        
        # Use OpenCV to capture from device 0
        if st.button("Start Webcam"):
            # Create a placeholder for the webcam feed
            video_placeholder = st.empty()
            stop_button_pressed = st.button("Stop")
            
            # Open webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open webcam")
            else:
                # Process webcam feed
                frame_count = 0
                start_time = time.time()
                
                while not stop_button_pressed:
                    # Read frame
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Failed to capture frame from webcam")
                        break
                    
                    # Perform object detection
                    processed_frame, _ = detect_objects(frame, model, confidence)
                    
                    # Calculate FPS
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / elapsed_time
                    
                    # Add FPS information to the frame
                    cv2.putText(processed_frame, f"FPS: {current_fps:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Convert to RGB for display
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display the processed frame
                    video_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Check if stop button was pressed
                    if stop_button_pressed:
                        break
                
                # Release resources
                cap.release()
                st.write(f"Webcam session ended. Processed {frame_count} frames at {frame_count / elapsed_time:.2f} FPS")

if __name__ == "__main__":
    main()