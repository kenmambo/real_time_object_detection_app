# Real-Time Object Detection with YOLOv8 and Streamlit

This project implements a real-time object detection system using the YOLOv8 model and a Streamlit web interface (`app2.py`). It allows users to perform object detection on various input sources.

## Key Features (app2.py)

*   **Real-time object detection from webcam:** Utilizes your webcam for live object detection.
*   **Object detection from uploaded video files:** Process video files (MP4, AVI, MOV) to detect objects within them.
*   **Object detection from uploaded image files:** Analyze uploaded images (JPEG, PNG) for object detection.
*   **Adjustable confidence threshold:** Allows users to set the confidence level for detections via a slider.
*   **Display of bounding boxes, class labels, and confidence scores:** Clearly visualizes detected objects with their respective information.
*   **Information about detected objects:** Provides details such as the count of detected objects and a table listing them.

## Getting Started

### Prerequisites

*   Python 3.x
*   `uv` package manager (can be installed via pip: `pip install uv`)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://your-repository-url/project-name.git
    ```
2.  Navigate into the project directory:
    ```bash
    cd project-name
    ```
3.  Install dependencies using `uv`:
    ```bash
    uv pip install streamlit opencv-python ultralytics numpy
    ```

### Running the Application

Run the main application:
```bash
streamlit run app2.py
```
(`app.py` is also available, providing a more basic interface for webcam and video detection.)

### How to Use the Application

Once the application is running (`streamlit run app2.py`):

1.  The sidebar will present different modes: 'About', 'Image', 'Video', and 'Webcam'.
2.  **About**: Shows a brief description of the application.
3.  **Image**: Allows you to upload an image file (JPEG, PNG). Click 'Detect Objects' to see the results.
4.  **Video**: Allows you to upload a video file (MP4, AVI, MOV). Click 'Process Video' to see the detections. The processed video with bounding boxes will be displayed.
5.  **Webcam**: Click 'Start Webcam' to begin real-time detection using your computer's webcam. Click 'Stop' to end the stream.
6.  **Confidence Threshold**: Adjust the slider in the sidebar to change the detection sensitivity.

## Project Structure

*   `app2.py`: The main Streamlit application for object detection, offering image, video, and webcam input, along with advanced features.
*   `app.py`: An alternative, simpler Streamlit application for webcam and video detection.
*   `yolov8s.pt`: The pre-trained YOLOv8 model weights used for object detection.
*   `detection_app.ipynb`: A Jupyter Notebook for development and experimentation with the detection model.
*   `pyproject.toml` / `uv.lock`: Files defining project dependencies and versions.
*   `README.md`: This file, providing information about the project.

## Model

This project utilizes the YOLOv8 model from Ultralytics for object detection. The specific pre-trained model weights file included in this repository is `yolov8s.pt`, which corresponds to the YOLOv8 small variant. This model is known for its balance of speed and accuracy, making it suitable for real-time applications.

For more information about YOLOv8 and the Ultralytics framework, you can visit their official GitHub repository: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
