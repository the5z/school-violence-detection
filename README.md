# School Violence Detection System Using MobileNetV2 and YOLOv8

This repository contains a school violence detection prototype built with **MobileNetV2**, **YOLOv8**, **YOLOv8-pose**, **TensorFlow/Keras**, **OpenCV**, and **Streamlit**.

The system supports both live-camera monitoring and uploaded-video analysis. It detects people in video frames, classifies possible violence, applies temporal verification, provides a hands-up auxiliary warning, and stores evidence images when alerts occur.

This project was developed for academic purposes as a lightweight decision-support system for school violence monitoring.

---

## Authors

- **Lai Minh Hiep**
- **Nguyen Quang Duy**

Faculty of Information Technology  
Dai Nam University  
Hanoi, Vietnam

---

## Main Features

- Fight / Non-Fight classification using a trained MobileNetV2 model.
- YOLOv8n-based person detection.
- YOLOv8n-pose-based hands-up warning.
- Live camera monitoring with Streamlit WebRTC.
- Uploaded-video processing.
- Configurable violence threshold.
- Configurable number of consecutive frames before generating an alert.
- Temporal smoothing to reduce unstable frame-level predictions.
- Minimum-person rule before considering violence.
- Visual overlays showing person count, Fight score, hands-up status, and alert state.
- Evidence storage for alert frames and cropped detected persons.
- Download support for processed videos and saved evidence images.

---

## System Overview

The proposed system consists of two main stages.

### Offline Training Stage

Frames are extracted from the RWF-2000 dataset and used to train a binary violence classification model.

The classifier is based on MobileNetV2 with ImageNet pre-trained weights. The final trained checkpoint is saved as:

```text
best_model.keras
```

### Online Monitoring Stage

During inference, the Streamlit application processes either live-camera frames or uploaded videos. Each frame goes through the following pipeline:

```text
Input Video / Camera
        ↓
YOLOv8 Person Detection
        ↓
Person Crop Extraction
        ↓
MobileNetV2 Violence Classification
        ↓
YOLOv8-Pose Hands-Up Analysis
        ↓
Temporal Smoothing
        ↓
Consecutive-Frame Verification
        ↓
Alert Generation and Evidence Storage
```

The system does not trigger an alert from a single uncertain frame. Instead, it uses smoothing and consecutive-frame verification to reduce false alarms.

---

## Project Structure

```text
TGMT/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── best_model.keras
├── yolov8n.pt
├── yolov8n-pose.pt
│
├── test_images/
│   ├── fight.jpg
│   └── nonfight.jpg
│
├── captures/
│   └── .gitkeep
│
└── outputs/
    └── .gitkeep
```

---

## File Description

| File / Folder | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `best_model.keras` | Trained MobileNetV2 violence classification model |
| `yolov8n.pt` | YOLOv8 model for person detection |
| `yolov8n-pose.pt` | YOLOv8 pose model for hands-up analysis |
| `test_images/` | Sample images for quick testing |
| `captures/` | Stores saved evidence images |
| `outputs/` | Stores processed uploaded videos |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files and folders excluded from Git |

---

## Model Files

The application expects the following model files to exist in the project root directory:

```text
best_model.keras
yolov8n.pt
yolov8n-pose.pt
```

`best_model.keras` is the trained MobileNetV2 violence classification model.

`yolov8n.pt` is used for person detection.

`yolov8n-pose.pt` is used for pose keypoint detection and hands-up warning.

If the YOLOv8 model files are not available locally, the Ultralytics package may download them automatically when the application starts, depending on the environment and internet access.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your actual GitHub username and repository name.

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

The main libraries used in this project are:

```text
streamlit
streamlit-webrtc
opencv-python
numpy
tensorflow
ultralytics
av
```

A typical `requirements.txt` file is:

```txt
streamlit
streamlit-webrtc
opencv-python
numpy
tensorflow
ultralytics
av
```

---

## Running the Application

Run the Streamlit application with:

```bash
streamlit run app.py
```

After running the command, Streamlit will open the web application in the browser.

---

## Application Modes

The application provides two modes:

```text
Camera trực tiếp
Upload video
```

### Live Camera Mode

This mode uses the webcam to perform real-time violence monitoring.

The system processes camera frames, detects people, estimates violence confidence, checks hands-up status, and displays alerts directly on the video frame.

### Upload Video Mode

This mode allows the user to upload a video file. The system processes the video frame by frame and generates an annotated output video.

Supported input formats:

```text
.mp4
.avi
.mov
.mkv
```

The processed video is saved in the `outputs/` directory.

---

## Sidebar Parameters

| Parameter | Description |
|---|---|
| Violence threshold | Minimum Fight confidence required to consider a frame suspicious or violent |
| Consecutive frames | Number of consecutive violent frames required before generating an alert |
| Smoothing window | Number of recent frames used for temporal smoothing |
| Minimum persons | Minimum number of detected people required before considering violence |
| Display width | Width of the displayed video |
| Center box | Optional visual guide box for camera positioning |
| Hands-up warning | Enables or disables the hands-up auxiliary warning |
| Hand keypoint threshold | Minimum confidence required for pose keypoints |
| Save warning image | Saves an image when a hands-up warning occurs |

---

## Alert Logic

The alert mechanism is based on three main conditions:

1. The smoothed Fight confidence must be greater than or equal to the selected violence threshold.
2. The number of detected people must be greater than or equal to the minimum-person setting.
3. The condition must remain true for a predefined number of consecutive frames.

This design helps reduce false alarms caused by isolated misclassified frames, motion blur, occlusion, or unstable predictions.

---

## Hands-Up Warning

The hands-up warning module uses YOLOv8-pose keypoints.

The system checks four keypoints:

```text
left shoulder
right shoulder
left wrist
right wrist
```

A hands-up condition is activated when both wrists are located above their corresponding shoulders with sufficient keypoint confidence.

This signal is treated only as an auxiliary warning cue. It is not considered direct evidence of violence because similar gestures may occur in normal non-violent activities.

---

## Evidence Storage

When an alert condition is satisfied, the system saves evidence images in the `captures/` directory.

The saved evidence may include:

- the full annotated frame;
- cropped person regions;
- hands-up warning frames.

For uploaded videos, the processed video is saved in:

```text
outputs/
```

These files can be reviewed or downloaded through the web interface.

---

## Experimental Results

The selected MobileNetV2 checkpoint was evaluated on 12,000 validation frames extracted from the RWF-2000 dataset.

| Metric | Value |
|---|---:|
| Accuracy | 72.06% |
| Precision (Fight) | 74.72% |
| Recall (Fight) | 66.67% |
| F1-score (Fight) | 70.47% |
| Macro F1-score | 71.98% |
| ROC-AUC | 80.92% |

### Confusion Matrix

| Actual / Predicted | Non-Fight | Fight |
|---|---:|---:|
| Non-Fight | 4647 | 1353 |
| Fight | 2000 | 4000 |

The results show that the model can support a lightweight monitoring prototype. However, the current performance is not sufficient for fully autonomous real-world deployment.

---

## Dataset

The violence classifier was trained using frames extracted from the RWF-2000 dataset.

The dataset contains two classes:

```text
Fight
Non-Fight
```

In this experiment, the extracted frame dataset contained:

| Subset | Fight | Non-Fight | Total |
|---|---:|---:|---:|
| Training | 24,000 | 24,000 | 48,000 |
| Validation | 6,000 | 6,000 | 12,000 |
| Total | 30,000 | 30,000 | 60,000 |

The dataset folder is not included in this repository because of its size.

---

## Limitations

This project has several limitations:

- The classifier works at frame level and does not explicitly learn long-term motion patterns.
- RWF-2000 is a general surveillance dataset and is not specifically collected from school environments.
- Rapid non-violent actions may sometimes be confused with violence.
- Occlusion, low light, crowded scenes, and small person regions may reduce detection reliability.
- The hands-up warning is only an auxiliary cue and should not be interpreted as independent evidence of violence.
- The system should support human decision-making rather than automatically applying disciplinary or security actions.

---

## Ethical Notice

This system is intended for academic research and prototype demonstration.

In real-world deployment, every generated alert should be reviewed by an authorized human operator. Privacy, access control, data retention, and responsible use of surveillance footage must be carefully considered.

The system should not be used as the only basis for disciplinary decisions.

---

## Future Work

Future improvements may include:

- collecting a school-specific violence dataset;
- improving data augmentation and regularization;
- incorporating temporal video models such as LSTM, 3D CNN, or transformer-based models;
- adding multi-person tracking;
- improving threshold calibration;
- evaluating the system in real school-like environments;
- optimizing inference speed for low-resource devices.

---

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- YOLOv8
- YOLOv8-pose
- OpenCV
- Streamlit
- Streamlit WebRTC
- NumPy

---

## Disclaimer

This project is a research prototype. It is not a certified safety system and should not be used for fully autonomous violence detection in real environments without further validation, privacy review, and human supervision.
