import base64
import os
import tempfile
import time
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Violence Detection V3", layout="wide")

MODEL_PATH = "models/best_model.keras"
PERSON_MODEL_PATH = "yolov8n.pt"
POSE_MODEL_PATH = "yolov8n-pose.pt"
IMG_SIZE = (224, 224)
CAPTURE_DIR = "captures"
OUTPUT_DIR = "outputs"
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Cài đặt")
app_mode = st.sidebar.radio("Chế độ chạy", ["Camera trực tiếp", "Upload video"], index=0)
violence_threshold = st.sidebar.slider("Ngưỡng bạo lực", 0.50, 0.99, 0.85, 0.01)
alert_streak = st.sidebar.slider("Số frame liên tiếp để báo đỏ", 1, 30, 15, 1)
smooth_window = st.sidebar.slider("Cửa sổ làm mượt", 1, 20, 5, 1)
min_persons = st.sidebar.slider("Số người tối thiểu để xét bạo lực", 1, 4, 2, 1)
preview_width = st.sidebar.slider("Độ rộng hiển thị video", 320, 1100, 680, 20)
show_center_box = st.sidebar.checkbox("Hiện khung trung tâm", value=False)
enable_hands_warning = st.sidebar.checkbox("Bật cảnh báo giơ 2 tay", value=True)
hand_conf_threshold = st.sidebar.slider("Ngưỡng keypoint tay", 0.10, 0.90, 0.25, 0.05)
warning_save_enabled = st.sidebar.checkbox("Lưu ảnh khi warning", value=True)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    violence_model = load_model(MODEL_PATH)
    person_yolo = YOLO(PERSON_MODEL_PATH)
    pose_yolo = YOLO(POSE_MODEL_PATH)
    return violence_model, person_yolo, pose_yolo

violence_model, person_yolo, pose_yolo = load_models()

# ============================================================
# HELPERS
# ============================================================
def preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    crop = cv2.resize(crop_bgr, IMG_SIZE)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = crop.astype(np.float32)
    crop = tf.keras.applications.mobilenet_v2.preprocess_input(crop)
    crop = np.expand_dims(crop, axis=0)
    return crop


def predict_violence(crop_bgr: np.ndarray) -> float:
    x = preprocess_crop(crop_bgr)
    pred = violence_model.predict(x, verbose=0)
    return float(pred[0][0])


def draw_label_box(frame, text, x, y, color, scale=0.40, thickness=1):
    (tw, th), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness
    )

    cv2.rectangle(
        frame,
        (x, y - th - 6),
        (x + tw + 6, y + 3),
        color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 3, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def hands_up_from_pose_result(pose_result, conf_thr: float = 0.25) -> bool:
    """Return True if any detected person has both wrists above shoulders."""
    if pose_result is None or pose_result.keypoints is None:
        return False

    try:
        kpts = pose_result.keypoints.data.cpu().numpy()  # (N, 17, 3)
    except Exception:
        return False

    # COCO keypoints: 5 left shoulder, 6 right shoulder, 9 left wrist, 10 right wrist
    for person in kpts:
        ls = person[5]
        rs = person[6]
        lw = person[9]
        rw = person[10]

        if min(ls[2], rs[2], lw[2], rw[2]) < conf_thr:
            continue

        if lw[1] < ls[1] and rw[1] < rs[1]:
            return True

    return False


def video_html(video_path: str, width_px: int) -> str:
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    b64 = base64.b64encode(video_bytes).decode()
    return f"""
    <video width="{width_px}" controls>
        <source src="data:video/mp4;base64,{b64}" type="video/mp4">
        Trình duyệt của bạn không hỗ trợ video HTML5.
    </video>
    """


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class ViolenceScanner:
    def __init__(self):
        self.prob_history = deque(maxlen=smooth_window)
        self.violent_streak = 0
        self.alert_saved = False
        self.warning_saved = False
        self.frame_idx = 0
        self.person_count = 0
        self.hands_up = False
        self.smooth_prob = 0.0
        self.adjusted_prob = 0.0
        self.status_text = "Status: Safe"
        self.status_color = (0, 255, 0)
        self.evidence_paths = []
        self.best_crop = None
        self.total_alerts = 0
        self.total_warnings = 0
        self.pose_counter = 0

    def process(self, frame: np.ndarray, save_prefix: str = "scan") -> np.ndarray:
        self.frame_idx += 1
        h, w = frame.shape[:2]

        # -------- Person detection --------
        person_results = person_yolo.predict(frame, conf=0.35, classes=[0], verbose=False)
        person_boxes = person_results[0].boxes

        self.person_count = 0
        best_prob = 0.0
        self.best_crop = None

        if person_boxes is not None and len(person_boxes) > 0:
            xyxy = person_boxes.xyxy.cpu().numpy().astype(int)
            for (x1, y1, x2, y2) in xyxy:
                self.person_count += 1

                pad_x = int((x2 - x1) * 0.10)
                pad_y = int((y2 - y1) * 0.10)

                x1p = max(0, x1 - pad_x)
                y1p = max(0, y1 - pad_y)
                x2p = min(w, x2 + pad_x)
                y2p = min(h, y2 + pad_y)

                crop = frame[y1p:y2p, x1p:x2p]
                if crop.size == 0:
                    continue

                prob = predict_violence(crop)
                if prob > best_prob:
                    best_prob = prob
                    self.best_crop = crop.copy()

                color = (0, 0, 255) if prob >= violence_threshold else (0, 255, 0)
                cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), color, 2)
                draw_label_box(
                    frame,
                    f"{'Fight' if prob >= violence_threshold else 'Safe'} {prob:.2f}",
                    x1p,
                    max(25, y1p),
                    color,
                    scale=0.50,
                    thickness=2,
                )

        # -------- Hands-up warning (pose) --------
        self.hands_up = False
        if enable_hands_warning and (self.frame_idx % 3 == 0):
            pose_results = pose_yolo.predict(frame, conf=0.25, verbose=False)
            if len(pose_results) > 0:
                self.hands_up = hands_up_from_pose_result(pose_results[0], conf_thr=hand_conf_threshold)

        # -------- Smoothing + rules --------
        self.prob_history.append(best_prob)
        self.smooth_prob = float(np.mean(self.prob_history)) if len(self.prob_history) > 0 else 0.0

        self.adjusted_prob = self.smooth_prob
        if self.person_count < min_persons:
            self.adjusted_prob *= 0.55

        violent_frame = self.adjusted_prob >= violence_threshold and self.person_count >= min_persons

        if violent_frame:
            self.violent_streak += 1
        else:
            self.violent_streak = 0

        alert = self.violent_streak >= alert_streak
        warning = self.hands_up and not alert

        # -------- UI overlay: nhỏ gọn, không che video --------
        status_color = (0, 255, 0)
        status_text = "Status: Safe"

        if alert:
            status_color = (0, 0, 255)
            status_text = f"ALERT! Violence for {self.violent_streak} frames"
        elif warning:
            status_color = (0, 255, 255)
            status_text = "WARNING! Hands up detected"
        elif self.adjusted_prob >= violence_threshold:
            status_color = (0, 165, 255)
            status_text = f"Suspicious ({self.violent_streak}/{alert_streak})"

        self.status_color = status_color
        self.status_text = status_text

        # small top labels
        draw_label_box(frame, f"Persons: {self.person_count}", 16, 32, (50, 50, 50), scale=0.55)
        draw_label_box(frame, f"Score: {self.adjusted_prob:.2f}", 16, 62, status_color, scale=0.55)
        draw_label_box(
            frame,
            f"HandsUp: {'Yes' if self.hands_up else 'No'}",
            16,
            92,
            (0, 255, 255) if self.hands_up else (50, 180, 50),
            scale=0.50,
        )

        # bottom label
        draw_label_box(frame, status_text, 16, h - 18, status_color, scale=0.60)

        # optional center guidance box (thin, small)
        if show_center_box:
            box_w, box_h = int(w * 0.42), int(h * 0.48)
            cx1 = (w - box_w) // 2
            cy1 = (h - box_h) // 2
            cx2 = cx1 + box_w
            cy2 = cy1 + box_h
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)

        # -------- Evidence save --------
        if alert and not self.alert_saved:
            ts = int(time.time())
            frame_file = os.path.join(CAPTURE_DIR, f"{save_prefix}_alert_frame_{ts}_{self.frame_idx}.jpg")
            cv2.imwrite(frame_file, frame)
            self.evidence_paths.append(frame_file)
            if self.best_crop is not None:
                crop_file = os.path.join(CAPTURE_DIR, f"{save_prefix}_alert_person_{ts}_{self.frame_idx}.jpg")
                cv2.imwrite(crop_file, self.best_crop)
                self.evidence_paths.append(crop_file)
            self.alert_saved = True
            self.total_alerts += 1

        if warning and warning_save_enabled and not self.warning_saved:
            ts = int(time.time())
            warn_file = os.path.join(CAPTURE_DIR, f"{save_prefix}_warning_{ts}_{self.frame_idx}.jpg")
            cv2.imwrite(warn_file, frame)
            self.evidence_paths.append(warn_file)
            self.warning_saved = True
            self.total_warnings += 1

        if not alert:
            self.alert_saved = False
        if not warning:
            self.warning_saved = False

        return frame


class LiveVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.scanner = ViolenceScanner()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        annotated = self.scanner.process(img, save_prefix="live")
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ============================================================
# OFFLINE VIDEO PROCESSING
# ============================================================
def process_video(uploaded_bytes: bytes, original_name: str):
    suffix = os.path.splitext(original_name)[1].lower()
    if suffix not in [".mp4", ".avi", ".mov", ".mkv"]:
        suffix = ".mp4"

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_input.write(uploaded_bytes)
    temp_input.close()

    cap = cv2.VideoCapture(temp_input.name)
    if not cap.isOpened():
        raise RuntimeError("Không mở được video đầu vào.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1

    ts = int(time.time())
    processed_path = os.path.join(OUTPUT_DIR, f"processed_{ts}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))

    scanner = ViolenceScanner()

    progress = st.progress(0)
    status = st.empty()
    preview = st.empty()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated = scanner.process(frame, save_prefix="upload")
        writer.write(annotated)

        preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_idx}", use_container_width=True)
        progress.progress(min(frame_idx / total_frames, 1.0))
        status.info(
            f"Frame {frame_idx}/{total_frames} | persons={scanner.person_count} | score={scanner.adjusted_prob:.2f} | streak={scanner.violent_streak} | hands_up={scanner.hands_up}"
        )

    cap.release()
    writer.release()

    return temp_input.name, processed_path, scanner.evidence_paths, scanner


# ============================================================
# UI
# ============================================================
st.title("Violence Detection Web App V3")
st.caption("Camera trực tiếp hoặc upload video, giao diện gọn hơn, và cảnh báo giơ 2 tay / bạo lực.")

if app_mode == "Camera trực tiếp":
    st.subheader("Camera trực tiếp")
    st.write("Bật camera để hệ thống nhận diện theo thời gian thực.")

    webrtc_streamer(
        key="live-violence",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )

else:
    uploaded = st.file_uploader("Chọn video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is not None:
        run_btn = st.button("Bắt đầu quét", type="primary")
        if run_btn:
            with st.spinner("Đang xử lý video..."):
                input_path, processed_path, evidence_paths, scanner = process_video(
                    uploaded.getvalue(),
                    uploaded.name,
                )

            st.success("Đã xử lý xong.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Model", "MobileNetV2")
            m2.metric("Detector", "YOLOv8")
            m3.metric("Pose warning", "ON" if enable_hands_warning else "OFF")

            st.write("## Video")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Video gốc")
                st.components.v1.html(
                    video_html(input_path, preview_width),
                    height=int(preview_width * 0.62) + 80,
                    scrolling=False,
                )
                st.download_button(
                    "Tải video gốc",
                    data=read_bytes(input_path),
                    file_name=os.path.basename(input_path),
                    mime="video/mp4",
                )
            with c2:
                st.subheader("Video đã quét")
                st.components.v1.html(
                    video_html(processed_path, preview_width),
                    height=int(preview_width * 0.62) + 80,
                    scrolling=False,
                )
                st.download_button(
                    "Tải video đã quét",
                    data=read_bytes(processed_path),
                    file_name=os.path.basename(processed_path),
                    mime="video/mp4",
                )

            st.write("## Ảnh bằng chứng")
            if evidence_paths:
                cols = st.columns(3)
                for i, p in enumerate(evidence_paths):
                    with cols[i % 3]:
                        st.image(p, caption=os.path.basename(p), use_container_width=True)
                        st.download_button(
                            f"Tải {os.path.basename(p)}",
                            data=read_bytes(p),
                            file_name=os.path.basename(p),
                            mime="image/jpeg",
                            key=f"upload_dl_{i}_{p}",
                        )
            else:
                st.info("Không có ảnh bằng chứng được lưu.")

            st.write("## Tệp đã tạo")
            info_text = f"""Input temp: {input_path}
Processed: {processed_path}
Capture dir: {CAPTURE_DIR}/
Output dir: {OUTPUT_DIR}/
Alerts: {scanner.total_alerts}
Warnings: {scanner.total_warnings}"""
            st.code(info_text)

            st.write("## Gợi ý đọc kết quả")
            st.markdown(
                """
- 🟢 **Xanh**: an toàn
- 🟡 **Vàng**: cảnh báo giơ 2 tay
- 🟠 **Cam**: nghi ngờ
- 🔴 **Đỏ**: phát hiện bạo lực liên tiếp
"""
            )
    else:
        st.info("Hãy upload video để bắt đầu.")
        st.write("### Chế độ upload")
        st.markdown(
            """
- Video gốc và video đã quét sẽ hiện ở 2 cột nhỏ gọn hơn.
- Bạn có thể chỉnh độ rộng trong sidebar.
- Ảnh bằng chứng sẽ hiện bên dưới sau khi quét xong.
"""
        )
