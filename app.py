import os
import time
import threading
from collections import deque
import queue

import av
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Vehicle Speed Detection System",
    page_icon="🚗",
    layout="wide",
)

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# Shared state protected by lock
state_lock = threading.Lock()
shared_state = {
    "vehicles": {},
    "events": deque(maxlen=100),
    "total": 0,
    "fps": 0.0,
    "annotated_frame": None,
}


# =========================================================
# BACKGROUND YOLO WORKER TO PREVENT WEBCAM HANGING
# =========================================================
class YOLOInferenceWorker:
    def __init__(self, model_name, confidence, line1_ratio, line2_ratio, distance_meters):
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.line1_ratio = line1_ratio
        self.line2_ratio = line2_ratio
        self.distance_meters = distance_meters

        self.previous_positions = {}
        self.crossing_times = {}
        self.speed_cache = {}
        self.counted_ids = set()
        self.frame_times = deque(maxlen=20)

        self.input_queue = queue.Queue(maxsize=2)
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def _process_loop(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            image = frame.copy()
            height, width = image.shape[:2]

            line1_y = int(height * self.line1_ratio)
            line2_y = int(height * self.line2_ratio)

            try:
                prediction = self.model.track(
                    image,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=list(VEHICLE_CLASSES.keys()),
                    conf=self.confidence,
                    verbose=False,
                )[0]
            except Exception as exc:
                cv2.putText(
                    image,
                    f"Detection error: {str(exc)[:50]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                with state_lock:
                    shared_state["annotated_frame"] = image
                continue

            # Draw calibration lines
            cv2.line(image, (0, line1_y), (width, line1_y), (0, 255, 255), 2)
            cv2.line(image, (0, line2_y), (width, line2_y), (255, 180, 0), 2)
            cv2.putText(image, "LINE A", (10, max(20, line1_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(image, "LINE B", (10, max(20, line2_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2)

            now = time.monotonic()
            self.frame_times.append(now)
            fps = 0.0
            if len(self.frame_times) > 1:
                elapsed = self.frame_times[-1] - self.frame_times[0]
                if elapsed > 0:
                    fps = (len(self.frame_times) - 1) / elapsed

            current_vehicles = {}
            boxes = prediction.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)

                for box, class_id, score, track_id in zip(xyxy, cls_ids, confs, track_ids):
                    if class_id not in VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    label = VEHICLE_CLASSES[class_id]
                    speed = self.speed_cache.get(track_id)

                    if track_id >= 0:
                        previous_y = self.previous_positions.get(track_id)
                        if previous_y is not None:
                            previous_side = 1 if previous_y < line1_y else (2 if previous_y < line2_y else 3)
                            current_side = 1 if cy < line1_y else (2 if cy < line2_y else 3)

                            if previous_side != current_side:
                                crossed = []
                                if previous_y < line1_y <= cy or previous_y >= line1_y > cy:
                                    crossed.append(1)
                                if previous_y < line2_y <= cy or previous_y >= line2_y > cy:
                                    crossed.append(2)

                                for line_number in crossed:
                                    key = (track_id, line_number)
                                    if key not in self.crossing_times:
                                        self.crossing_times[key] = now

                                    other_line = 2 if line_number == 1 else 1
                                    other_key = (track_id, other_line)

                                    if other_key in self.crossing_times:
                                        elapsed = abs(self.crossing_times[key] - self.crossing_times[other_key])
                                        if elapsed > 0.05:
                                            speed = (self.distance_meters / elapsed) * 3.6
                                            self.speed_cache[track_id] = speed

                                            with state_lock:
                                                shared_state["events"].append({
                                                    "ID": track_id,
                                                    "Vehicle": label,
                                                    "Speed (km/h)": round(speed, 1),
                                                    "Time": time.strftime("%H:%M:%S"),
                                                })

                        self.previous_positions[track_id] = cy

                    color = (0, 220, 0)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                    if speed is not None:
                        text_label = f"{label} ID:{track_id} {speed:.1f} km/h"
                    else:
                        text_label = f"{label} ID:{track_id} {score:.2f} | Measuring..."

                    cv2.putText(image, text_label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    current_vehicles[str(track_id)] = {
                        "ID": track_id,
                        "Vehicle": label,
                        "Speed (km/h)": round(speed, 1) if speed is not None else None,
                        "Confidence": round(float(score), 2),
                    }

            cv2.putText(image, "VEHICLE SPEED MONITOR", (10, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            with state_lock:
                shared_state["vehicles"] = current_vehicles
                shared_state["fps"] = round(fps, 1)
                shared_state["annotated_frame"] = image

    def push(self, frame):
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.input_queue.put_nowait(frame)
        except queue.Full:
            pass

    def stop(self):
        self.running = False


# =========================================================
# WEBCAM PROCESSOR (NON-BLOCKING)
# =========================================================
class VehicleSpeedProcessor(VideoProcessorBase):
    def __init__(self):
        self.worker = st.session_state.get("yolo_worker")

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        
        if self.worker:
            self.worker.push(image)

        with state_lock:
            annotated = shared_state["annotated_frame"]

        if annotated is not None:
            display_image = annotated
        else:
            display_image = image

        return av.VideoFrame.from_ndarray(display_image, format="bgr24")


# =========================================================
# USER INTERFACE
# =========================================================
st.title("🚗 Vehicle Speed Detection System")
st.caption("Designed by Engr. Bilal Mehmood | Python • Streamlit • Ultralytics YOLO")

st.info(
    "Speed is estimated using two calibrated road lines. "
    "The background processing worker prevents stream freezing and lag."
)

with st.sidebar:
    st.header("⚙️ Detection Settings")

    model_name = st.selectbox(
        "YOLO model",
        ["yolo11n.pt", "yolov8n.pt", "yolo11s.pt"],
        index=0,
    )

    confidence = st.slider("Detection confidence", 0.10, 0.95, 0.35, 0.05)
    distance_meters = st.number_input("Distance between lines (meters)", 1.0, 100.0, 5.0, 0.5)

    st.subheader("Line placement")
    line1_ratio = st.slider("Line A position (frame height)", 0.10, 0.80, 0.35, 0.05)
    line2_ratio = st.slider("Line B position (frame height)", 0.20, 0.95, 0.65, 0.05)

    if line2_ratio <= line1_ratio:
        st.error("Line B must be below Line A.")
        st.stop()

# Initialize or update worker in session state safely
if "yolo_worker" not in st.session_state or st.session_state.get("model_name_cached") != model_name:
    if "yolo_worker" in st.session_state:
        st.session_state["yolo_worker"].stop()
    st.session_state["yolo_worker"] = YOLOInferenceWorker(
        model_name, confidence, line1_ratio, line2_ratio, distance_meters
    )
    st.session_state["model_name_cached"] = model_name

# Update worker config live if sliders change
worker = st.session_state["yolo_worker"]
worker.confidence = confidence
worker.distance_meters = distance_meters
worker.line1_ratio = line1_ratio
worker.line2_ratio = line2_ratio

# =========================================================
# WEBCAM STREAMER
# =========================================================
st.subheader("📹 Live Webcam Feed")

webrtc_streamer(
    key="vehicle-speed-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VehicleSpeedProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# =========================================================
# LIVE METRICS & DASHBOARD
# =========================================================
@st.fragment(run_every=1)
def show_live_metrics():
    with state_lock:
        vehicles = list(shared_state["vehicles"].values())
        events = list(shared_state["events"])
        fps = shared_state["fps"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Vehicles currently tracked", len(vehicles))
    col2.metric("Processing FPS", fps)
    col3.metric("Vehicles with speed estimates", sum(v["Speed (km/h)"] is not None for v in vehicles))

    st.subheader("📊 Live Vehicle Details")
    if vehicles:
        st.dataframe(vehicles, use_container_width=True, hide_index=True)
    else:
        st.write("No vehicles detected yet.")

    st.subheader("🧾 Speed Measurement History")
    if events:
        import pandas as pd
        history = pd.DataFrame(events)
        st.dataframe(history.iloc[::-1], use_container_width=True, hide_index=True)
        
        csv_data = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Speed History CSV",
            data=csv_data,
            file_name="vehicle_speed_history.csv",
            mime="text/csv",
        )
    else:
        st.write("No vehicle has crossed both measurement lines yet.")

show_live_metrics()

st.divider()
st.caption("For research and educational use.")
