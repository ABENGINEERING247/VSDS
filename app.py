
import os
import time
import threading
from collections import deque

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

# Shared results updated by the webcam processing thread.
results_lock = threading.Lock()
results = {
    "vehicles": {},
    "events": deque(maxlen=100),
    "total": 0,
    "fps": 0.0,
}


# =========================================================
# VEHICLE DETECTION AND SPEED ESTIMATION
# =========================================================
class VehicleSpeedProcessor(VideoProcessorBase):

    def __init__(
        self,
        model_name,
        confidence,
        line1_ratio,
        line2_ratio,
        distance_meters,
    ):
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
        self.lock = threading.Lock()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        height, width = image.shape[:2]

        line1_y = int(height * self.line1_ratio)
        line2_y = int(height * self.line2_ratio)

        # Use a YOLO detection/tracking model.
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
                f"Detection error: {str(exc)[:65]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            return av.VideoFrame.from_ndarray(
                image, format="bgr24"
            )

        # Draw calibration lines.
        cv2.line(
            image, (0, line1_y), (width, line1_y),
            (0, 255, 255), 2
        )
        cv2.line(
            image, (0, line2_y), (width, line2_y),
            (255, 180, 0), 2
        )
        cv2.putText(
            image, "LINE A", (10, max(20, line1_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 255, 255), 2
        )
        cv2.putText(
            image, "LINE B", (10, max(20, line2_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 180, 0), 2
        )

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

            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [-1] * len(xyxy)

            for box, class_id, score, track_id in zip(
                xyxy, cls_ids, confs, track_ids
            ):
                if class_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                label = VEHICLE_CLASSES[class_id]
                speed = self.speed_cache.get(track_id)

                # Calculate speed from the time between line crossings.
                # This assumes the vehicle travels along the calibrated
                # road segment and crosses both lines.
                if track_id >= 0:
                    previous_y = self.previous_positions.get(track_id)

                    if previous_y is not None:
                        previous_side = (
                            1 if previous_y < line1_y else
                            2 if previous_y < line2_y else 3
                        )
                        current_side = (
                            1 if cy < line1_y else
                            2 if cy < line2_y else 3
                        )

                        if previous_side != current_side:
                            # Record first crossing of either line.
                            crossed = []
                            if (
                                previous_y < line1_y <= cy
                                or previous_y >= line1_y > cy
                            ):
                                crossed.append(1)
                            if (
                                previous_y < line2_y <= cy
                                or previous_y >= line2_y > cy
                            ):
                                crossed.append(2)

                            for line_number in crossed:
                                key = (track_id, line_number)

                                if key not in self.crossing_times:
                                    self.crossing_times[key] = now

                                other_line = 2 if line_number == 1 else 1
                                other_key = (track_id, other_line)

                                if other_key in self.crossing_times:
                                    elapsed = abs(
                                        self.crossing_times[key]
                                        - self.crossing_times[other_key]
                                    )

                                    if elapsed > 0.05:
                                        speed = (
                                            self.distance_meters
                                            / elapsed
                                        ) * 3.6

                                        self.speed_cache[track_id] = speed

                                        with results_lock:
                                            results["events"].append({
                                                "ID": track_id,
                                                "Vehicle": label,
                                                "Speed (km/h)": round(
                                                    speed, 1
                                                ),
                                                "Time": time.strftime(
                                                    "%H:%M:%S"
                                                ),
                                            })

                    self.previous_positions[track_id] = cy

                # Draw vehicle box and speed.
                color = (0, 220, 0)
                cv2.rectangle(
                    image, (x1, y1), (x2, y2), color, 2
                )

                if speed is not None:
                    text_label = (
                        f"{label} ID:{track_id} "
                        f"{speed:.1f} km/h"
                    )
                else:
                    text_label = (
                        f"{label} ID:{track_id} "
                        f"{score:.2f} | Measuring..."
                    )

                cv2.putText(
                    image,
                    text_label,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

                current_vehicles[str(track_id)] = {
                    "ID": track_id,
                    "Vehicle": label,
                    "Speed (km/h)": (
                        round(speed, 1)
                        if speed is not None else None
                    ),
                    "Confidence": round(float(score), 2),
                }

        with results_lock:
            results["vehicles"] = current_vehicles
            results["total"] = max(
                results["total"], len(self.counted_ids)
            )
            results["fps"] = round(fps, 1)

        cv2.putText(
            image,
            "VEHICLE SPEED MONITOR",
            (10, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return av.VideoFrame.from_ndarray(
            image, format="bgr24"
        )


# =========================================================
# USER INTERFACE
# =========================================================
st.title("🚗 Vehicle Speed Detection System")
st.caption(
    "Designed by Engr. Bilal Mehmood | "
    "Python • Streamlit • Ultralytics YOLO"
)

st.info(
    "Speed is estimated using two calibrated road lines. "
    "For accurate results, mount the camera in a fixed position, "
    "measure the road distance between the lines, and keep the "
    "lines aligned with the direction of vehicle travel."
)

with st.sidebar:
    st.header("⚙️ Detection Settings")

    model_name = st.selectbox(
        "YOLO model",
        ["yolo11n.pt", "yolov8n.pt", "yolo11s.pt"],
        index=0,
        help="Nano models are faster on most CPUs."
    )

    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.35,
        step=0.05,
    )

    distance_meters = st.number_input(
        "Distance between lines (meters)",
        min_value=1.0,
        max_value=100.0,
        value=5.0,
        step=0.5,
    )

    st.subheader("Line placement")
    line1_ratio = st.slider(
        "Line A position (frame height)",
        0.10, 0.80, 0.35, 0.05
    )
    line2_ratio = st.slider(
        "Line B position (frame height)",
        0.20, 0.95, 0.65, 0.05
    )

    if line2_ratio <= line1_ratio:
        st.error("Line B must be below Line A.")
        st.stop()

    st.warning(
        "Speed values are estimates. Perspective, camera angle, "
        "frame drops, and tracking errors can affect accuracy."
    )

    st.markdown("### Vehicle classes")
    st.write("🚗 Car")
    st.write("🏍️ Motorcycle")
    st.write("🚌 Bus")
    st.write("🚚 Truck")

# =========================================================
# WEBCAM
# =========================================================
st.subheader("📹 Live Webcam Feed")

ctx = webrtc_streamer(
    key="vehicle-speed-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: VehicleSpeedProcessor(
        model_name=model_name,
        confidence=confidence,
        line1_ratio=line1_ratio,
        line2_ratio=line2_ratio,
        distance_meters=distance_meters,
    ),
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)

if ctx.state.playing:
    st.success("Webcam is running. Move vehicles through both lines.")

# =========================================================
# LIVE METRICS
# =========================================================
@st.fragment(run_every=1)
def show_live_metrics():
    with results_lock:
        vehicles = list(results["vehicles"].values())
        events = list(results["events"])
        fps = results["fps"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Vehicles currently tracked", len(vehicles))
    col2.metric("Processing FPS", fps)
    col3.metric(
        "Vehicles with speed estimates",
        sum(v["Speed (km/h)"] is not None for v in vehicles),
    )

    st.subheader("📊 Live Vehicle Details")

    if vehicles:
        st.dataframe(
            vehicles,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No vehicles detected yet.")

    st.subheader("🧾 Speed Measurement History")

    if events:
        import pandas as pd

        history = pd.DataFrame(events)
        st.dataframe(
            history.iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Speed History CSV",
            data=csv_data,
            file_name="vehicle_speed_history.csv",
            mime="text/csv",
        )
    else:
        st.write(
            "No vehicle has crossed both measurement lines yet."
        )


show_live_metrics()

st.divider()
st.caption(
    "For research and educational use. "
    "Validate measurements against a calibrated speed-measuring "
    "device before using them for traffic enforcement."
)
