import os
import gdown
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.title("PPE Compliance Detection System")

# ── Load model once and cache it ──────────────────────────────────────────────
# Paste your Google Drive File ID here
GDRIVE_FILE_ID = "1Ysg4LquEcUb_T86oHkGNOm7B5j8R2RpW"
MODEL_PATH = r"C:\Users\Acer\Documents\Grant_Thornton\4_ConstructionSiteSafety\Full_Version\YOLO_PPE_Full.pt"

@st.cache_resource
def load_model():
    # If the file is not found locally, auto-download from Google Drive
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights from Google Drive... Please wait."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = load_model()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Stream Detection", "About the Model"])

# ── Violation Classes ────────────────────────────────────────────────────────
VIOLATION_CLASSES = {"NO-Gloves", "NO-Goggles", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Fall-Detected"}

# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    # ── Controls above Uploader ───────────────────────────────────────────────
    st.subheader("Model Parameters")
    c1, c2 = st.columns(2)
    with c1:
        conf_threshold = st.slider("Confidence Threshold", min_value=0.05, max_value=1.0, value=0.15, step=0.05)
    with c2:
        img_size = st.select_slider(
            "Inference Resolution",
            options=[640, 800, 960, 1280],
            value=1280,
            help="Higher resolution preserves small objects like hands and goggles in wide-angle scenes."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if file is not None:
        image = Image.open(file).convert("RGB")

        # Run inference
        results = model.predict(image, conf=conf_threshold, iou=0.45, imgsz=img_size, verbose=False)
        annotated_bgr = results[0].plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]

        detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
        violations = [c for c in detected_classes if c in VIOLATION_CLASSES]
        compliant  = [c for c in detected_classes if c not in VIOLATION_CLASSES]

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Two-Column Layout ─────────────────────────────────────────────────
        left_col, right_col = st.columns([1, 1], gap="medium")

        with left_col:
            st.subheader("Original Image")
            st.image(image, width=480)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Detection Result")
            st.image(annotated_rgb, width=480)

        with right_col:
            st.subheader("Compliance & Infraction Audit")

            if len(detected_classes) == 0:
                st.info("No target PPE items or personnel identified in this frame.")
            else:
                if len(violations) > 0:
                    st.error(f"NON-COMPLIANCE DETECTED: {len(violations)} infraction instance(s) flagged.")
                elif len(compliant) > 0:
                    st.success(f"VERIFIED: {len(compliant)} compliant PPE item(s) positively identified.")

                col_comp, col_viol = st.columns(2)

                with col_comp:
                    st.markdown("**Compliant Equipment**")
                    if compliant:
                        for item in sorted(set(compliant)):
                            count = compliant.count(item)
                            st.success(f"{item}: {count}")
                    else:
                        st.info("None detected.")

                with col_viol:
                    st.markdown("**Violations Flagged**")
                    if violations:
                        for item in sorted(set(violations)):
                            count = violations.count(item)
                            st.error(f"{item}: {count}")
                    else:
                        st.info("None flagged.")

            st.markdown("---")
            st.subheader("Detailed Detections")

            if len(results[0].boxes) > 0:
                boxes_data = []
                for box in results[0].boxes:
                    class_name = model.names[int(box.cls)]
                    confidence = float(box.conf)
                    boxes_data.append({
                        "Class": class_name,
                        "Type": "Violation" if class_name in VIOLATION_CLASSES else "Compliant",
                        "Confidence": f"{confidence:.2%}"
                    })
                st.dataframe(boxes_data, use_container_width=True, hide_index=True)
            else:
                st.info("No detection records.")

    else:
        st.info("Upload an image above to run analysis.")

# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Video Stream & Surveillance Analytics")

    # Video parameters
    v_c1, v_c2 = st.columns(2)
    with v_c1:
        v_conf = st.slider("Video Confidence Threshold", min_value=0.05, max_value=1.0, value=0.20, step=0.05, key="v_conf")
    with v_c2:
        frame_skip = st.slider("Frame Skip (Speedup)", min_value=1, max_value=5, value=2, help="Process every Nth frame for higher playback FPS.", key="v_skip")

    st.markdown("<br>", unsafe_allow_html=True)
    video_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")

    if video_file is not None:
        import tempfile
        import cv2
        import time

        # Save uploaded video to a temp file for OpenCV VideoCapture
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        st.markdown(f"**Video Loaded:** {total_frames} frames | Native FPS: {orig_fps:.1f}")

        start_btn = st.button("Start Video Analysis")
        stop_btn = st.button("Stop Analysis")

        if start_btn:
            v_col1, v_col2 = st.columns([3, 2], gap="medium")

            with v_col1:
                frame_placeholder = st.empty()

            with v_col2:
                fps_metric = st.empty()
                status_metric = st.empty()
                breakdown_placeholder = st.empty()

            frame_count = 0
            prev_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                # Run inference on frame
                results = model.predict(frame, conf=v_conf, iou=0.45, imgsz=640, verbose=False)
                annotated_bgr = results[0].plot()
                annotated_rgb = annotated_bgr[:, :, ::-1]

                # Calculate real-time FPS
                current_time = time.time()
                fps = 1.0 / (current_time - prev_time + 1e-6)
                prev_time = current_time

                # Parse detections
                d_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
                v_violations = [c for c in d_classes if c in VIOLATION_CLASSES]
                v_compliant  = [c for c in d_classes if c not in VIOLATION_CLASSES]

                # Resize frame for smooth cloud WebSocket streaming and display
                display_frame = cv2.resize(annotated_rgb, (640, 360))
                frame_placeholder.image(display_frame, width=640)
                time.sleep(0.02)  # Yield to WebSocket to ensure fluid frame transmission

                # Update live stats
                fps_metric.metric("Inference Throughput", f"{fps:.1f} FPS")

                if len(v_violations) > 0:
                    status_metric.error(f"LIVE INFRACTIONS: {len(v_violations)} active violation(s)")
                else:
                    status_metric.success("STATUS: Normal / Compliant")

                # Show breakdown
                with breakdown_placeholder.container():
                    st.markdown("**Current Frame Detections:**")
                    if v_violations:
                        for item in sorted(set(v_violations)):
                            st.write(f"{item}: {v_violations.count(item)}")
                    if v_compliant:
                        for item in sorted(set(v_compliant)):
                            st.write(f"{item}: {v_compliant.count(item)}")
                    if not v_violations and not v_compliant:
                        st.write("No active targets in frame.")

            cap.release()
            st.success("Video processing completed.")
    else:
        st.info("Upload an MP4 or AVI video to run real-time surveillance detection.")

# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Model Specifications & Architecture")
    st.markdown("""
    - **Architecture:** YOLO11n (Anchor-Free Convolutional Neural Network)
    - **Training Dataset:** Roboflow Combined PPE Benchmark
    - **Training Volume:** 44,000 Annotations (22,077 Test Instances Evaluated)
    - **Optimization Epochs:** 25 Epochs with Decoupled Detection Head
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Empirical Performance Benchmark")
    st.table({
        "Metric": ["Precision (B)", "Recall (B)", "mAP@0.5", "mAP@0.5:0.95", "Inference Speed"],
        "Value":  ["69.5%",         "81.5%",       "76.8%",   "49.1%",         "3.4 ms / frame"]
    })

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Class Taxonomy (14 Monitored Targets)")
    st.markdown("""
    | Compliance Classes | Violation Classes | Auxiliary Classes |
    |---|---|---|
    | Gloves | NO-Gloves | Person |
    | Goggles | NO-Goggles | Safety Cone |
    | Hardhat | NO-Hardhat | Ladder |
    | Mask | NO-Mask | Fall-Detected |
    | Safety Vest | NO-Safety Vest | — |
    """)
