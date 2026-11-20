import sys
import os
import streamlit as st

# Allow running directly with 'python dashboard.py' or the IDE Play button
if not st.runtime.exists():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(stcli.main())

import cv2
import numpy as np

# Ensure working directory is the script directory for relative model paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Page configuration
st.set_page_config(
    page_title="Gender & Age Detection Dashboard",
    page_icon="🧑‍🤝‍🧑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e9ecef;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-gender {
        background-color: #e0f2fe;
        color: #0369a1;
    }
    .badge-age {
        background-color: #fef3c7;
        color: #b45309;
    }
</style>
""", unsafe_allow_html=True)

# Model definitions & paths
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

FACE_PROTO = os.path.join(SCRIPT_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join(SCRIPT_DIR, "opencv_face_detector_uint8.pb")
AGE_PROTO = os.path.join(SCRIPT_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(SCRIPT_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(SCRIPT_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(SCRIPT_DIR, "gender_net.caffemodel")

@st.cache_resource(show_spinner="Loading deep learning models into memory...")
def load_models():
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    return face_net, age_net, gender_net

def detect_faces(net, frame, conf_threshold=0.7):
    frame_h, frame_w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    confidences = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > conf_threshold:
            x1 = max(0, int(detections[0, 0, i, 3] * frame_w))
            y1 = max(0, int(detections[0, 0, i, 4] * frame_h))
            x2 = min(frame_w - 1, int(detections[0, 0, i, 5] * frame_w))
            y2 = min(frame_h - 1, int(detections[0, 0, i, 6] * frame_h))
            face_boxes.append([x1, y1, x2, y2])
            confidences.append(confidence)
            
    return face_boxes, confidences

def process_image(image_bgr, face_net, age_net, gender_net, conf_thresh, padding, box_color, box_thick):
    annotated = image_bgr.copy()
    h, w = image_bgr.shape[:2]
    face_boxes, confidences = detect_faces(face_net, image_bgr, conf_thresh)
    
    results = []
    
    for idx, (box, face_conf) in enumerate(zip(face_boxes, confidences)):
        x1, y1, x2, y2 = box
        
        # Crop face with padding
        fx1 = max(0, x1 - padding)
        fy1 = max(0, y1 - padding)
        fx2 = min(w - 1, x2 + padding)
        fy2 = min(h - 1, y2 + padding)
        
        face_crop = image_bgr[fy1:fy2, fx1:fx2]
        if face_crop.size == 0:
            continue
            
        # Blob for Age & Gender
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Gender inference
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()[0]
        gender_idx = int(gender_preds.argmax())
        gender = GENDER_LIST[gender_idx]
        gender_prob = float(gender_preds[gender_idx])
        
        # Age inference
        age_net.setInput(blob)
        age_preds = age_net.forward()[0]
        age_idx = int(age_preds.argmax())
        age = AGE_LIST[age_idx]
        age_prob = float(age_preds[age_idx])
        
        # Draw bounding box & text
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thick)
        label = f"{gender}, {age}"
        cv2.putText(
            annotated, label,
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (255, 255, 255), 3, cv2.LINE_AA
        )
        cv2.putText(
            annotated, label,
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75, box_color, 2, cv2.LINE_AA
        )
        
        results.append({
            "face_index": idx + 1,
            "box": box,
            "face_confidence": face_conf,
            "gender": gender,
            "gender_confidence": gender_prob,
            "gender_distribution": {GENDER_LIST[i]: float(gender_preds[i]) for i in range(len(GENDER_LIST))},
            "age": age,
            "age_confidence": age_prob,
            "age_distribution": {AGE_LIST[i]: float(age_preds[i]) for i in range(len(AGE_LIST))},
            "face_crop": cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        })
        
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, results

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/face-id.png", width=64)
    st.title("Settings & Controls")
    
    input_mode = st.radio(
        "Choose Input Source",
        ["🖼️ Sample Images", "📤 Upload Image", "📸 Live Camera Snap"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("Model Hyperparameters")
    conf_threshold = st.slider("Face Detector Confidence", min_value=0.2, max_value=0.99, value=0.7, step=0.05)
    padding = st.slider("Face ROI Padding (px)", min_value=0, max_value=60, value=20, step=5)
    
    st.markdown("---")
    st.subheader("Visual Overlay")
    box_color_hex = st.color_picker("Bounding Box Color", "#00FF66")
    box_thickness = st.slider("Box Line Thickness", 1, 8, 2)
    
    # Convert hex color to BGR for OpenCV
    h_col = box_color_hex.lstrip('#')
    bgr_color = tuple(int(h_col[i:i+2], 16) for i in (4, 2, 0))

# Main UI Header
st.markdown('<div class="main-header">🧑‍🤝‍🧑 Gender & Age Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by OpenCV DNN, TensorFlow SSD Face Detector, and Caffe Levi & Hassner CNNs</div>', unsafe_allow_html=True)

try:
    face_net, age_net, gender_net = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Load image based on selection
image_bgr = None
image_title = "Input Image"

if input_mode == "🖼️ Sample Images":
    samples = {
        "Young Woman (girl1.jpg)": "girl1.jpg",
        "Young Child (girl2.jpg)": "girl2.jpg",
        "Boy (kid1.jpg)": "kid1.jpg",
        "Elderly Gentleman (man1.jpg)": "man1.jpg",
        "Adult Woman (woman1.jpg)": "woman1.jpg",
        "Mature Woman (woman3.jpg)": "woman3.jpg",
        "Animated Character (minion.jpg)": "minion.jpg"
    }
    selected_sample = st.selectbox("Select a sample image:", list(samples.keys()))
    sample_file = os.path.join(SCRIPT_DIR, samples[selected_sample])
    if os.path.exists(sample_file):
        image_bgr = cv2.imread(sample_file)
        image_title = selected_sample

elif input_mode == "📤 Upload Image":
    uploaded = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, 1)
        image_title = uploaded.name

elif input_mode == "📸 Live Camera Snap":
    camera_photo = st.camera_input("Take a photo using your webcam")
    if camera_photo is not None:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, 1)
        image_title = "Webcam Snapshot"

if image_bgr is not None:
    # Run processing
    annotated_rgb, detections = process_image(
        image_bgr, face_net, age_net, gender_net,
        conf_threshold, padding, bgr_color, box_thickness
    )
    
    # Top stats
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Detected Faces", len(detections))
    with col_m2:
        if detections:
            primary = detections[0]
            st.metric("Primary Gender", f"{primary['gender']} ({primary['gender_confidence']*100:.1f}%)")
        else:
            st.metric("Primary Gender", "None")
    with col_m3:
        if detections:
            primary = detections[0]
            st.metric("Primary Age", f"{primary['age']} yrs")
        else:
            st.metric("Primary Age", "None")
    with col_m4:
        st.metric("Detector Model", "SSD + 2x Caffe")
        
    st.markdown("---")
    
    # Visual comparison columns
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📷 Original Image")
        orig_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image(orig_rgb, width="stretch")
        
    with col_right:
        st.subheader("🎯 Detected & Classified")
        st.image(annotated_rgb, width="stretch")
        
        # Download button for annotated image
        is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
        if is_success:
            st.download_button(
                label="📥 Download Annotated Image",
                data=buffer.tobytes(),
                file_name="annotated_detection.jpg",
                mime="image/jpeg",
                width="stretch"
            )

    # Detailed face analysis
    if detections:
        st.markdown("---")
        st.subheader("🔍 Individual Face Breakdown & Probability Distributions")
        
        for det in detections:
            with st.expander(f"👤 Face #{det['face_index']}: {det['gender']} | {det['age']} (Face Confidence: {det['face_confidence']*100:.1f}%)", expanded=True):
                col_f1, col_f2, col_f3 = st.columns([1, 1.5, 2])
                
                with col_f1:
                    st.write("**Cropped ROI**")
                    st.image(det["face_crop"], width="stretch")
                    st.caption(f"Bounding Box: {det['box']}")
                    
                with col_f2:
                    st.write("**Gender Probabilities**")
                    st.progress(det["gender_distribution"]["Male"], text=f"Male: {det['gender_distribution']['Male']*100:.1f}%")
                    st.progress(det["gender_distribution"]["Female"], text=f"Female: {det['gender_distribution']['Female']*100:.1f}%")
                    
                with col_f3:
                    st.write("**Age Range Distribution**")
                    age_data = det["age_distribution"]
                    st.bar_chart(age_data, height=180)
    else:
        st.info("No face detected in this image matching the current confidence threshold. Try reducing the 'Face Detector Confidence' slider in the sidebar.")
else:
    st.info("👈 Please select or upload an image from the sidebar to begin.")
