import os, io, csv, time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
import streamlit as st
from ultralytics import YOLO

# -------------------- CONFIG --------------------
DEFAULT_MODEL_PATH = "best.pt"
IMGSZ = 320
DEFAULT_TOPK = 5

st.set_page_config(page_title="Casting QA", page_icon="🧪", layout="wide")

# -------------------- Custom CSS for Styling and Animations --------------------
st.markdown("""
<style>
/* General Styling */
.stApp {
    background-color: #f5f7fa;
    font-family: 'Inter', sans-serif;
}
h1 {
    color: #1a3c34;
    font-weight: 700;
    text-align: center;
    animation: fadeIn 1s ease-in-out;
}
h2 {
    color: #2e5b52;
    font-weight: 600;
}
.sidebar .sidebar-content {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.stButton>button {
    background-color: #1a3c34;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    transition: transform 0.2s ease-in-out;
}
.stButton>button:hover {
    transform: scale(1.05);
    background-color: #2e5b52;
}
.stFileUploader {
    border: 2px dashed #1a3c34;
    border-radius: 10px;
    padding: 15px;
    background-color: #ffffff;
    transition: border-color 0.3s ease;
}
.stFileUploader:hover {
    border-color: #2e5b52;
}

/* Image Animation */
.stImage img {
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    animation: slideIn 0.5s ease-out;
}

/* Prediction Card */
.prediction-card {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    animation: fadeInUp 0.5s ease-out;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes slideIn {
    from { transform: translateX(-20px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}
@keyframes fadeInUp {
    from { transform: translateY(20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

/* Responsive Design */
@media (max-width: 768px) {
    .stColumns {
        flex-direction: column;
    }
    .stImage img {
        max-width: 100%;
    }
}
</style>
""", unsafe_allow_html=True)

# -------------------- UTILITIES --------------------
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = YOLO(path)
    model.model.eval()
    return model

def to_pil(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode != "RGB":
        img = ImageOps.grayscale(img).convert("RGB")
    return img

def pil_to_tensor(img: Image.Image, imgsz: int) -> torch.Tensor:
    img = img.resize((imgsz, imgsz))
    arr = (np.asarray(img).astype(np.float32) / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("Settings", anchor=False)
    model_path = st.text_input("Model path", DEFAULT_MODEL_PATH, help="Enter the path to your YOLO model weights.")
    topk = st.slider("Top-K predictions to display", 1, 5, DEFAULT_TOPK, 1, help="Select how many top predictions to show.")
    device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0, help="Choose the device for inference (auto selects cuda if available).")
    st.caption("Tip: Set a secret **MODEL_URL** to auto-download weights at startup.")

# Try downloading if needed
def maybe_download_model(target_path: str) -> None:
    url = st.secrets.get("MODEL_URL", os.getenv("MODEL_URL", "")).strip() if hasattr(st, "secrets") else os.getenv("MODEL_URL", "").strip()
    if not url or Path(target_path).exists():
        return
    import urllib.request
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    with st.spinner("Downloading model weights…"):
        urllib.request.urlretrieve(url, target_path)
maybe_download_model(model_path)

# Load model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Pick device
if device_choice == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = device_choice
model.to(device)
class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names

# -------------------- MAIN UI --------------------
st.title("Casting Defect Classification")
st.caption("Upload casting images to classify as OK or DEFECT. Results appear instantly.")

# File uploader with enhanced UX
uploads = st.file_uploader(
    "Upload image(s)",
    type=["jpg", "jpeg", "png", "bmp", "tif"],
    accept_multiple_files=True,
    help="Upload one or more images in supported formats."
)
if not uploads:
    st.info("Please upload one or more images to begin analysis.")
    st.stop()

# Layout for images and predictions
cols = st.columns([1, 1], gap="medium")
with cols[0]:
    st.subheader("Uploaded Images", anchor=False)
    for f in uploads:
        st.image(to_pil(f.getvalue()), caption=f.name, use_container_width=True)

with cols[1]:
    st.subheader("Prediction Results", anchor=False)
    with st.spinner("Processing images..."):
        for f in uploads:
            pil = to_pil(f.getvalue())
            start = time.time()
            results = model.predict(source=[np.array(pil)], imgsz=IMGSZ, device=device, verbose=False)
            elapsed_ms = (time.time() - start) * 1000

            r = results[0]
            if r.probs is None:
                st.error("No probabilities returned; ensure classification weights (yolov8*-cls).")
                continue

            probs = r.probs.data.cpu().numpy()
            idx = np.argsort(-probs)[:topk]
            names = [class_names[i] for i in idx]
            scores = [float(probs[i]) for i in idx]

            # Display predictions in a card-like format
            with st.container(border=True):
                st.markdown(f"**{f.name}** — Inference time: **{elapsed_ms:.1f} ms**")
                st.metric(
                    label="Top Prediction",
                    value=names[0],
                    delta=f"{scores[0]*100:.2f}% confidence",
                    delta_color="normal"
                )
                st.dataframe(
                    {"Class": names, "Confidence": [f"{s*100:.2f}%" for s in scores]},
                    use_container_width=True,
                    hide_index=True
                )

st.markdown("---")
st.caption("Note: This app performs **image-level classification** (e.g., def_front vs ok_front). Keep validation/test sets augmentation-free for fair evaluation.")
