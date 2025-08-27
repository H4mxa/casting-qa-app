# --- YOLOv8 Image Inference App (images only, grayscale-safe, in-memory) ---

import io
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import streamlit as st

# headless backend for any plotting Ultralytics does internally
import matplotlib
matplotlib.use("Agg")

from ultralytics import YOLO
import torch

# -----------------------------
# App settings (defaults match your notebook)
# -----------------------------
PAGE_TITLE = "Casting Defects App"
DEFAULT_WEIGHTS_HINT = "best.pt"  # prefer this exact file if present

DEFAULT_CONF = 0.30
DEFAULT_IOU = 0.50
DEFAULT_IMGSZ = 640

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

# -----------------------------
# Helpers
# -----------------------------
def discover_weights() -> Path | None:
    """Return DEFAULT_WEIGHTS_HINT if present, else the first *.pt in repo root."""
    p = Path(DEFAULT_WEIGHTS_HINT)
    if p.exists():
        return p
    cands = sorted(Path(".").glob("*.pt"))
    return cands[0] if cands else None

def to_png_bytes(arr_rgb: np.ndarray) -> bytes:
    """Encode an RGB numpy array as PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(arr_rgb).save(buf, format="PNG")
    return buf.getvalue()

@st.cache_resource(show_spinner=False)
def load_model_cached() -> tuple[YOLO, str]:
    w = discover_weights()
    if not w:
        st.error(
            "No weights found. Place a YOLOv8 `.pt` file in the repo root "
            f"(e.g., `{DEFAULT_WEIGHTS_HINT}`) and redeploy."
        )
        st.stop()
    model = YOLO(str(w))
    return model, str(w.resolve())

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Model")
model, resolved_path = load_model_cached()
st.sidebar.success(f"Loaded weights: {Path(resolved_path).name}")

# show names **from the model** to avoid any hand-coded mapping mistakes
st.sidebar.write("`model.names`:", model.names)
st.sidebar.caption("Expected order → ['abrasion', 'crack', 'indentation']")

conf_thres = st.sidebar.slider(
    "Confidence threshold",
    0.05, 0.95, DEFAULT_CONF, 0.05,
    help="Minimum confidence to keep a detection. Higher = fewer, more reliable boxes."
)
iou_thres = st.sidebar.slider(
    "IoU (NMS)",
    0.30, 0.90, DEFAULT_IOU, 0.05,
    help="Overlap threshold for Non-Maximum Suppression. Higher = keep more overlapping boxes."
)
imgsz = st.sidebar.selectbox(
    "Image size",
    [320, 512, 640, 768, 896],
    index=[320, 512, 640, 768, 896].index(DEFAULT_IMGSZ),
    help="Input resolution for inference. Larger = more detail (slower)."
)

show_raw = st.sidebar.checkbox("Show raw detections (debug)", value=False)
device_opt = 0 if torch.cuda.is_available() else "cpu"
st.sidebar.write("Device:", device_opt)

# -----------------------------
# Image inference UI
# -----------------------------
st.subheader("Upload image(s) for detection")
imgs = st.file_uploader(
    "Choose image files",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)

run_btn = st.button("Run detection", type="primary", disabled=not imgs)

if run_btn and imgs:
    rendered_rgb_images: list[np.ndarray] = []
    downloadable: list[tuple[str, bytes]] = []

    with st.spinner("Running inference..."):
        for up in imgs:
            raw = up.read()
            pil = Image.open(io.BytesIO(raw))

            # --- Normalize to 3-channel RGB (handles grayscale and paletted) ---
            if pil.mode not in ("RGB",):
                pil = pil.convert("RGB")
            im_rgb = np.array(pil)              # HxWx3, RGB

            # *** CRITICAL: convert to BGR for Ultralytics/OpenCV pipeline ***
            im_bgr = im_rgb[..., ::-1].copy()   # HxWx3, BGR

            # --- YOLO inference (in-memory, no disk writes) ---
            results = model.predict(
                source=im_bgr,          # pass BGR
                imgsz=imgsz,
                conf=conf_thres,
                iou=iou_thres,
                augment=False,
                agnostic_nms=False,     # keep classes distinct
                max_det=300,
                verbose=False,
                save=False,
                device=device_opt
            )

            r = results[0]

            # optional debug: list raw boxes with class ids & mapped names
            if show_raw:
                names = model.names
                rows = []
                for c, cf, xyxy in zip(r.boxes.cls.tolist(),
                                       r.boxes.conf.tolist(),
                                       r.boxes.xyxy.tolist()):
                    rows.append({
                        "cls_id": int(c),
                        "name": names[int(c)],
                        "conf": float(cf),
                        "xyxy": [float(x) for x in xyxy]
                    })
                st.write(rows)

            # Ultralytics r.plot() returns **BGR**; convert to RGB to display
            plotted_bgr = r.plot()
            plotted_rgb = plotted_bgr[..., ::-1].copy()

            rendered_rgb_images.append(plotted_rgb)
            downloadable.append((f"{Path(up.name).stem}_pred.png", to_png_bytes(plotted_rgb)))

    st.success(f"Done. Rendered {len(rendered_rgb_images)} image(s).")

    # Show results
    cols = st.columns(2 if len(rendered_rgb_images) > 1 else 1)
    for i, arr in enumerate(rendered_rgb_images):
        with cols[i % len(cols)]:
            st.image(arr, caption=f"{imgs[i].name}", use_container_width=True)

    # ZIP download of all rendered outputs
    if downloadable:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, data in downloadable:
                zf.writestr(fname, data)
        zbuf.seek(0)
        st.download_button(
            "Download all results (ZIP)",
            data=zbuf.read(),
            file_name="detections.zip",
            mime="application/zip",
        )
else:
    st.info("Upload one or more images, then click **Run detection**.")
