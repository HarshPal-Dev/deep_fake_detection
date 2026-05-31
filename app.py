import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; text-align: center;
    }
    .subtitle { text-align:center; color:#888; font-size:0.95rem; margin-bottom:1rem; }
    .result-box { border-radius:12px; padding:1.5rem; text-align:center; margin:0.8rem 0; }
    .real-box  { background:linear-gradient(135deg,#11998e18,#38ef7d18); border:2px solid #38ef7d; }
    .fake-box  { background:linear-gradient(135deg,#ff416c18,#ff4b2b18); border:2px solid #ff416c; }
    .real-label { font-size:2rem; font-weight:800; color:#38ef7d; letter-spacing:2px; }
    .fake-label { font-size:2rem; font-weight:800; color:#ff416c; letter-spacing:2px; }
    .conf-text  { color:#ccc; font-size:1rem; margin-top:0.3rem; }
    .warn-box   { background:#2a2000; border:1px solid #f0a500;
                  border-radius:8px; padding:0.8rem 1rem; margin:0.5rem 0;
                  color:#f0c040; font-size:0.88rem; }
    .info-card  { background:#1e1e2e; border-left:3px solid #667eea;
                  border-radius:8px; padding:0.8rem 1rem; margin:0.4rem 0; }
    .metric-label { color:#aaa; font-size:0.8rem; }
    .metric-value { color:#fff; font-size:1rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xception_baseline.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 299
MEAN = [0.5, 0.5, 0.5]   # FaceForensics++ standard normalization
STD  = [0.5, 0.5, 0.5]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


@st.cache_resource(show_spinner="Loading model weights…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: `{MODEL_PATH}`  — place `xception_baseline.pth` in the app folder.")
        st.stop()

    # ── Load raw state dict ───────────────────────────────────────────────────
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # Unwrap common checkpoint wrappers
    if isinstance(checkpoint, dict):
        state = (checkpoint.get("model_state_dict")
                 or checkpoint.get("state_dict")
                 or checkpoint.get("model")
                 or checkpoint)
    else:
        # Full model object saved
        checkpoint.eval()
        checkpoint.to(DEVICE)
        return checkpoint, "unknown"

    # Strip any "module." / "model." prefixes (DataParallel artifacts)
    state = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}

    # ── Auto-detect num_classes from fc.weight shape ──────────────────────────
    num_classes = 1  # default: BCE with single output
    if "fc.weight" in state:
        num_classes = state["fc.weight"].shape[0]   # 1 or 2

    # ── Build model with matching num_classes ─────────────────────────────────
    # timm's 'xception' uses the same layer naming as the FaceForensics++
    # reference implementation: conv1, bn1, block1..block12, fc, global_pool
    model = timm.create_model("xception", pretrained=False, num_classes=num_classes)

    missing, unexpected = model.load_state_dict(state, strict=False)

    # Critical check: if fc layer is missing, weights didn't load — stop
    fc_loaded = not any("fc" in k for k in missing)
    if not fc_loaded:
        st.error(
            "⚠️ Model weights could not be matched to the Xception architecture. "
            f"Missing layers include the classifier (fc). "
            f"Missing: {missing[:5]}…"
        )
        st.stop()

    model.eval()
    model.to(DEVICE)
    return model, num_classes


@torch.no_grad()
def predict(model, num_classes, image: Image.Image):
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    logit  = model(tensor)

    if num_classes == 1:
        # BCE training: sigmoid → p(fake)
        prob_fake = torch.sigmoid(logit).squeeze().item()
    else:
        # CrossEntropy training: class 1 = fake
        probs     = torch.softmax(logit, dim=1)[0]
        prob_fake = probs[1].item()

    prob_real = 1.0 - prob_fake
    label     = "FAKE" if prob_fake >= 0.5 else "REAL"
    confidence = prob_fake if label == "FAKE" else prob_real
    return label, confidence, prob_fake


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Model")
    for k, v in [("Architecture", "Xception"), ("Training set", "FaceForensics++ C23"),
                 ("Input size", "299 × 299 px"), ("Loss", "Binary CE")]:
        st.markdown(f"""<div class="info-card">
            <div class="metric-label">{k}</div>
            <div class="metric-value">{v}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Reported AUC")
    for k, v in [("FF++ Validation", "~99%"), ("Celeb-DF v2", "~65–68%"), ("DFDC", "—")]:
        st.markdown(f"""<div class="info-card">
            <div class="metric-label">{k}</div>
            <div class="metric-value">{v}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚠️ Important Limitation")
    st.markdown("""<div class="warn-box">
    This model detects <b>face-swap deepfakes</b> only (trained on FF++ manipulation types:
    Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures).<br><br>
    It is <b>NOT reliable</b> for fully AI-generated images from GANs
    (StyleGAN, DALL-E, Midjourney, Stable Diffusion etc.) — those will typically
    be predicted as <b>REAL</b>. This is a known limitation documented in the paper.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**Device:** {'🟢 GPU' if torch.cuda.is_available() else '🔵 CPU'} `{DEVICE}`")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 DeepFake Image Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a face image to check if it is a face-swap deepfake</div>',
            unsafe_allow_html=True)
st.markdown("---")

# ── Load model ────────────────────────────────────────────────────────────────
model, num_classes = load_model()

# ── Warning banner ────────────────────────────────────────────────────────────
st.markdown("""<div class="warn-box">
⚠️ <b>Scope:</b> This detector is trained on <b>face-swap / face-manipulation deepfakes</b>
(FaceForensics++ dataset). For AI-generated portraits (GAN / diffusion models),
confidence scores are not meaningful — such images are outside the training distribution.
</div>""", unsafe_allow_html=True)

st.markdown("")

# ── Two-column layout ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("#### 📤 Upload Face Image")
    uploaded = st.file_uploader(
        "Upload", type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed"
    )
    st.caption("Best results on face-cropped images · JPG / PNG / WEBP")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption=f"{uploaded.name}  ({image.size[0]}×{image.size[1]} px)",
                 use_container_width=True)

with col_right:
    st.markdown("#### 🧠 Prediction")

    if not uploaded:
        st.info("Upload a face image on the left to run detection.")
    else:
        with st.spinner("Running inference…"):
            label, confidence, prob_fake = predict(model, num_classes, image)

        # ── Result banner ──────────────────────────────────────────────────
        icon = "✅" if label == "REAL" else "🚨"
        css  = "real" if label == "REAL" else "fake"
        st.markdown(f"""
        <div class="result-box {css}-box">
            <div class="{css}-label">{icon} {label}</div>
        </div>""", unsafe_allow_html=True)



# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#555;font-size:0.8rem;'>"
    "Image Deepfake Detection · NIT Jalandhar Major Project · "
    "Paras Garg · Komal Rani · Milan Kapoor · Harsh Pal</p>",
    unsafe_allow_html=True,
)
