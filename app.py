import streamlit as st
import torch
import time
from PIL import Image
from torchvision import transforms
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from distillation import build_teacher_student

DEVICE = "cpu"
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

st.set_page_config(page_title="Efficient ViT Compression", page_icon="⚡", layout="wide")

# ---------- Custom CSS for premium look ----------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background: linear-gradient(180deg, #0e1117 0%, #131722 100%); }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    .hero-subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1.05rem;
        margin-top: 4px;
        margin-bottom: 30px;
    }

    .stat-badge {
        background: rgba(79, 172, 254, 0.1);
        border: 1px solid rgba(79, 172, 254, 0.3);
        border-radius: 12px;
        padding: 14px 20px;
        text-align: center;
        margin: 4px;
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #4facfe;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .model-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
    }
    .model-card-teacher {
        border-top: 3px solid #d62728;
    }
    .model-card-student {
        border-top: 3px solid #00f2fe;
    }
    .model-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 16px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<p class="hero-title">⚡ Efficient ViT Compression</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Token pruning + knowledge distillation — '
            'same task, a fraction of the compute</p>', unsafe_allow_html=True)

# ---------- Top stat badges ----------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-badge"><div class="stat-number">71.6%</div>'
                '<div class="stat-label">FLOPs Reduction</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-badge"><div class="stat-number">2.27x</div>'
                '<div class="stat-label">Faster Inference</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-badge"><div class="stat-number">5.8x</div>'
                '<div class="stat-label">Fewer Parameters</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-badge"><div class="stat-number">+16.4%</div>'
                '<div class="stat-label">Accuracy Gain</div></div>', unsafe_allow_html=True)

st.write("")
st.write("")


@st.cache_resource
def load_models():
    teacher, student = build_teacher_student()
    teacher.load_state_dict(torch.load("checkpoints/teacher.pth", map_location=DEVICE))
    student.load_state_dict(torch.load("checkpoints/student_prune2_keep0.7.pth", map_location=DEVICE))
    teacher.eval()
    student.eval()
    return teacher, student


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform(image).unsqueeze(0)


def predict_with_timing(model, image_tensor, prune_after_layer=None, keep_ratio=0.7):
    with torch.no_grad():
        start = time.time()
        logits = model(image_tensor, prune_after_layer=prune_after_layer, keep_ratio=keep_ratio)
        end = time.time()

    probs = torch.softmax(logits, dim=1)
    pred_class = probs.argmax(dim=1).item()
    confidence = probs[0, pred_class].item()
    latency_ms = (end - start) * 1000
    return CIFAR10_CLASSES[pred_class], confidence, latency_ms


teacher, student = load_models()

st.markdown("### Try it yourself")
uploaded_file = st.file_uploader(
    "Upload an image — works best with simple CIFAR-10-style objects "
    "(planes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    img_col, _ = st.columns([1, 3])
    with img_col:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("")
    image_tensor = preprocess_image(image)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="model-card model-card-teacher">', unsafe_allow_html=True)
        st.markdown('<div class="model-title">🔴 Teacher — Full Model</div>', unsafe_allow_html=True)
        pred, conf, latency = predict_with_timing(teacher, image_tensor)
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", pred.capitalize())
        m2.metric("Confidence", f"{conf*100:.1f}%")
        m3.metric("Latency", f"{latency:.1f} ms")
        st.caption("2.38M params · 154.79M FLOPs")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="model-card model-card-student">', unsafe_allow_html=True)
        st.markdown('<div class="model-title">🔵 Student — Pruned + Distilled</div>', unsafe_allow_html=True)
        pred, conf, latency = predict_with_timing(student, image_tensor,
                                                    prune_after_layer=2, keep_ratio=0.7)
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", pred.capitalize())
        m2.metric("Confidence", f"{conf*100:.1f}%")
        m3.metric("Latency", f"{latency:.1f} ms")
        st.caption("0.80M params · 44.01M FLOPs")
        st.markdown('</div>', unsafe_allow_html=True)

    st.info("Trained on a CIFAR-10 subset for fast CPU training — recognizes 10 general "
            "object categories rather than arbitrary real-world objects.")
else:
    st.markdown(
        '<div style="text-align:center; padding: 60px 20px; color: #6b7280;">'
        '📤 Upload an image above to compare both models side by side'
        '</div>', unsafe_allow_html=True
    )

# ---------- Ablation results section ----------
st.write("---")
st.markdown("### Ablation Study Results")
st.write("Accuracy vs. compute trade-off across different token-pruning ratios:")

plot_col1, plot_col2 = st.columns(2)
with plot_col1:
    if os.path.exists("results/plots/accuracy_vs_flops.png"):
        st.image("results/plots/accuracy_vs_flops.png", use_container_width=True)
with plot_col2:
    if os.path.exists("results/plots/latency_comparison.png"):
        st.image("results/plots/latency_comparison.png", use_container_width=True)