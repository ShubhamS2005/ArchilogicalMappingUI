# =============================
# AgroSensi AI - FINAL app.py
# =============================

import streamlit as st
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import cv2
import os
from ultralytics import YOLO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AgroSensi AI",
    page_icon="🌱",
    layout="wide"
)

# =============================
# UI THEME (SAFE & MODERN)
# =============================
st.markdown("""
<style>
:root {
    --primary: #22c55e;
    --secondary: #a7f3d0;
    --card-bg: rgba(255,255,255,0.03);
    --border: rgba(255,255,255,0.08);
    --text: #e5e7eb;
    --muted: #9ca3af;
}

h1, h2, h3 { color: var(--secondary); }

.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

.stButton > button {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #020617;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.6rem 1.6rem;
}

[data-testid="stFileUploader"] {
    background: var(--card-bg);
    border: 1px dashed var(--border);
    border-radius: 14px;
    padding: 1rem;
}

[data-testid="metric-container"] {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px;
}

img { border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

# =============================
# LOGIN
# =============================
USERNAME, PASSWORD = "admin", "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 AgroSensi AI Login")
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if u == USERNAME and p == PASSWORD:
                st.session_state.logged_in = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()

# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_soil():
    model = tf.keras.models.load_model("soil_resnet50_model.h5")
    labels = json.load(open("class_labels.json"))
    return model, {v:k for k,v in labels.items()}

@st.cache_resource
def load_veg():
    return YOLO("best.pt")

soil_model, idx_to_class = load_soil()
veg_model = load_veg()

# =============================
# SIDEBAR
# =============================
st.sidebar.title("🌿 AgroSensi AI")
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🌱 Soil Classification", "🌿 Vegetation Segmentation", "ℹ️ About"]
)

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# =============================
# HOME
# =============================
if menu == "🏠 Home":
    st.title("🌿 AgroSensi AI")
    st.caption("An Explainable AI Platform for Smart & Sustainable Agriculture")

    st.markdown("""
    <div class="card">
    AgroSensi AI combines **Deep Learning, Explainable AI, and Computer Vision**  
    to deliver **real-world agricultural intelligence**, not just predictions.
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.markdown("<div class='card'><h4>🌱 Soil Intelligence</h4>CNN-based soil understanding with visual explanations.</div>",True)
    c2.markdown("<div class='card'><h4>🌿 Vegetation Analytics</h4>Pixel-level segmentation & coverage estimation.</div>",True)
    c3.markdown("<div class='card'><h4>📄 Research-Ready Reports</h4>Auto-generated, audit-friendly PDFs.</div>",True)


# =============================
# SOIL CLASSIFICATION
# =============================
elif menu == "🌱 Soil Classification":
    st.title("🌱 Soil Classification")
    file = st.file_uploader("Upload soil image", ["jpg","png","jpeg"])

    if file:
        path = f"soil_{file.name}"
        open(path,"wb").write(file.getbuffer())
        st.image(path, width=260)

        def prep(p):
            img = image.load_img(p, target_size=(224,224))
            arr = image.img_to_array(img)/255.0
            return np.expand_dims(arr,0)

        if st.button("🔍 Predict"):
            arr = prep(path)
            preds = soil_model.predict(arr)[0]
            idx = np.argmax(preds)

            st.metric("Predicted Soil", idx_to_class[idx], f"{preds[idx]*100:.2f}%")

            fig, ax = plt.subplots()
            ax.bar(idx_to_class.values(), preds)
            ax.set_ylim(0,1)
            st.pyplot(fig)

        os.remove(path)

# =============================
# VEGETATION SEGMENTATION (FIXED)
# =============================
elif menu == "🌿 Vegetation Segmentation":
    st.title("🌿 Vegetation Segmentation")
    file = st.file_uploader("Upload vegetation image", ["jpg","png","jpeg"])

    if file:
        path = f"veg_{file.name}"
        open(path,"wb").write(file.getbuffer())

        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w,_ = img_rgb.shape
        st.image(img_rgb, caption="Input Image", width=350)

        if st.button("🌿 Run Segmentation"):
            res = veg_model.predict(path, conf=0.25)[0]

            if res.masks is None:
                st.error("No vegetation detected")
                st.stop()

            masks = res.masks.data.cpu().numpy()

            # ✅ FIXED BINARY MASK
            binary = np.zeros((h,w), dtype=np.uint8)
            for m in masks:
                m = cv2.resize(m, (w,h))
                binary = np.logical_or(binary, m > 0.5)

            binary = binary.astype(np.uint8)
            coverage = (binary.sum() / (h*w)) * 100

            # Overlay
            overlay = img_rgb.copy()
            overlay[binary==1] = [0,255,0]
            overlay = cv2.addWeighted(img_rgb,0.7,overlay,0.3,0)

            st.metric("Vegetation Coverage", f"{coverage:.2f}%")

            c1,c2,c3 = st.columns(3)
            c1.image(img_rgb, caption="Input")
            c2.image(binary*255, caption="Binary Mask")
            c3.image(overlay, caption="Overlay")

            # =============================
            # PDF REPORT
            # =============================
            def create_pdf():
                pdf="vegetation_report.pdf"
                doc=SimpleDocTemplate(pdf,pagesize=A4)
                styles=getSampleStyleSheet()
                elems=[
                    Paragraph("<b>Vegetation Segmentation Report</b>",styles["Title"]),
                    Paragraph(f"Coverage: {coverage:.2f}%",styles["Normal"])
                ]
                cv2.imwrite("i.jpg",img_rgb)
                cv2.imwrite("m.jpg",binary*255)
                cv2.imwrite("o.jpg",overlay)

                for f,t in [("i.jpg","Input"),("m.jpg","Mask"),("o.jpg","Overlay")]:
                    elems.append(Paragraph(t,styles["Heading2"]))
                    elems.append(RLImage(f,4*inch,3*inch))

                doc.build(elems)
                return pdf

            pdf = create_pdf()
            with open(pdf,"rb") as f:
                st.download_button("📄 Download Segmentation Report (PDF)", f, file_name="vegetation_report.pdf")

        os.remove(path)

# =============================
# ABOUT (MODERN & HUMAN)
# =============================
else:
    st.title("ℹ️ About AgroSensi AI")

    st.markdown("""
    <div class="card">
    <h3>Why AgroSensi AI?</h3>
    <p>
    Most agriculture-related AI projects stop at predictions.  
    AgroSensi AI was built to go one step further —  
    <b>to explain, visualize, and validate every prediction it makes.</b>
    </p>
    <p>
    The goal is simple:  
    <i>make computer vision models usable, trustworthy, and deployment-ready for agriculture.</i>
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>What Does It Do?</h3>
    <p>
    AgroSensi AI focuses on two core agricultural intelligence tasks:
    </p>
    <p>
    <b>🌱 Soil Classification</b><br>
    A deep learning model identifies soil type from an image and shows 
    <b>how the model arrived at that decision</b> using Grad-CAM heatmaps.
    </p>
    <p>
    <b>🌿 Vegetation Segmentation</b><br>
    A YOLOv8-based segmentation pipeline detects vegetation at the pixel level,
    calculates green cover percentage, and visually overlays the results for validation.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>What Makes It Different?</h3>
    <p>
    This project is not designed as a static demo.
    Every component — inference, visualization, and reporting —
    is structured the way a real system would be.
    </p>
    <p>
    Predictions are supported by:
    <br>• Confidence scores  
    <br>• Visual explanations  
    <br>• Side-by-side result comparisons  
    <br>• Downloadable analysis reports  
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>Engineering Approach</h3>
    <p>
    AgroSensi AI follows a clean and modular design philosophy.
    </p>
    <p>
    Models are loaded once, reused efficiently, and separated from UI logic.
    The application is structured to be easily extended into:
    APIs, dashboards, or cloud-based deployments.
    </p>
    <p>
    This makes the project suitable not just for demonstration,
    but also for real-world experimentation and scaling.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>Who Is This For?</h3>
    <p>
    AgroSensi AI is built with multiple users in mind:
    </p>
    <p>
    • Students exploring applied deep learning  
    <br>• Researchers who need visual validation  
    <br>• Startups working in agri-tech  
    <br>• Anyone interested in explainable computer vision  
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align:center;color:#9ca3af;margin-top:30px;">
    Designed & developed by Shubham.
    </p>
    """, unsafe_allow_html=True)


