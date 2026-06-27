import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
import streamlit as st
import numpy as np
from PIL import Image
import h5py
import cv2
import html as html_lib

# ── Page config — must be first Streamlit call ───────────────────────────────
st.set_page_config(
    page_title="LeafScan AI — Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Apply Inter only to text — NOT with * which breaks Material Icons */
body, p, h1, h2, h3, h4, h5, h6, label,
.stMarkdown, .stText, [data-testid="stMarkdownContainer"] * {
    font-family: 'Inter', sans-serif !important;
}

/* Hide only footer and hamburger menu */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* Sidebar light green background */
[data-testid="stSidebar"] {
    background-color: #f0fdf4 !important;
}

/* Main background */
[data-testid="stAppViewContainer"] > .main {
    background: #f8fafc;
}

/* Spinner color */
.stSpinner > div { border-top-color: #2D6A4F !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
def _load_weights(model, h5_path):
    with h5py.File(h5_path, 'r') as f:
        wg = f['model_weights']
        dn = model.get_layer('densenet121')
        dn_grp = wg['densenet121']
        for layer in dn.layers:
            if layer.name not in dn_grp or not layer.weights:
                continue
            lg = dn_grp[layer.name]
            for w in layer.weights:
                key = w.name.split(':')[0].split('/')[-1]
                if key in lg:
                    w.assign(np.array(lg[key][()], dtype='float32'))
        for h5_name in ['batch_normalization_4', 'batch_normalization_5', 'dense_9', 'dense_10']:
            layer = model.get_layer(h5_name)
            model_key = list(wg[h5_name].keys())[0]
            lg = wg[h5_name][model_key][h5_name]
            for w in layer.weights:
                key = w.name.split(':')[0].split('/')[-1]
                if key in lg:
                    w.assign(np.array(lg[key][()], dtype='float32'))

@st.cache_resource
def load_model():
    base = DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3))
    model = Sequential([
        base,
        GlobalAveragePooling2D(name='global_average_pooling2d_5'),
        BatchNormalization(name='batch_normalization_4'),
        Dropout(0.5, name='dropout_7'),
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense_9'),
        BatchNormalization(name='batch_normalization_5'),
        Dropout(0.5, name='dropout_8'),
        Dense(38, activation='softmax', name='dense_10'),
    ])
    model(tf.zeros((1, 224, 224, 3)))
    _load_weights(model, "plant_disease_densenet.h5")
    return model

model = load_model()

# ── Data ──────────────────────────────────────────────────────────────────────
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]
plant_names = sorted(set(
    n.split("___")[0].replace("_(including_sour)", "").replace("_", " ").replace(",", "").strip()
    for n in class_names
))


# ── Sidebar ───────────────────────────────────────────────────────────────────
# Build diseases grouped by plant
diseases_by_plant = {}
for name in class_names:
    plant_raw, cond_raw = name.split("___")
    plant = plant_raw.replace("_(including_sour)", "").replace("_", " ").replace(",", "").strip()
    cond  = cond_raw.replace("_", " ").strip()
    diseases_by_plant.setdefault(plant, []).append(cond)

with st.sidebar:
    st.markdown("## 🌿 LeafScan AI")
    st.caption("Plant Disease Intelligence")
    st.divider()
    st.markdown("**🌱 Detectable Diseases**")
    st.caption(f"{len(class_names)} conditions across {len(plant_names)} plants")
    st.markdown("")

    for plant, diseases in sorted(diseases_by_plant.items()):
        with st.expander(f"🪴 {plant}  ({len(diseases)})"):
            for d in diseases:
                icon = "✅" if d.lower() == "healthy" else "🔴"
                st.markdown(f"<small>{icon} {d}</small>", unsafe_allow_html=True)

    st.divider()
    st.warning("For educational & research purposes only. Do not use for agricultural decisions.")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1B4332 0%,#2D6A4F 45%,#40916C 100%);
            border-radius:24px; padding:3rem 2rem 2.5rem; text-align:center;
            margin-bottom:2rem; box-shadow:0 20px 60px rgba(27,67,50,0.22);
            position:relative; overflow:hidden;">
    <div style="position:absolute;top:-50px;right:-50px;width:220px;height:220px;
                background:rgba(255,255,255,0.03);border-radius:50%;"></div>
    <div style="position:absolute;bottom:-70px;left:-40px;width:280px;height:280px;
                background:rgba(255,255,255,0.03);border-radius:50%;"></div>
    <div style="font-size:3.2rem; position:relative; margin-bottom:0.6rem;">🌿</div>
    <h1 style="color:white; font-size:2.8rem; font-weight:800; margin:0 0 0.5rem;
               letter-spacing:-1.5px; line-height:1.1; position:relative;
               font-family:'Inter',sans-serif;">
        Plant Disease&nbsp;<span style="color:#95D5B2;">Detector</span>
    </h1>
    <p style="color:rgba(149,213,178,0.85); font-size:1.05rem; margin:0 0 2rem;
              font-weight:400; position:relative;">
        Upload a leaf image · Get instant AI-powered disease diagnosis
    </p>
    <div style="display:flex; justify-content:center; gap:0; flex-wrap:wrap; position:relative;">
        <div style="padding:0 2.5rem; border-right:1px solid rgba(255,255,255,0.15);">
            <div style="color:white; font-size:1.9rem; font-weight:800; line-height:1;">38</div>
            <div style="color:rgba(149,213,178,0.65); font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:1.5px; margin-top:0.25rem;">Disease Classes</div>
        </div>
        <div style="padding:0 2.5rem; border-right:1px solid rgba(255,255,255,0.15);">
            <div style="color:white; font-size:1.9rem; font-weight:800; line-height:1;">96.6%</div>
            <div style="color:rgba(149,213,178,0.65); font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:1.5px; margin-top:0.25rem;">Test Accuracy</div>
        </div>
        <div style="padding:0 2.5rem; border-right:1px solid rgba(255,255,255,0.15);">
            <div style="color:white; font-size:1.9rem; font-weight:800; line-height:1;">14</div>
            <div style="color:rgba(149,213,178,0.65); font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:1.5px; margin-top:0.25rem;">Plant Species</div>
        </div>
        <div style="padding:0 2.5rem;">
            <div style="color:white; font-size:1.4rem; font-weight:800; line-height:1.4;">DenseNet121</div>
            <div style="color:rgba(149,213,178,0.65); font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:1.5px; margin-top:0.25rem;">Architecture</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────
def is_likely_leaf(image: Image.Image) -> tuple:
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([100, 255, 255]))
    veg_ratio = float(mask.sum()) / (255.0 * mask.size)
    return veg_ratio >= 0.08, veg_ratio

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def parse_class(raw: str) -> tuple:
    plant_raw, cond_raw = raw.split("___") if "___" in raw else (raw, "Unknown")
    plant = plant_raw.replace("_(including_sour)", "").replace("_", " ").replace(",", "").strip()
    condition = cond_raw.replace("_", " ").strip().rstrip()
    healthy = condition.lower() == "healthy"
    return plant, condition, healthy


# ── Upload card ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=['jpg', 'jpeg', 'png', 'jfif', 'webp'],
    help="Use a clear close-up photo with a plain background for best results",
)


# ── Results ───────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("""
        <div style="background:white; border-radius:20px; padding:1.5rem;
                    box-shadow:0 4px 24px rgba(0,0,0,0.06);">
            <div style="font-size:0.72rem; font-weight:700; color:#9CA3AF; text-transform:uppercase;
                        letter-spacing:2px; margin-bottom:1rem;">UPLOADED IMAGE</div>
        """, unsafe_allow_html=True)
        st.image(image, width='stretch')
        w, h = image.size
        st.markdown(f"""
        <div style="display:flex; gap:1rem; margin-top:0.75rem; flex-wrap:wrap;">
            <div style="background:#f0fdf4; border-radius:8px; padding:0.3rem 0.75rem;
                        font-size:0.78rem; color:#166534; font-weight:500;">
                📐 {w} × {h} px
            </div>
            <div style="background:#f0fdf4; border-radius:8px; padding:0.3rem 0.75rem;
                        font-size:0.78rem; color:#166534; font-weight:500;">
                🖼️ {uploaded_file.type.split('/')[-1].upper()}
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_result:
        with st.spinner("🔬 Analyzing leaf..."):
            is_leaf, veg_ratio = is_likely_leaf(image)
            processed  = preprocess_image(image)
            prediction = model.predict(processed, verbose=0)
            pred_index = int(np.argmax(prediction))
            confidence = float(prediction[0][pred_index])

        if not is_leaf:
            card = f"""
            <style>*{{font-family:'Inter',sans-serif;box-sizing:border-box;}}</style>
            <div style="background:white;border-radius:20px;padding:2rem;
                        box-shadow:0 4px 24px rgba(0,0,0,0.06);border-top:4px solid #EF4444;">
                <div style="font-size:2.5rem;margin-bottom:1rem;">&#x1F6AB;</div>
                <div style="font-size:1.35rem;font-weight:800;color:#DC2626;margin-bottom:0.6rem;">
                    Not a Plant Leaf
                </div>
                <div style="color:#6B7280;font-size:0.9rem;line-height:1.65;margin-bottom:1.5rem;">
                    Only <strong style="color:#374151;">{veg_ratio:.1%}</strong> of pixels contain
                    plant-like color &mdash; below the 8% threshold required.
                </div>
                <div style="background:#FEF2F2;border-radius:12px;padding:1rem 1.2rem;
                            border-left:4px solid #FCA5A5;">
                    <div style="font-size:0.8rem;font-weight:700;color:#991B1B;
                                text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;">
                        Tips for better results
                    </div>
                    <ul style="color:#7F1D1D;font-size:0.83rem;line-height:1.7;margin:0;padding-left:1.1rem;">
                        <li>Use a close-up photo of a single leaf</li>
                        <li>Ensure good lighting</li>
                        <li>Use a plain white or neutral background</li>
                        <li>Avoid blurry or dark images</li>
                    </ul>
                </div>
            </div>
            """
            st.html(card)

        else:
            plant, condition, healthy = parse_class(class_names[pred_index])
            conf_pct = int(confidence * 100)

            if healthy:
                top_border="#22C55E"; badge_bg="#DCFCE7"; badge_fg="#166534"
                badge_txt="&#x2705; Healthy"
                bar_grad="linear-gradient(90deg,#16A34A,#4ADE80)"
                note_bg="#F0FDF4"; note_border="#86EFAC"; note_fg="#166534"
                note_text="&#x1F331; This plant appears to be in <strong>good health</strong>. Keep up with regular care and monitoring."
            elif conf_pct >= 60:
                top_border="#F59E0B"; badge_bg="#FEF3C7"; badge_fg="#92400E"
                badge_txt="&#x26A0;&#xFE0F; Disease Detected"
                bar_grad="linear-gradient(90deg,#D97706,#FCD34D)"
                note_bg="#FFFBEB"; note_border="#FCD34D"; note_fg="#78350F"
                note_text="&#x1F52C; <strong>Recommendation:</strong> Consult an agronomist or plant specialist for treatment advice."
            else:
                top_border="#EF4444"; badge_bg="#FEE2E2"; badge_fg="#991B1B"
                badge_txt="&#x26A0;&#xFE0F; Disease Detected"
                bar_grad="linear-gradient(90deg,#DC2626,#F87171)"
                note_bg="#EFF6FF"; note_border="#93C5FD"; note_fg="#1E3A8A"
                note_text="&#x1F4A1; <strong>Low confidence</strong> &mdash; try a clearer image with better lighting and a plain background."

            # Build top-3 rows
            top3_idx = np.argsort(prediction[0])[::-1][:3]
            rows_html = ""
            for i, idx in enumerate(top3_idx):
                p, c, _ = parse_class(class_names[idx])
                pct_raw  = float(prediction[0][idx]) * 100
                pct_str  = f"{pct_raw:.1f}%" if pct_raw >= 1 else "&lt;1%"
                opacity  = "1" if i == 0 else "0.55" if i == 1 else "0.35"
                weight   = "700" if i == 0 else "400"
                divider  = "<hr style='border:none;border-top:1px solid #F3F4F6;margin:0.15rem 0;'>" if i < 2 else ""
                rows_html += f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:0.35rem 0;opacity:{opacity};">
                    <div style="font-size:0.82rem;color:#374151;font-weight:{weight};">
                        {html_lib.escape(p)} &middot; {html_lib.escape(c)}
                    </div>
                    <div style="font-size:0.82rem;font-weight:700;color:#1B4332;
                                min-width:42px;text-align:right;">{pct_str}</div>
                </div>{divider}"""

            card = f"""
            <style>*{{font-family:'Inter',sans-serif;box-sizing:border-box;}}</style>
            <div style="background:white;border-radius:20px;padding:2rem;
                        box-shadow:0 4px 24px rgba(0,0,0,0.06);border-top:4px solid {top_border};">
                <span style="background:{badge_bg};color:{badge_fg};padding:0.35rem 1rem;
                             border-radius:50px;font-size:0.78rem;font-weight:700;
                             text-transform:uppercase;letter-spacing:1px;">{badge_txt}</span>
                <div style="margin-top:1.4rem;">
                    <div style="font-size:0.7rem;color:#9CA3AF;text-transform:uppercase;
                                letter-spacing:2px;margin-bottom:0.2rem;">Plant</div>
                    <div style="font-size:1.65rem;font-weight:800;color:#111827;line-height:1.2;">
                        {html_lib.escape(plant)}
                    </div>
                </div>
                <div style="margin-top:1rem;">
                    <div style="font-size:0.7rem;color:#9CA3AF;text-transform:uppercase;
                                letter-spacing:2px;margin-bottom:0.2rem;">Condition</div>
                    <div style="font-size:1.05rem;font-weight:600;color:#374151;">
                        {html_lib.escape(condition)}
                    </div>
                </div>
                <div style="margin-top:1.4rem;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:0.45rem;">
                        <div style="font-size:0.7rem;color:#9CA3AF;text-transform:uppercase;
                                    letter-spacing:2px;">Confidence</div>
                        <div style="font-size:1rem;font-weight:800;color:#111827;">{conf_pct}%</div>
                    </div>
                    <div style="background:#F3F4F6;border-radius:10px;height:10px;overflow:hidden;">
                        <div style="width:{conf_pct}%;height:100%;border-radius:10px;
                                    background:{bar_grad};"></div>
                    </div>
                </div>
                <div style="margin-top:1.4rem;background:{note_bg};border-radius:12px;
                            padding:0.9rem 1.1rem;border-left:4px solid {note_border};
                            font-size:0.83rem;color:{note_fg};line-height:1.55;">
                    {note_text}
                </div>
                <div style="margin-top:1.4rem;">
                    <div style="font-size:0.7rem;color:#9CA3AF;text-transform:uppercase;
                                letter-spacing:2px;margin-bottom:0.6rem;">Top Predictions</div>
                    <div style="background:#F9FAFB;border-radius:12px;padding:0.75rem 1rem;">
                        {rows_html}
                    </div>
                </div>
            </div>
            """
            st.html(card)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3.5rem; padding:1.75rem 1rem;
            border-top:1px solid #E5E7EB; color:#9CA3AF; font-size:0.82rem; line-height:1.8;">
    Built with
    <span style="color:#2D6A4F; font-weight:700;">DenseNet121</span>
    trained on the
    <span style="color:#2D6A4F; font-weight:700;">PlantVillage</span>
    dataset &nbsp;·&nbsp;
    <span style="color:#2D6A4F; font-weight:700;">96.6%</span> test accuracy
    &nbsp;·&nbsp; For research purposes only
</div>
""", unsafe_allow_html=True)

