"""dashboard.py — Streamlit UI for AI Livestock Biometric & Health Monitoring Platform"""

import streamlit as st
from PIL import Image
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")

SAMPLES = {
    "Sample 1":  "breed_ayrshire.jpg",
    "Sample 2":  "breed_Holstein_Friesian_cattle.jpg",
    "Sample 3":  "breed_Jersey_cattle.jpg",
    "Sample 4":  "breed_Brown_Swiss_cattle.jpg",
    "Sample 5":  "disease_healthy.jpg",
    "Sample 6":  "disease_lumpy.jpg",
    "Sample 14": "disease_h14.jpg",
    "Sample 15": "disease_h15.jpg",
    "Sample 16": "disease_h16.jpg",
    "Sample 17": "disease_h17.jpg",
    "Sample 18": "disease_h18.jpg",
    "Sample 19": "disease_l19.jpg",
    "Sample 20": "disease_l20.jpg",
    "Sample 21": "disease_l21.jpg",
    "Sample 22": "disease_l22.jpg",
    "Sample 23": "disease_l23.jpg",
    "Sample 7":  "muzzle_7.jpg",
    "Sample 8":  "muzzle_8.jpg",
    "Sample 9":  "muzzle_9.jpg",
    "Sample 10": "muzzle_10.jpg",
    "Sample 11": "muzzle_11.jpg",
    "Sample 12": "muzzle_12.jpg",
    "Sample 13": "muzzle_13.jpg",
}

st.set_page_config(
    page_title="AI Livestock Monitor",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .healthy-badge  { color: #22c55e; font-weight: bold; font-size: 1.2em; }
    .disease-badge  { color: #ef4444; font-weight: bold; font-size: 1.2em; }
    .info-badge     { color: #3b82f6; font-weight: bold; font-size: 1.2em; }
    div[data-testid="stImage"] img { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    from src.inference import predict_breed, predict_disease, identify_muzzle, estimate_weight
    return predict_breed, predict_disease, identify_muzzle, estimate_weight


with st.sidebar:
    st.title("🐄 AI Livestock Monitor")
    st.caption("ROSCODE TECH · Cattle Biometric Platform")
    st.divider()
    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "🔍 Full Analysis",
        "🐄 Breed ID",
        "🦠 Disease Detection",
        "👃 Muzzle Biometrics",
        "⚖️ Weight Estimator",
    ])
    st.divider()
    st.caption("EfficientNet-B0 · ResNet50 · SimCLR · FAISS")


try:
    predict_breed, predict_disease, identify_muzzle, estimate_weight = load_models()
    models_ok = True
except Exception as e:
    st.error(f"Model loading failed: {e}")
    models_ok = False


def sample_picker(suggested_keys, key_prefix):
    """Show sample image thumbnails; user clicks one or uploads their own."""
    st.markdown("**Try a sample image:**")
    cols = st.columns(len(suggested_keys))
    chosen = None
    for col, name in zip(cols, suggested_keys):
        path = os.path.join(SAMPLE_DIR, SAMPLES[name])
        if os.path.exists(path):
            thumb = Image.open(path).convert("RGB")
            col.image(thumb, use_column_width=True)
            if col.button(name.split()[0], key=f"{key_prefix}_{name}"):
                chosen = thumb
                st.session_state[f"{key_prefix}_sample"] = thumb
                st.session_state[f"{key_prefix}_label"]  = name

    # Persist selection across reruns
    if chosen is None and f"{key_prefix}_sample" in st.session_state:
        chosen = st.session_state[f"{key_prefix}_sample"]

    st.markdown("**Or upload your own:**")
    f = st.file_uploader("", type=["jpg", "jpeg", "png"], key=f"{key_prefix}_upload")
    if f:
        chosen = Image.open(f).convert("RGB")
        st.session_state[f"{key_prefix}_sample"] = chosen
        st.session_state[f"{key_prefix}_label"]  = f.name

    if chosen:
        label = st.session_state.get(f"{key_prefix}_label", "Selected image")
        st.image(chosen, caption=label, use_column_width=True)

    return chosen


# ── Dashboard ─────────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.title("🐄 AI Livestock Biometric & Health Monitoring")
    st.markdown("Real-time cattle identification, breed classification, and disease detection.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Breed Accuracy", "94.6%", "EfficientNet-B0")
    with c2:
        st.metric("Disease Model", "Trained", "EfficientNet-B0")
    with c3:
        st.metric("Muzzle Vectors", "1,309", "FAISS IndexFlatIP")
    with c4:
        st.metric("Models Ready", "3 / 3", "GPU: RTX 2050")

    st.divider()

    st.subheader("Sample Images")
    thumb_cols = st.columns(len(SAMPLES))
    for col, (name, fname) in zip(thumb_cols, SAMPLES.items()):
        path = os.path.join(SAMPLE_DIR, fname)
        if os.path.exists(path):
            col.image(Image.open(path).convert("RGB"), caption=name, use_column_width=True)

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.info("Pick a sample or upload a cattle photo.")
    with col2:
        st.markdown("### 2️⃣ Identify")
        st.info("Muzzle biometrics match the cow via FAISS similarity search.")
    with col3:
        st.markdown("### 3️⃣ Classify")
        st.info("EfficientNet classifies breed (18 classes) and detects lumpy skin disease.")
    with col4:
        st.markdown("### 4️⃣ Report")
        st.info("Combined risk score and full report generated instantly.")


# ── Full Analysis ─────────────────────────────────────────────────────────────
elif page == "🔍 Full Analysis":
    st.title("🔍 Full Cattle Analysis")

    img = sample_picker(
        ["Sample 1", "Sample 2", "Sample 5", "Sample 6", "Sample 7", "Sample 9", "Sample 11"],
        "full"
    )

    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        height_cm = st.number_input("Height (cm)", 80.0, 200.0, 130.0, step=1.0)
    with col_w2:
        volume_l = st.number_input("Volume (L)", 50.0, 1500.0, 400.0, step=10.0)
    with col_w3:
        feed_type = st.selectbox("Feed type", ["Grass (0)", "Grain (1)", "Mixed (2)"])
        feed_idx  = int(feed_type.split("(")[1][0])
    with col_w4:
        sunlight = st.slider("Sunlight (h/day)", 0.0, 16.0, 8.0, 0.5)

    if st.button("🚀 Run Full Analysis", disabled=(not models_ok or img is None), type="primary"):
        with st.spinner("Running all models..."):
            breed_res   = predict_breed(img)
            disease_res = predict_disease(img)
            muzzle_res  = identify_muzzle(img)
            weight_res  = estimate_weight(height_cm, volume_l, feed_idx, sunlight)

        st.divider()
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown("**Breed**")
            st.markdown(f"<span class='info-badge'>{breed_res['breed']}</span>", unsafe_allow_html=True)
            st.caption(f"Confidence: {breed_res['confidence']}%")
        with r2:
            is_healthy = disease_res["disease"] == "healthy"
            badge = "healthy-badge" if is_healthy else "disease-badge"
            label = "✅ Healthy" if is_healthy else "⚠️ Lumpy Skin"
            st.markdown("**Disease Status**")
            st.markdown(f"<span class='{badge}'>{label}</span>", unsafe_allow_html=True)
            st.caption(f"Healthy: {disease_res['healthy_prob']}% · Lumpy: {disease_res['lumpy_prob']}%")
        with r3:
            matched = muzzle_res["matched"]
            cow_id  = muzzle_res["identified_as"]
            sim     = muzzle_res["similarity"]
            st.markdown("**Muzzle ID**")
            if matched:
                st.markdown(f"<span class='healthy-badge'>🐄 {cow_id}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='disease-badge'>Unknown</span>", unsafe_allow_html=True)
            st.caption(f"Similarity: {sim:.3f}")
        with r4:
            w = weight_res["estimated_weight_kg"]
            st.markdown("**Est. Weight**")
            st.markdown(f"<span class='info-badge'>{w} kg</span>", unsafe_allow_html=True)

        risk = 0
        if not is_healthy:
            risk += 60
        if disease_res["lumpy_prob"] > 30:
            risk += 20
        risk = min(risk, 100)

        st.divider()
        col_risk, col_breed = st.columns(2)
        with col_risk:
            st.subheader("Health Risk Score")
            st.progress(risk / 100)
            st.markdown(f"**Risk: {risk}%** — {'High ⚠️' if risk > 50 else ('Moderate 🔶' if risk > 20 else 'Low ✅')}")
        with col_breed:
            st.subheader("Breed Probabilities (Top 5)")
            top5 = sorted(breed_res["all_probs"].items(), key=lambda x: -x[1])[:5]
            for breed, prob in top5:
                st.progress(prob / 100, text=f"{breed}: {prob}%")


# ── Breed ID ──────────────────────────────────────────────────────────────────
elif page == "🐄 Breed ID":
    st.title("🐄 Cattle Breed Classification")
    st.markdown("Identifies breed from 18 cattle classes — **94.6% validation accuracy**.")

    img = sample_picker(
        ["Sample 1", "Sample 2", "Sample 3", "Sample 4"],
        "breed"
    )

    if st.button("Classify Breed", disabled=(not models_ok or img is None), type="primary"):
        with st.spinner("Classifying..."):
            result = predict_breed(img)

        st.success(f"**Breed: {result['breed']}** ({result['confidence']}% confident)")
        st.divider()
        for breed, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
            st.progress(prob / 100, text=f"{breed}: {prob}%")


# ── Disease Detection ─────────────────────────────────────────────────────────
elif page == "🦠 Disease Detection":
    st.title("🦠 Disease Detection")
    st.markdown("Detects **Lumpy Skin Disease** vs Healthy cattle.")

    img = sample_picker([
        "Sample 5",  "Sample 14", "Sample 15", "Sample 16", "Sample 17", "Sample 18",
        "Sample 6",  "Sample 19", "Sample 20", "Sample 21", "Sample 22", "Sample 23",
    ], "disease")

    if st.button("Detect Disease", disabled=(not models_ok or img is None), type="primary"):
        with st.spinner("Analyzing..."):
            result = predict_disease(img)

        is_healthy = result["disease"] == "healthy"
        if is_healthy:
            st.success(f"✅ **HEALTHY** — Confidence: {result['healthy_prob']}%")
        else:
            st.error(f"⚠️ **LUMPY SKIN DISEASE DETECTED** — Confidence: {result['lumpy_prob']}%")
            st.warning("**Action Required:** Isolate the animal and consult a veterinarian immediately.")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Healthy Probability", f"{result['healthy_prob']}%")
            st.progress(result["healthy_prob"] / 100)
        with col2:
            st.metric("Lumpy Skin Probability", f"{result['lumpy_prob']}%")
            st.progress(result["lumpy_prob"] / 100)


# ── Muzzle Biometrics ─────────────────────────────────────────────────────────
elif page == "👃 Muzzle Biometrics":
    st.title("👃 Muzzle Biometric Identification")
    st.markdown("Identifies individual cattle via muzzle patterns — **SimCLR + FAISS** · 1,309 indexed vectors.")

    img = sample_picker(["Sample 7", "Sample 8", "Sample 9", "Sample 10", "Sample 11", "Sample 12", "Sample 13"], "muzzle")
    top_k = st.slider("Top-K matches", 1, 10, 3)

    if st.button("Identify Cow", disabled=(not models_ok or img is None), type="primary"):
        with st.spinner("Searching FAISS index..."):
            result = identify_muzzle(img, top_k=top_k)

        if result["matched"]:
            st.success(f"🐄 **Identified as: {result['identified_as']}** (similarity: {result['similarity']:.3f})")
        else:
            st.warning(f"❓ **Unknown cow** — best similarity {result['similarity']:.3f} below threshold.")

        st.divider()
        for i, match in enumerate(result["top_matches"], 1):
            sim = match["similarity"]
            st.progress(max(0.0, float(sim)), text=f"#{i} {match['cow_id']} — similarity: {sim:.4f}")


# ── Weight Estimator ──────────────────────────────────────────────────────────
elif page == "⚖️ Weight Estimator":
    st.title("⚖️ Weight Estimation")
    st.markdown("Estimates cattle weight from biometric measurements using MLP regression.")

    col1, col2 = st.columns(2)
    with col1:
        height_cm = st.number_input("Height (cm)", 80.0, 200.0, 130.0, step=1.0)
        volume_l  = st.number_input("Volume (L)", 50.0, 1500.0, 400.0, step=10.0)
    with col2:
        feed_type = st.selectbox("Feed type", ["Grass (0)", "Grain (1)", "Mixed (2)"])
        feed_idx  = int(feed_type.split("(")[1][0])
        sunlight  = st.slider("Sunlight (h/day)", 0.0, 16.0, 8.0, 0.5)

    if st.button("Estimate Weight", disabled=(not models_ok), type="primary"):
        with st.spinner("Computing..."):
            result = estimate_weight(height_cm, volume_l, feed_idx, sunlight)

        w = result["estimated_weight_kg"]
        st.success(f"**Estimated Weight: {w} kg**")
        st.metric("Weight", f"{w} kg", f"Height: {height_cm} cm · Volume: {volume_l} L")

        if w < 150:
            cat = "Calf"
        elif w < 300:
            cat = "Young"
        elif w < 500:
            cat = "Adult"
        else:
            cat = "Heavy"
        st.info(f"Category: **{cat}**")
