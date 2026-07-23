"""gradio_app.py — HuggingFace Spaces Gradio interface for AI Livestock Platform"""

import os, sys, json
import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import faiss
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
EMB_DIR     = os.path.join(BASE_DIR, "outputs", "embeddings")
SAMPLE_DIR  = os.path.join(BASE_DIR, "sample_images")

BREED_MODEL_PATH   = os.path.join(MODELS_DIR, "breed_classifier.pt")
DISEASE_MODEL_PATH = os.path.join(MODELS_DIR, "disease_detector.pt")
MUZZLE_MODEL_PATH  = os.path.join(MODELS_DIR, "arcface_muzzle.pt")
WEIGHT_MODEL_PATH  = os.path.join(MODELS_DIR, "weight_estimator.pt")
FAISS_INDEX_PATH   = os.path.join(EMB_DIR, "muzzle_index.faiss")
FAISS_META_PATH    = os.path.join(EMB_DIR, "muzzle_metadata.json")
SIMILARITY_THRESHOLD = 0.75

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms ────────────────────────────────────────────────────────────────
tfm_standard = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
tfm_muzzle = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ── Model definitions ─────────────────────────────────────────────────────────
class SimCLRModel(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.backbone  = timm.create_model("resnet50", pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, emb_dim), nn.BatchNorm1d(emb_dim),
        )

    def forward(self, x):
        h = self.backbone(x)
        return h, F.normalize(self.projector(h), dim=1)


class WeightMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ── Lazy model loading ────────────────────────────────────────────────────────
_models = {}

def _detect_backbone(state_dict):
    keys = list(state_dict.keys())
    if any(k.startswith("stages.") for k in keys):
        return "convnext_tiny"
    return "efficientnet_b0"

def _load_breed():
    if "breed" not in _models:
        ckpt = torch.load(BREED_MODEL_PATH, map_location=device, weights_only=False)
        backbone = ckpt.get("backbone") or _detect_backbone(ckpt["model_state"])
        model = timm.create_model(backbone, pretrained=False,
                                  num_classes=ckpt["num_classes"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _models["breed"] = (model, ckpt["classes"])
    return _models["breed"]

def _load_disease():
    if "disease" not in _models:
        ckpt = torch.load(DISEASE_MODEL_PATH, map_location=device, weights_only=False)
        backbone = ckpt.get("backbone") or _detect_backbone(ckpt["model_state"])
        model = timm.create_model(backbone, pretrained=False,
                                  num_classes=len(ckpt["classes"])).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _models["disease"] = (model, ckpt["classes"])
    return _models["disease"]

def _load_muzzle():
    if "muzzle" not in _models:
        ckpt  = torch.load(MUZZLE_MODEL_PATH, map_location=device, weights_only=False)
        model = SimCLRModel(ckpt["emb_dim"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_META_PATH) as f:
            meta = json.load(f)
        _models["muzzle"] = (model, index, meta)
    return _models["muzzle"]

def _load_weight():
    if "weight" not in _models:
        ckpt  = torch.load(WEIGHT_MODEL_PATH, map_location=device, weights_only=False)
        model = WeightMLP(ckpt["input_dim"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _models["weight"] = (model, ckpt)
    return _models["weight"]


# ── Inference functions ───────────────────────────────────────────────────────
def predict_breed(img: Image.Image):
    model, classes = _load_breed()
    x = tfm_standard(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    idx = probs.argmax().item()
    top5 = sorted(zip(classes, probs.tolist()), key=lambda x: -x[1])[:5]
    return classes[idx], round(probs[idx].item() * 100, 2), top5

def predict_disease(img: Image.Image):
    model, classes = _load_disease()
    x = tfm_standard(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].tolist()
    idx = int(probs.index(max(probs)))
    return classes[idx], round(probs[0]*100, 2), round(probs[1]*100, 2)

def identify_muzzle(img: Image.Image, top_k=3):
    model, index, meta = _load_muzzle()
    x = tfm_muzzle(img).unsqueeze(0).to(device)
    with torch.no_grad():
        h, _ = model(x)
        emb = F.normalize(h, dim=1).cpu().numpy().astype("float32")
    dists, idxs = index.search(emb, top_k)
    id_to_cow = meta["id_to_cow"]
    labels    = meta["labels"]
    matches = []
    for dist, i in zip(dists[0], idxs[0]):
        cow_name = id_to_cow.get(str(labels[i]), f"COW-{labels[i]}")
        matches.append((cow_name, round(float(dist), 4)))
    top_sim = matches[0][1]
    matched = top_sim >= SIMILARITY_THRESHOLD
    return matches[0][0], top_sim, matched, matches

def estimate_weight(height_cm, volume_l, feed_idx, sunlight):
    model, ckpt = _load_weight()
    means, stds = ckpt["means"], ckpt["stds"]
    feat_cols = ckpt["feature_cols"]
    raw = {"Height (cm)": height_cm, "Volume (liter)": volume_l,
           "Type of feed": feed_idx, "Sunlight intensity": sunlight}
    vals = [(raw.get(c, 0.0) - means.get(c, 0.0)) / max(stds.get(c, 1.0), 1e-8)
            for c in feat_cols]
    x = torch.tensor([vals], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x).item()
    return round(max(50.0, min(1200.0, pred)), 1)


# ── Sample images ─────────────────────────────────────────────────────────────
def _samples(pattern_list):
    out = []
    for fname in pattern_list:
        path = os.path.join(SAMPLE_DIR, fname)
        if os.path.exists(path):
            out.append(path)
    return out

BREED_SAMPLES   = _samples(["breed_ayrshire.jpg","breed_Holstein_Friesian_cattle.jpg",
                             "breed_Jersey_cattle.jpg","breed_Brown_Swiss_cattle.jpg"])
DISEASE_SAMPLES = _samples(["disease_healthy.jpg","disease_h14.jpg","disease_h15.jpg",
                             "disease_h16.jpg","disease_h17.jpg","disease_h18.jpg",
                             "disease_lumpy.jpg","disease_l19.jpg","disease_l20.jpg",
                             "disease_l21.jpg","disease_l22.jpg","disease_l23.jpg"])
MUZZLE_SAMPLES  = _samples([f"muzzle_{i}.jpg" for i in range(7, 14)])


# ── Tab: Breed Classification ─────────────────────────────────────────────────
def run_breed(img):
    if img is None:
        return "No image provided.", {}
    breed, conf, top5 = predict_breed(Image.fromarray(img).convert("RGB"))
    label = f"🐄 **{breed}** — {conf}% confidence"
    probs = {b: round(p * 100, 2) for b, p in top5}
    return label, probs

breed_tab = gr.Interface(
    fn=run_breed,
    inputs=gr.Image(label="Upload or select a cattle image"),
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Label(label="Top-5 Breed Probabilities", num_top_classes=5),
    ],
    examples=[[p] for p in BREED_SAMPLES],
    title="🐄 Cattle Breed Classification",
    description="Identifies cattle breed from 18 classes using EfficientNet-B0 (94.6% val accuracy).",
    allow_flagging="never",
)


# ── Tab: Disease Detection ────────────────────────────────────────────────────
def run_disease(img):
    if img is None:
        return "No image provided.", 0.0, 0.0
    disease, healthy_p, lumpy_p = predict_disease(Image.fromarray(img).convert("RGB"))
    is_healthy = disease == "healthy"
    status = f"✅ **HEALTHY** — {healthy_p}% confidence" if is_healthy \
             else f"⚠️ **LUMPY SKIN DISEASE DETECTED** — {lumpy_p}% confidence"
    return status, healthy_p, lumpy_p

disease_tab = gr.Interface(
    fn=run_disease,
    inputs=gr.Image(label="Upload cattle image"),
    outputs=[
        gr.Markdown(label="Disease Status"),
        gr.Number(label="Healthy Probability (%)"),
        gr.Number(label="Lumpy Skin Probability (%)"),
    ],
    examples=[[p] for p in DISEASE_SAMPLES],
    title="🦠 Disease Detection",
    description="Detects Lumpy Skin Disease vs Healthy cattle using EfficientNet-B0 with weighted loss.",
    allow_flagging="never",
)


# ── Tab: Muzzle Biometrics ────────────────────────────────────────────────────
def run_muzzle(img):
    if img is None:
        return "No image provided.", "—", "—"
    cow_id, sim, matched, matches = identify_muzzle(Image.fromarray(img).convert("RGB"), top_k=3)
    status = f"🐄 **Identified: {cow_id}** (similarity: {sim:.3f})" if matched \
             else f"❓ **Unknown** — best similarity {sim:.3f} (below threshold 0.75)"
    top3 = "\n".join([f"#{i+1} {m[0]} — {m[1]:.4f}" for i, m in enumerate(matches)])
    return status, f"{sim:.4f}", top3

muzzle_tab = gr.Interface(
    fn=run_muzzle,
    inputs=gr.Image(label="Upload muzzle image"),
    outputs=[
        gr.Markdown(label="Identification Result"),
        gr.Textbox(label="Top Similarity Score"),
        gr.Textbox(label="Top-3 Matches", lines=3),
    ],
    examples=[[p] for p in MUZZLE_SAMPLES],
    title="👃 Muzzle Biometric Identification",
    description="Identifies individual cattle via muzzle patterns — SimCLR + FAISS (1,309 indexed vectors).",
    allow_flagging="never",
)


# ── Tab: Weight Estimator ─────────────────────────────────────────────────────
def run_weight(height_cm, volume_l, feed_type, sunlight):
    feed_map = {"Grass": 0, "Grain": 1, "Mixed": 2}
    feed_idx = feed_map.get(feed_type, 0)
    w = estimate_weight(height_cm, volume_l, feed_idx, sunlight)
    cat = "Calf" if w < 150 else ("Young" if w < 300 else ("Adult" if w < 500 else "Heavy"))
    return f"**{w} kg** — {cat}"

weight_tab = gr.Interface(
    fn=run_weight,
    inputs=[
        gr.Slider(80, 200, value=130, label="Height (cm)"),
        gr.Slider(50, 1500, value=400, label="Volume (L)"),
        gr.Dropdown(["Grass", "Grain", "Mixed"], value="Grass", label="Feed Type"),
        gr.Slider(0, 16, value=8, step=0.5, label="Sunlight (h/day)"),
    ],
    outputs=gr.Markdown(label="Estimated Weight"),
    examples=[[130, 400, "Grass", 8], [150, 600, "Grain", 10], [110, 250, "Mixed", 6]],
    title="⚖️ Weight Estimation",
    description="Estimates cattle weight from biometric measurements using MLP regression.",
    allow_flagging="never",
)


# ── Full Analysis Tab ─────────────────────────────────────────────────────────
def run_full(img, height_cm, volume_l, feed_type, sunlight):
    if img is None:
        return "Please upload an image.", "", "", ""
    pil = Image.fromarray(img).convert("RGB")

    breed, conf, _    = predict_breed(pil)
    disease, h_p, l_p = predict_disease(pil)
    cow_id, sim, matched, _ = identify_muzzle(pil, top_k=1)
    feed_map = {"Grass": 0, "Grain": 1, "Mixed": 2}
    w = estimate_weight(height_cm, volume_l, feed_map.get(feed_type, 0), sunlight)

    is_healthy = disease == "healthy"
    risk = 0
    if not is_healthy: risk += 60
    if l_p > 30:       risk += 20
    risk = min(risk, 100)
    risk_label = "🔴 High" if risk > 50 else ("🟠 Moderate" if risk > 20 else "🟢 Low")

    breed_out   = f"🐄 **{breed}** ({conf}%)"
    disease_out = f"✅ Healthy ({h_p}%)" if is_healthy else f"⚠️ Lumpy Skin ({l_p}%)"
    muzzle_out  = f"🐄 {cow_id} (sim={sim:.3f})" if matched else f"❓ Unknown (sim={sim:.3f})"
    summary = (
        f"**Weight:** {w} kg\n\n"
        f"**Health Risk:** {risk_label} ({risk}%)\n\n"
        f"> ⚠️ Isolate and consult vet immediately." if not is_healthy else
        f"**Weight:** {w} kg\n\n**Health Risk:** {risk_label} ({risk}%)"
    )
    return breed_out, disease_out, muzzle_out, summary

full_tab = gr.Interface(
    fn=run_full,
    inputs=[
        gr.Image(label="Cattle Image"),
        gr.Slider(80, 200, value=130, label="Height (cm)"),
        gr.Slider(50, 1500, value=400, label="Volume (L)"),
        gr.Dropdown(["Grass", "Grain", "Mixed"], value="Grass", label="Feed Type"),
        gr.Slider(0, 16, value=8, step=0.5, label="Sunlight (h/day)"),
    ],
    outputs=[
        gr.Markdown(label="🐄 Breed"),
        gr.Markdown(label="🦠 Disease"),
        gr.Markdown(label="👃 Muzzle ID"),
        gr.Markdown(label="📋 Summary"),
    ],
    examples=[
        [BREED_SAMPLES[0],   130, 400, "Grass", 8],
        [DISEASE_SAMPLES[0], 140, 500, "Grain", 10],
        [DISEASE_SAMPLES[6], 120, 350, "Mixed", 6],
        [MUZZLE_SAMPLES[0],  130, 400, "Grass", 8],
    ] if BREED_SAMPLES and DISEASE_SAMPLES and MUZZLE_SAMPLES else [],
    title="🔍 Full Cattle Analysis",
    description="Run all models on one image — breed, disease, muzzle ID, and weight estimation.",
    allow_flagging="never",
)


# ── Launch ────────────────────────────────────────────────────────────────────
demo = gr.TabbedInterface(
    [full_tab, breed_tab, disease_tab, muzzle_tab, weight_tab],
    ["🔍 Full Analysis", "🐄 Breed ID", "🦠 Disease", "👃 Muzzle", "⚖️ Weight"],
    title="🐄 AI Livestock Biometric & Health Monitoring — ROSCODE TECH",
)

if __name__ == "__main__":
    demo.launch()
