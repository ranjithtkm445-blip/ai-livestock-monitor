"""app.py — AI Livestock Platform for Render (fast startup, lazy heavy imports)"""

import os, sys, json, threading
import gradio as gr
from PIL import Image

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMB_DIR    = os.path.join(BASE_DIR, "outputs", "embeddings")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")

BREED_MODEL_PATH   = os.path.join(MODELS_DIR, "breed_classifier.pt")
DISEASE_MODEL_PATH = os.path.join(MODELS_DIR, "disease_detector.pt")
MUZZLE_MODEL_PATH  = os.path.join(MODELS_DIR, "arcface_muzzle.pt")
WEIGHT_MODEL_PATH  = os.path.join(MODELS_DIR, "weight_estimator.pt")
FAISS_INDEX_PATH   = os.path.join(EMB_DIR, "muzzle_index.faiss")
FAISS_META_PATH    = os.path.join(EMB_DIR, "muzzle_metadata.json")
SIMILARITY_THRESHOLD = 0.75

_models = {}
_ready  = False

def _load_all():
    global _ready
    # Heavy imports — happen in background thread, not at startup
    import torch, torch.nn as nn, torch.nn.functional as F
    import timm, faiss, numpy as np
    from torchvision import transforms

    device = torch.device("cpu")

    tfm_std = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tfm_muz = transforms.Compose([
        transforms.Resize((112, 112)), transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])

    class SimCLRModel(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.backbone  = timm.create_model("resnet50", pretrained=False, num_classes=0)
            fd = self.backbone.num_features
            self.projector = nn.Sequential(
                nn.Linear(fd, fd), nn.BatchNorm1d(fd), nn.ReLU(),
                nn.Linear(fd, emb_dim), nn.BatchNorm1d(emb_dim),
            )
        def forward(self, x):
            h = self.backbone(x)
            return h, F.normalize(self.projector(h), dim=1)

    class WeightMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,1),
            )
        def forward(self, x): return self.net(x).squeeze(1)

    def detect_backbone(sd):
        return "convnext_tiny" if any(k.startswith("stages.") for k in sd) else "efficientnet_b0"

    # Breed
    ckpt = torch.load(BREED_MODEL_PATH, map_location=device, weights_only=False)
    bb   = ckpt.get("backbone") or detect_backbone(ckpt["model_state"])
    m    = timm.create_model(bb, pretrained=False, num_classes=ckpt["num_classes"])
    m.load_state_dict(ckpt["model_state"]); m.eval()
    _models["breed"] = (m, ckpt["classes"], tfm_std, device)

    # Disease
    ckpt = torch.load(DISEASE_MODEL_PATH, map_location=device, weights_only=False)
    bb   = ckpt.get("backbone") or detect_backbone(ckpt["model_state"])
    m    = timm.create_model(bb, pretrained=False, num_classes=len(ckpt["classes"]))
    m.load_state_dict(ckpt["model_state"]); m.eval()
    _models["disease"] = (m, ckpt["classes"], tfm_std, device, F)

    # Muzzle
    ckpt = torch.load(MUZZLE_MODEL_PATH, map_location=device, weights_only=False)
    m    = SimCLRModel(ckpt["emb_dim"])
    m.load_state_dict(ckpt["model_state"]); m.eval()
    idx  = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH) as f: meta = json.load(f)
    _models["muzzle"] = (m, idx, meta, tfm_muz, device, F, np)

    # Weight
    ckpt = torch.load(WEIGHT_MODEL_PATH, map_location=device, weights_only=False)
    m    = WeightMLP(ckpt["input_dim"])
    m.load_state_dict(ckpt["model_state"]); m.eval()
    _models["weight"] = (m, ckpt, device, torch)

    _ready = True
    print("All models loaded.")


def predict_breed(img):
    if not _ready: return "Loading models...", "Please wait"
    import torch, torch.nn.functional as F
    model, classes, tfm, device = _models["breed"]
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    idx  = probs.argmax().item()
    top5 = sorted(zip(classes, probs.tolist()), key=lambda x: -x[1])[:5]
    label = f"{classes[idx]} — {round(probs[idx].item()*100,2)}% confidence"
    top5_text = "\n".join([f"{b}: {round(p*100,2)}%" for b,p in top5])
    return label, top5_text


def predict_disease(img):
    if not _ready: return "Loading models...", "—", "—"
    import torch, torch.nn.functional as F
    model, classes, tfm, device, _ = _models["disease"]
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].tolist()
    is_healthy = probs[0] > probs[1]
    status = f"HEALTHY — {round(probs[0]*100,2)}%" if is_healthy else f"LUMPY SKIN DETECTED — {round(probs[1]*100,2)}%"
    return status, f"{round(probs[0]*100,2)}%", f"{round(probs[1]*100,2)}%"


def identify_muzzle(img, top_k=3):
    if not _ready: return "Loading models...", "—", "—"
    import torch, torch.nn.functional as F
    model, index, meta, tfm, device, F, np = _models["muzzle"]
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        h, _ = model(x)
        emb  = F.normalize(h, dim=1).cpu().numpy().astype("float32")
    dists, idxs = index.search(emb, top_k)
    id_to_cow = meta["id_to_cow"]; labels = meta["labels"]
    matches = [(id_to_cow.get(str(labels[i]), f"COW-{labels[i]}"), round(float(d),4))
               for d, i in zip(dists[0], idxs[0])]
    sim     = matches[0][1]
    matched = sim >= SIMILARITY_THRESHOLD
    status  = f"Identified: {matches[0][0]} (sim={sim:.3f})" if matched else f"Unknown (sim={sim:.3f})"
    top3    = "\n".join([f"#{i+1} {m[0]} — {m[1]:.4f}" for i,m in enumerate(matches)])
    return status, f"{sim:.4f}", top3


def estimate_weight(height_cm, volume_l, feed_type, sunlight):
    if not _ready: return "Loading models..."
    import torch
    model, ckpt, device, torch = _models["weight"]
    feed_map = {"Grass": 0, "Grain": 1, "Mixed": 2}
    feed_idx = feed_map.get(feed_type, 0)
    means, stds = ckpt["means"], ckpt["stds"]
    feat_cols   = ckpt["feature_cols"]
    raw  = {"Height (cm)": height_cm, "Volume (liter)": volume_l,
            "Type of feed": feed_idx, "Sunlight intensity": sunlight}
    vals = [(raw.get(c,0.0)-means.get(c,0.0))/max(stds.get(c,1.0),1e-8) for c in feat_cols]
    x    = torch.tensor([vals], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x).item()
    w   = round(max(50.0, min(1200.0, pred)), 1)
    cat = "Calf" if w<150 else ("Young" if w<300 else ("Adult" if w<500 else "Heavy"))
    return f"{w} kg — {cat}"


def run_full(img, height_cm, volume_l, feed_type, sunlight):
    if img is None: return "Upload an image","","",""
    pil = Image.fromarray(img).convert("RGB")
    b,_  = predict_breed(pil)
    d,hp,lp = predict_disease(pil)
    mu,sim,_ = identify_muzzle(pil, top_k=1)
    w = estimate_weight(height_cm, volume_l, feed_type, sunlight)
    return b, d, mu, w


def _samples(fnames):
    return [os.path.join(SAMPLE_DIR,f) for f in fnames if os.path.exists(os.path.join(SAMPLE_DIR,f))]

BREED_SAMPLES   = _samples(["breed_ayrshire.jpg","breed_Holstein_Friesian_cattle.jpg",
                             "breed_Jersey_cattle.jpg","breed_Brown_Swiss_cattle.jpg"])
DISEASE_SAMPLES = _samples(["disease_healthy.jpg","disease_h14.jpg","disease_h15.jpg",
                             "disease_lumpy.jpg","disease_l19.jpg","disease_l20.jpg"])
MUZZLE_SAMPLES  = _samples([f"muzzle_{i}.jpg" for i in range(7,14)])

breed_tab = gr.Interface(
    fn=lambda img: predict_breed(Image.fromarray(img).convert("RGB")) if img is not None else ("—","—"),
    inputs=gr.Image(label="Cattle Image"),
    outputs=[gr.Textbox(label="Breed"), gr.Textbox(label="Top-5", lines=5)],
    examples=[[p] for p in BREED_SAMPLES], title="🐄 Breed Classification", allow_flagging="never",
)
disease_tab = gr.Interface(
    fn=lambda img: predict_disease(Image.fromarray(img).convert("RGB")) if img is not None else ("—","—","—"),
    inputs=gr.Image(label="Cattle Image"),
    outputs=[gr.Textbox(label="Status"), gr.Textbox(label="Healthy %"), gr.Textbox(label="Lumpy %")],
    examples=[[p] for p in DISEASE_SAMPLES], title="🦠 Disease Detection", allow_flagging="never",
)
muzzle_tab = gr.Interface(
    fn=lambda img: identify_muzzle(Image.fromarray(img).convert("RGB")) if img is not None else ("—","—","—"),
    inputs=gr.Image(label="Muzzle Image"),
    outputs=[gr.Textbox(label="ID"), gr.Textbox(label="Similarity"), gr.Textbox(label="Top-3",lines=3)],
    examples=[[p] for p in MUZZLE_SAMPLES], title="👃 Muzzle Biometrics", allow_flagging="never",
)
weight_tab = gr.Interface(
    fn=estimate_weight,
    inputs=[gr.Slider(80,200,130,label="Height (cm)"), gr.Slider(50,1500,400,label="Volume (L)"),
            gr.Dropdown(["Grass","Grain","Mixed"],value="Grass",label="Feed"),
            gr.Slider(0,16,8,step=0.5,label="Sunlight (h/day)")],
    outputs=gr.Textbox(label="Weight"), title="⚖️ Weight Estimator", allow_flagging="never",
)
full_tab = gr.Interface(
    fn=run_full,
    inputs=[gr.Image(label="Cattle Image"), gr.Slider(80,200,130,label="Height (cm)"),
            gr.Slider(50,1500,400,label="Volume (L)"),
            gr.Dropdown(["Grass","Grain","Mixed"],value="Grass",label="Feed"),
            gr.Slider(0,16,8,step=0.5,label="Sunlight (h/day)")],
    outputs=[gr.Textbox(label="Breed"), gr.Textbox(label="Disease"),
             gr.Textbox(label="Muzzle ID"), gr.Textbox(label="Weight")],
    examples=[[BREED_SAMPLES[0],130,400,"Grass",8]] if BREED_SAMPLES else [],
    title="🔍 Full Analysis", allow_flagging="never",
)

demo = gr.TabbedInterface(
    [full_tab, breed_tab, disease_tab, muzzle_tab, weight_tab],
    ["🔍 Full Analysis","🐄 Breed","🦠 Disease","👃 Muzzle","⚖️ Weight"],
    title="🐄 AI Livestock Monitor — ROSCODE TECH",
)

threading.Thread(target=_load_all, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
