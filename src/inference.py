# src/inference.py — Shared inference utilities: model loading and prediction helpers

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import timm
import faiss

from config import *

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


# ── ArcFace + Muzzle model (must mirror train_muzzle.py) ─────────────────────
class ArcFaceHead(nn.Module):
    def __init__(self, emb_dim, num_classes, s=32.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x, labels=None):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        cosine = F.linear(x_norm, w_norm)
        if labels is None:
            return cosine * self.s
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return (one_hot * (theta + self.m) + (1 - one_hot) * theta).cos() * self.s


class SimCLRModel(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.backbone  = timm.create_model("resnet50", pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
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


# ── Model registry (lazy-loaded singletons) ───────────────────────────────────
_models = {}


def _detect_backbone(state_dict):
    keys = list(state_dict.keys())
    if any(k.startswith("stages.") for k in keys):
        return "convnext_tiny"
    if any(k.startswith("conv_stem") for k in keys):
        return "efficientnet_b0"
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
        ckpt = torch.load(MUZZLE_MODEL_PATH, map_location=device, weights_only=False)
        model = SimCLRModel(ckpt["emb_dim"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH) as f:
            meta = json.load(f)
        _models["muzzle"] = (model, index, meta)
    return _models["muzzle"]


def _load_weight():
    if "weight" not in _models:
        ckpt = torch.load(WEIGHT_MODEL_PATH, map_location=device, weights_only=False)
        model = WeightMLP(ckpt["input_dim"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _models["weight"] = (model, ckpt)
    return _models["weight"]


# ── Public prediction functions ───────────────────────────────────────────────
def predict_breed(img: Image.Image) -> dict:
    model, classes = _load_breed()
    x = tfm_standard(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()
    return {
        "breed":      classes[idx],
        "confidence": round(probs[idx].item() * 100, 2),
        "all_probs":  {c: round(p.item() * 100, 2) for c, p in zip(classes, probs)},
    }


def predict_disease(img: Image.Image) -> dict:
    model, classes = _load_disease()
    x = tfm_standard(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()
    return {
        "disease":    classes[idx],
        "confidence": round(probs[idx].item() * 100, 2),
        "healthy_prob": round(probs[0].item() * 100, 2),
        "lumpy_prob":   round(probs[1].item() * 100, 2),
    }


def identify_muzzle(img: Image.Image, top_k: int = 3) -> dict:
    model, index, meta = _load_muzzle()
    x = tfm_muzzle(img).unsqueeze(0).to(device)
    with torch.no_grad():
        h, _ = model(x)
        emb = F.normalize(h, dim=1).cpu().numpy().astype("float32")
    distances, indices = index.search(emb, top_k)
    id_to_cow = meta["id_to_cow"]
    labels    = meta["labels"]
    matches = []
    for dist, idx in zip(distances[0], indices[0]):
        cow_class = str(labels[idx])
        cow_name  = id_to_cow.get(cow_class, f"COW-{cow_class}")
        matches.append({"cow_id": cow_name, "similarity": round(float(dist), 4)})
    top = matches[0]
    return {
        "identified_as": top["cow_id"],
        "similarity":    top["similarity"],
        "matched":       top["similarity"] >= SIMILARITY_THRESHOLD,
        "top_matches":   matches,
    }


def estimate_weight(height_cm: float, volume_l: float,
                    feed_type: int = 0, sunlight: float = 5.0) -> dict:
    model, ckpt = _load_weight()
    means = ckpt["means"]
    stds  = ckpt["stds"]
    feat_cols = ckpt["feature_cols"]

    raw = {"Height (cm)": height_cm, "Volume (liter)": volume_l,
           "Type of feed": feed_type, "Sunlight intensity": sunlight}

    vals = []
    for col in feat_cols:
        v = raw.get(col, 0.0)
        v = (v - means.get(col, 0.0)) / max(stds.get(col, 1.0), 1e-8)
        vals.append(v)

    x = torch.tensor([vals], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x).item()

    pred = max(WEIGHT_MIN_KG, min(WEIGHT_MAX_KG, pred))
    return {"estimated_weight_kg": round(pred, 1)}
