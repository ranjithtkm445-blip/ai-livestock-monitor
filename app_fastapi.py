# app.py — FastAPI backend for AI Livestock Biometric & Health Monitoring Platform

import os, sys, io, uuid
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from typing import Optional

from src.inference import predict_breed, predict_disease, identify_muzzle, estimate_weight
from config import MAX_IMAGE_SIZE_MB

app = FastAPI(
    title="AI Livestock Biometric & Health Monitoring Platform",
    description="Multi-task CV system for cattle identification, breed, disease and weight",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory animal registry (replace with PostgreSQL in production) ──────────
animal_registry: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def read_image(file: UploadFile) -> Image.Image:
    contents = file.file.read()
    if len(contents) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {MAX_IMAGE_SIZE_MB} MB limit")
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")


# ── Status ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Status"])
def root():
    return {"status": "running", "platform": "AI Livestock Monitor v1.0"}


@app.get("/health", tags=["Status"])
def health():
    import torch
    return {
        "status":    "healthy",
        "device":    "cuda" if torch.cuda.is_available() else "cpu",
        "gpu":       torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Module 3: Muzzle Identification ──────────────────────────────────────────
@app.post("/identify", tags=["Biometric"])
async def identify_cow(file: UploadFile = File(...), top_k: int = Query(3, ge=1, le=10)):
    """Identify a cow from its muzzle image using ArcFace + FAISS."""
    img = read_image(file)
    result = identify_muzzle(img, top_k=top_k)
    return {"module": "muzzle_identification", "timestamp": datetime.utcnow().isoformat(), **result}


# ── Module 4: Breed Classification ───────────────────────────────────────────
@app.post("/breed", tags=["Classification"])
async def classify_breed(file: UploadFile = File(...)):
    """Classify cattle breed from image using ConvNeXt Tiny."""
    img = read_image(file)
    result = predict_breed(img)
    return {"module": "breed_classification", "timestamp": datetime.utcnow().isoformat(), **result}


# ── Module 7: Disease Detection ───────────────────────────────────────────────
@app.post("/disease", tags=["Health"])
async def detect_disease(file: UploadFile = File(...)):
    """Detect visible disease (Healthy / Lumpy Skin Disease) from image."""
    img = read_image(file)
    result = predict_disease(img)
    return {"module": "disease_detection", "timestamp": datetime.utcnow().isoformat(), **result}


# ── Module 6: Weight Estimation ───────────────────────────────────────────────
class WeightInput(BaseModel):
    height_cm: float
    volume_l:  float
    feed_type: int   = 0
    sunlight:  float = 5.0


@app.post("/weight", tags=["Health"])
def estimate_cow_weight(data: WeightInput):
    """Estimate cattle weight from body measurements."""
    result = estimate_weight(
        height_cm=data.height_cm,
        volume_l=data.volume_l,
        feed_type=data.feed_type,
        sunlight=data.sunlight,
    )
    return {"module": "weight_estimation", "timestamp": datetime.utcnow().isoformat(), **result}


# ── Full Pipeline ─────────────────────────────────────────────────────────────
@app.post("/analyze", tags=["Pipeline"])
async def full_analysis(
    file: UploadFile = File(...),
    height_cm: float = Query(130.0),
    volume_l:  float = Query(300.0),
):
    """Run the complete pipeline: Identify → Breed → Disease → Weight → Report."""
    img = read_image(file)

    identity = identify_muzzle(img)
    breed    = predict_breed(img)
    disease  = predict_disease(img)
    weight   = estimate_weight(height_cm=height_cm, volume_l=volume_l)

    risk_score = 0
    if disease["disease"] != "healthy":
        risk_score += 40
    if weight["estimated_weight_kg"] < 200:
        risk_score += 20

    report = {
        "report_id":      str(uuid.uuid4())[:8].upper(),
        "timestamp":      datetime.utcnow().isoformat(),
        "animal_id":      identity["identified_as"],
        "identified":     identity["matched"],
        "similarity":     identity["similarity"],
        "breed":          breed["breed"],
        "breed_conf":     breed["confidence"],
        "disease":        disease["disease"],
        "disease_conf":   disease["confidence"],
        "weight_kg":      weight["estimated_weight_kg"],
        "risk_score":     min(risk_score, 100),
        "risk_level":     "HIGH" if risk_score >= 40 else "MEDIUM" if risk_score >= 20 else "LOW",
        "recommendations": _get_recommendations(disease["disease"], weight["estimated_weight_kg"]),
    }

    cow_id = identity["identified_as"]
    if cow_id not in animal_registry:
        animal_registry[cow_id] = {"visits": []}
    animal_registry[cow_id]["visits"].append(report)

    return report


def _get_recommendations(disease: str, weight_kg: float) -> list:
    recs = []
    if disease != "healthy":
        recs.append("Immediate veterinary consultation recommended.")
        recs.append("Isolate animal from rest of herd.")
    if weight_kg < 200:
        recs.append("Increase protein-rich feed intake.")
        recs.append("Schedule nutritional assessment.")
    if not recs:
        recs.append("Animal appears healthy. Maintain regular monitoring.")
    return recs


# ── Registry ──────────────────────────────────────────────────────────────────
@app.get("/animals", tags=["Registry"])
def list_animals():
    return {
        "total": len(animal_registry),
        "animals": [{"cow_id": k, "visits": len(v["visits"])} for k, v in animal_registry.items()],
    }


@app.get("/animals/{cow_id}", tags=["Registry"])
def get_animal(cow_id: str):
    if cow_id not in animal_registry:
        raise HTTPException(404, f"Animal '{cow_id}' not found")
    return animal_registry[cow_id]


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True)
