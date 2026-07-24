"""api.py — FastAPI REST backend for AI Livestock Monitor"""

import os, sys, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")

SAMPLES = {
    "breed":   ["breed_ayrshire.jpg", "breed_Holstein_Friesian_cattle.jpg",
                 "breed_Jersey_cattle.jpg", "breed_Brown_Swiss_cattle.jpg"],
    "disease": ["disease_healthy.jpg", "disease_h14.jpg", "disease_h15.jpg",
                "disease_lumpy.jpg", "disease_l19.jpg", "disease_l20.jpg"],
    "muzzle":  [f"muzzle_{i}.jpg" for i in range(7, 14)],
}

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    print("Loading models at startup...")
    _load()
    print("All models ready.")
    yield

app = FastAPI(title="AI Livestock Monitor API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load models once
_predict_breed    = None
_predict_disease  = None
_identify_muzzle  = None
_estimate_weight  = None

def _load():
    global _predict_breed, _predict_disease, _identify_muzzle, _estimate_weight
    if _predict_breed is None:
        from src.inference import predict_breed, predict_disease, identify_muzzle, estimate_weight
        _predict_breed   = predict_breed
        _predict_disease = predict_disease
        _identify_muzzle = identify_muzzle
        _estimate_weight = estimate_weight

def _img(file: UploadFile) -> Image.Image:
    return Image.open(io.BytesIO(file.file.read())).convert("RGB")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/samples")
def list_samples():
    return {
        category: [f for f in files if os.path.exists(os.path.join(SAMPLE_DIR, f))]
        for category, files in SAMPLES.items()
    }


@app.get("/samples/{filename}")
def get_sample(filename: str):
    path = os.path.join(SAMPLE_DIR, filename)
    if not os.path.exists(path) or not filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/predict/breed")
async def breed(file: UploadFile = File(...)):
    try:
        _load()
        result = _predict_breed(_img(file))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/disease")
async def disease(file: UploadFile = File(...)):
    try:
        _load()
        result = _predict_disease(_img(file))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/muzzle")
async def muzzle(file: UploadFile = File(...), top_k: int = 3):
    try:
        _load()
        result = _identify_muzzle(_img(file), top_k=top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/weight")
async def weight(
    height_cm: float = Form(...),
    volume_l:  float = Form(...),
    feed_idx:  int   = Form(...),
    sunlight:  float = Form(...),
):
    try:
        _load()
        result = _estimate_weight(height_cm, volume_l, feed_idx, sunlight)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/full")
async def full(
    file:      UploadFile = File(...),
    height_cm: float = Form(130.0),
    volume_l:  float = Form(400.0),
    feed_idx:  int   = Form(0),
    sunlight:  float = Form(8.0),
):
    try:
        _load()
        img    = _img(file)
        breed  = _predict_breed(img)
        disease = _predict_disease(img)
        muzzle  = _identify_muzzle(img, top_k=3)
        wt      = _estimate_weight(height_cm, volume_l, feed_idx, sunlight)
        return {
            "breed":   breed,
            "disease": disease,
            "muzzle":  muzzle,
            "weight":  wt,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
