---
title: AI Livestock Biometric Health Monitor
emoji: 🐄
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: gradio_app.py
pinned: false
---

# AI Livestock Biometric & Health Monitoring Platform

A multi-task computer vision system for cattle identification, health assessment, and farm management.

## Pipeline

```
Image Upload
    │
    ▼
Image Quality Assessment (OpenCV + Laplacian Variance)
    │
    ▼
YOLOv11 Cattle Detection
    │
    ├── Muzzle ROI ──► ArcFace + FAISS ──► Animal ID
    │
    └── Full Body ROI ──► Breed (ConvNeXt Tiny)
                      ──► BCS  (EfficientNet-B3)
                      ──► Weight (ViT Regression)
                      ──► Disease (YOLOv11)
    │
    ▼
Animal Profile (PostgreSQL)
    │
    ▼
PDF Health Report
    │
    ▼
FastAPI Backend + Streamlit Dashboard
```

## Modules

| Module | Model | Task |
|--------|-------|------|
| Quality Assessment | OpenCV | Blur / brightness / resolution |
| Cattle Detection | YOLOv11 | Detect + extract ROIs |
| Muzzle ID | ArcFace + FAISS | Biometric identification |
| Breed Classification | ConvNeXt Tiny | 8-class breed |
| Body Condition Scoring | EfficientNet-B3 | BCS 1–5 |
| Weight Estimation | ViT Regression | kg prediction |
| Disease Detection | YOLOv11 | 5 disease classes |
| Database | PostgreSQL | Animal profiles + history |
| Report | ReportLab | PDF export |
| Dashboard | Streamlit | Farmer UI |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate
pip install -r requirements.txt
```

## Run with Docker

```bash
docker-compose up -d
```

- API:       http://localhost:8000
- Dashboard: http://localhost:8501
- API Docs:  http://localhost:8000/docs

## Run locally (no Docker)

```bash
# Terminal 1 — API
uvicorn app:app --reload

# Terminal 2 — Dashboard
streamlit run dashboard.py
```

> PostgreSQL must be running and configured in `.env`.

## Datasets

| Module | Dataset |
|--------|---------|
| Muzzle ID | Open Cattle Muzzle Dataset (OCMD) |
| Breed | Cow Breed Dataset / Indian Cattle Breeds (Kaggle) |
| BCS | DairyNZ Body Condition Score Dataset |
| Weight | RGB-D Cow Weight Estimation Dataset |
| Disease | Lumpy Skin Disease + Cow Skin Disease (Kaggle) |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze` | Full pipeline inference |
| POST | `/animals` | Register animal |
| GET | `/animals` | List animals |
| GET | `/animals/{id}` | Animal profile |
| POST | `/animals/{id}/register-muzzle` | Register biometric |
| GET | `/animals/{id}/health` | Health history |
| GET | `/animals/{id}/weight-history` | Weight history |
| POST | `/animals/{id}/vaccinations` | Add vaccination |
| GET | `/vaccinations/upcoming` | Upcoming reminders |
| GET | `/animals/{id}/report` | Download PDF report |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | YOLOv11 (Ultralytics) |
| Identification | ArcFace + FAISS |
| Classification | ConvNeXt Tiny, EfficientNet-B3 |
| Regression | Vision Transformer (ViT) |
| Disease | YOLOv11 |
| Explainability | GradCAM |
| Backend | FastAPI |
| Database | PostgreSQL |
| Dashboard | Streamlit |
| Deployment | Docker Compose |
