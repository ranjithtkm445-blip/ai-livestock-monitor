# config.py — Central configuration for the AI Livestock Biometric & Health Monitoring Platform

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
EMBEDDINGS_DIR = os.path.join(OUTPUTS_DIR, "embeddings")

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Image Quality Assessment ──────────────────────────────────────────────────
IQA_BLUR_THRESHOLD = 100.0       # Laplacian variance below this = blurry
IQA_MIN_BRIGHTNESS = 40          # 0-255
IQA_MAX_BRIGHTNESS = 220         # 0-255
IQA_MIN_RESOLUTION = (224, 224)  # Minimum (width, height)

# ── Cattle Detection (YOLOv11) ────────────────────────────────────────────────
DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "cattle_detector.pt")
DETECTION_CONF_THRESHOLD = 0.5
DETECTION_IOU_THRESHOLD = 0.45
DETECTION_IMG_SIZE = 640

# ── Muzzle Biometric Identification ──────────────────────────────────────────
MUZZLE_MODEL_PATH = os.path.join(MODELS_DIR, "arcface_muzzle.pt")
MUZZLE_DETECTOR_PATH = os.path.join(MODELS_DIR, "muzzle_detector.pt")
EMBEDDING_DIM = 512
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "muzzle_index.faiss")
FAISS_METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "muzzle_metadata.json")
SIMILARITY_THRESHOLD = 0.75      # Cosine similarity threshold for positive match
ARCFACE_BACKBONE = "resnet50"
ARCFACE_PRETRAINED = True

# ── Breed Classification (ConvNeXt Tiny) ─────────────────────────────────────
BREED_MODEL_PATH = os.path.join(MODELS_DIR, "breed_classifier.pt")
BREED_CLASSES = [
    "Angus", "Brahman", "Gir", "Holstein_Friesian",
    "Jersey", "Red_Sindhi", "Sahiwal", "Tharparkar"
]
NUM_BREEDS = len(BREED_CLASSES)
BREED_IMG_SIZE = 224
BREED_BACKBONE = "convnext_tiny"

# ── Body Condition Scoring (EfficientNet-B3) ──────────────────────────────────
BCS_MODEL_PATH = os.path.join(MODELS_DIR, "bcs_classifier.pt")
BCS_CLASSES = ["BCS_1", "BCS_2", "BCS_3", "BCS_4", "BCS_5"]
BCS_LABELS = {
    "BCS_1": "Very Thin",
    "BCS_2": "Thin",
    "BCS_3": "Ideal",
    "BCS_4": "Fat",
    "BCS_5": "Obese",
}
NUM_BCS_CLASSES = len(BCS_CLASSES)
BCS_IMG_SIZE = 300
BCS_BACKBONE = "efficientnet_b3"

# ── Weight Estimation (ViT Regression) ───────────────────────────────────────
WEIGHT_MODEL_PATH = os.path.join(MODELS_DIR, "weight_estimator.pt")
WEIGHT_IMG_SIZE = 224
WEIGHT_MIN_KG = 50.0
WEIGHT_MAX_KG = 1200.0
WEIGHT_BACKBONE = "vit_base_patch16_224"

# ── Disease Detection (YOLOv11) ───────────────────────────────────────────────
DISEASE_MODEL_PATH = os.path.join(MODELS_DIR, "disease_detector.pt")
DISEASE_CLASSES = [
    "Mastitis", "Lumpy_Skin_Disease", "Foot_Rot",
    "Eye_Infection", "Skin_Wound"
]
DISEASE_CONF_THRESHOLD = 0.4
DISEASE_IMG_SIZE = 640

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
NUM_WORKERS = 0        # Windows/Docker safe
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ── Database (PostgreSQL) ─────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "livestock_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_IMAGE_SIZE_MB = 10

# ── Report ────────────────────────────────────────────────────────────────────
REPORT_LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")
FARM_NAME = "SmartFarm AI"
