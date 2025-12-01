import os
from pathlib import Path

# Resolve repository root (absolute) -> .../Traffic-Sign-Detection
REPO_ROOT = Path(__file__).resolve().parents[2]
# Resolve weight base dir: prefer src/weight, then src/weights
_WEIGHTS_CANDIDATES = [REPO_ROOT / 'src' / 'weight', REPO_ROOT / 'src' / 'weights']
WEIGHT_BASE_DIR = str(next((p for p in _WEIGHTS_CANDIDATES if p.exists()), _WEIGHTS_CANDIDATES[0]))

class Config:
    # Paths
    DATASET_PATH = "dataset"
    TRAIN_DIR = os.path.join(DATASET_PATH, "train")
    VAL_DIR = os.path.join(DATASET_PATH, "valid")
    TEST_DIR = os.path.join(DATASET_PATH, "test")

    # Model
    MODEL_SIZE = "yolov8m.pt"  # Using YOLOv8m as specified
    EPOCHS = 100
    BATCH_SIZE = 8
    IMAGE_SIZE = 640

    # Training
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 0.0005
    MOMENTUM = 0.937

    # Data augmentation
    AUGMENTATION = True
    HORIZONTAL_FLIP = 0.5
    VERTICAL_FLIP = 0.0
    ROTATION = 15
    BRIGHTNESS = 0.2
    CONTRAST = 0.2
    SATURATION = 0.2
    HUE = 0.1

    # Enhanced Inference Settings
    CONFIDENCE_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.45
    MAX_DETECTIONS = 100
    MULTI_LABEL = True
    VERBOSE = True

    # Image Enhancement Settings
    ENABLE_IMAGE_ENHANCEMENT = True
    DENOISE_STRENGTH = 0.5
    SHARPENING_STRENGTH = 1.2
    GAMMA_CORRECTION = 1.2
    CONTRAST_ENHANCEMENT = 1.3

    # IO Paths (under repo_root/data)
    DATA_DIR = os.path.join(REPO_ROOT, "data")
    OUTPUT_DIR = os.path.join(DATA_DIR, "output")
    INPUT_DIR = os.path.join(DATA_DIR, 'input')
    REAL_TIME_OUTPUT_DIR = os.path.join(DATA_DIR, 'real_time_output')
    # Trained weights live under src/weight/all_weight or src/weights/all_weight
    ALL_WEIGHT_DIR = os.path.join(WEIGHT_BASE_DIR, 'all_weight')
    PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

    # Model save (legacy placeholders; training/predict use ALL_WEIGHT_DIR)
    BEST_MODEL_PATH = os.path.join(ALL_WEIGHT_DIR, "best_traffic_sign_model.pt")
    LAST_MODEL_PATH = os.path.join(ALL_WEIGHT_DIR, "last_traffic_sign_model.pt")

    # Workers
    WORKERS = 2

    @classmethod
    def create_directories(cls):
        """Create necessary directories under data/"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.INPUT_DIR, exist_ok=True)
        os.makedirs(cls.REAL_TIME_OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.ALL_WEIGHT_DIR, exist_ok=True)
        os.makedirs(cls.PREDICTIONS_DIR, exist_ok=True)

    @classmethod
    def get_dataset_yaml(cls):
        """Generate dataset.yaml content with test set"""
        return f"""
# Dataset configuration
""" 