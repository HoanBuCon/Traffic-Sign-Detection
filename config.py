import os
from pathlib import Path
class Config:
    # Paths
    DATASET_PATH = "dataset"
    TRAIN_DIR = os.path.join(DATASET_PATH, "train")
    VAL_DIR = os.path.join(DATASET_PATH, "valid")
    TEST_DIR = os.path.join(DATASET_PATH, "test")

    # Model
    MODEL_SIZE = "yolov8m.pt"  # Using YOLOv8m as specified
    EPOCHS = 200
    BATCH_SIZE = 16
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

    # Output
    OUTPUT_DIR = "output"
    INPUT_DIR = 'input'
    PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

    # Model save
    BEST_MODEL_PATH = "best_traffic_sign_model.pt"
    LAST_MODEL_PATH = "last_traffic_sign_model.pt"

    # Workers
    WORKERS = 4

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.INPUT_DIR, exist_ok=True)
        os.makedirs(cls.PREDICTIONS_DIR, exist_ok=True)

    @classmethod
    def get_dataset_yaml(cls):
        """Generate dataset.yaml content with test set"""
        return f"""
# Dataset configuration
""" 