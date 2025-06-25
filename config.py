import os
from pathlib import Path

class Config:
    # Paths
    DATASET_PATH = "datasetv2"
    TRAIN_IMAGES = os.path.join(DATASET_PATH, "images", "train")
    TRAIN_LABELS = os.path.join(DATASET_PATH, "labels", "train")
    VAL_IMAGES = os.path.join(DATASET_PATH, "images", "val")
    VAL_LABELS = os.path.join(DATASET_PATH, "labels", "val")
    
    # Model
    MODEL_SIZE = "yolov8m.pt"  # Using YOLOv8m as specified
    EPOCHS = 25
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
    ROTATION = 15  # Increased rotation angle
    BRIGHTNESS = 0.2  # Increased brightness variation
    CONTRAST = 0.2  # Increased contrast variation
    SATURATION = 0.2  # Increased saturation variation
    HUE = 0.1
    
    # Enhanced Inference Settings
    CONFIDENCE_THRESHOLD = 0.6
    NMS_THRESHOLD = 0.45
    MAX_DETECTIONS = 100
    MULTI_LABEL = True  # Allow multiple labels per box
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
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.INPUT_DIR, exist_ok=True)
        os.makedirs(cls.PREDICTIONS_DIR, exist_ok=True)
        
    @classmethod
    def get_dataset_yaml(cls):
        """Generate dataset.yaml content"""
        return f"""
# Dataset configuration
path: {os.path.abspath(cls.DATASET_PATH)}
train: images/train
val: images/val

# Number of classes
nc: 43  # Adjust based on your traffic sign classes

# Class names (example - adjust based on your dataset)
names:
  0: speed_limit_20
  1: speed_limit_30
  2: speed_limit_50
  3: speed_limit_60
  4: speed_limit_70
  5: speed_limit_80
  6: end_of_speed_limit_80
  7: speed_limit_100
  8: speed_limit_120
  9: no_passing
  10: no_passing_for_vehicles_over_3_5_metric_tons
  11: right_of_way_at_the_next_intersection
  12: priority_road
  13: yield
  14: stop
  15: no_vehicles
  16: vehicles_over_3_5_metric_tons_prohibited
  17: no_entry
  18: general_caution
  19: dangerous_curve_left
  20: dangerous_curve_right
  21: double_curve
  22: bumpy_road
  23: slippery_road
  24: road_narrows_on_the_right
  25: road_work
  26: traffic_signals
  27: pedestrians
  28: children_crossing
  29: bicycles_crossing
  30: snow
  31: wild_animals_crossing
  32: end_of_all_speed_and_passing_limits
  33: turn_right_ahead
  34: turn_left_ahead
  35: ahead_only
  36: go_straight_or_right
  37: go_straight_or_left
  38: keep_right
  39: keep_left
  40: roundabout_mandatory
  41: end_of_no_passing
  42: end_of_no_passing_by_vehicles_over_3_5_metric_tons
""" 