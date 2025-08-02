import os
from pathlib import Path

class Config:
    # Paths
    DATASET_PATH = "dataset"
    TRAIN_IMAGES = os.path.join(DATASET_PATH, "train", "images")
    TRAIN_LABELS = os.path.join(DATASET_PATH, "train", "labels")
    VAL_IMAGES = os.path.join(DATASET_PATH, "valid", "images")
    VAL_LABELS = os.path.join(DATASET_PATH, "valid", "labels")
    
    # Model - Tối ưu cho chất lượng cao
    MODEL_SIZE = "yolov8m.pt"  # YOLOv8m cân bằng giữa tốc độ và độ chính xác
    EPOCHS = 150  # Tăng từ 100 lên 150 để đạt chất lượng cao hơn
    BATCH_SIZE = 16  # Tăng từ 8 lên 16 (nếu GPU có đủ VRAM)
    IMAGE_SIZE = 640  # Giữ nguyên cho độ chính xác tốt
    
    # Training - Tối ưu learning rate và schedule
    LEARNING_RATE = 0.001  # Giảm từ 0.01 xuống 0.001 cho ổn định hơn
    WEIGHT_DECAY = 0.0005
    MOMENTUM = 0.937
    
    # Advanced Data Augmentation Settings - Tăng cường cho dataset nhỏ
    AUGMENTATION = True
    AUGMENTATION_STRENGTH = 0.9  # Tăng từ 0.8 lên 0.9 cho dataset nhỏ
    
    # Geometric Augmentation - Tối ưu cho biển báo giao thông
    HORIZONTAL_FLIP = 0.5
    VERTICAL_FLIP = 0.0  # Bỏ vertical flip vì biển báo không nên lật dọc
    ROTATION = 15  # Giảm từ 30 xuống 15 độ (biển báo cần giữ hướng)
    SHIFT_LIMIT = 0.15  # Tăng từ 0.1 lên 0.15
    SCALE_LIMIT = 0.3  # Tăng từ 0.2 lên 0.3
    PERSPECTIVE_SCALE = (0.02, 0.05)  # Giảm perspective (biển báo phẳng)
    
    # Color and Brightness Augmentation - Tăng cường cho điều kiện thực tế
    BRIGHTNESS_LIMIT = 0.4  # Tăng từ 0.3 lên 0.4
    CONTRAST_LIMIT = 0.4  # Tăng từ 0.3 lên 0.4
    SATURATION_LIMIT = 0.4  # Tăng từ 0.3 lên 0.4
    HUE_LIMIT = 15  # Giảm từ 20 xuống 15 (giữ màu biển báo)
    GAMMA_LIMIT = (70, 130)  # Tăng range từ (80,120) lên (70,130)
    
    # Noise and Blur Augmentation - Tăng cường cho điều kiện thực tế
    GAUSSIAN_NOISE_VAR = (5.0, 30.0)  # Giảm noise để không làm mất chi tiết
    ISO_NOISE_COLOR_SHIFT = (0.005, 0.03)  # Giảm noise
    MULTIPLICATIVE_NOISE = (0.95, 1.05)  # Giảm noise
    BLUR_LIMIT = 5  # Tăng từ 3 lên 5
    MOTION_BLUR_LIMIT = 5  # Tăng từ 3 lên 5
    
    # Weather and Lighting Effects - Tăng cường cho điều kiện thực tế
    RAIN_PROBABILITY = 0.3  # Tăng từ 0.2 lên 0.3
    FOG_PROBABILITY = 0.3  # Tăng từ 0.2 lên 0.3
    SUNFLARE_PROBABILITY = 0.3  # Tăng từ 0.2 lên 0.3
    
    # Occlusion and Cutout - Tăng cường để model robust hơn
    DROPOUT_PROBABILITY = 0.4  # Tăng từ 0.3 lên 0.4
    COARSE_DROPOUT_HOLES = (2, 10)  # Tăng từ (1,8) lên (2,10)
    COARSE_DROPOUT_SIZE = (10, 40)  # Tăng từ (8,32) lên (10,40)
    GRID_DROPOUT_RATIO = 0.15  # Tăng từ 0.1 lên 0.15
    
    # Elastic and Optical Distortions - Giảm để giữ hình dạng biển báo
    ELASTIC_TRANSFORM_PROBABILITY = 0.1  # Giảm từ 0.2 xuống 0.1
    OPTICAL_DISTORTION_PROBABILITY = 0.1  # Giảm từ 0.2 xuống 0.1
    
    # Mosaic and Mixup (YOLO specific) - Tối ưu cho biển báo
    MOSAIC_PROBABILITY = 0.6  # Tăng từ 0.5 lên 0.6
    MIXUP_PROBABILITY = 0.4  # Tăng từ 0.3 lên 0.4
    
    # Enhanced Inference Settings - Tối ưu cho độ chính xác
    CONFIDENCE_THRESHOLD = 0.25  # Giảm từ 0.3 xuống 0.25 để phát hiện nhiều hơn
    NMS_THRESHOLD = 0.4  # Giảm từ 0.45 xuống 0.4
    MAX_DETECTIONS = 50  # Giảm từ 100 xuống 50 (biển báo thường ít)
    MULTI_LABEL = True
    VERBOSE = True
    
    # Image Enhancement Settings - Tối ưu cho biển báo
    ENABLE_IMAGE_ENHANCEMENT = True
    DENOISE_STRENGTH = 0.6  # Tăng từ 0.5 lên 0.6
    SHARPENING_STRENGTH = 1.3  # Tăng từ 1.2 lên 1.3
    GAMMA_CORRECTION = 1.1  # Giảm từ 1.2 xuống 1.1
    CONTRAST_ENHANCEMENT = 1.4  # Tăng từ 1.3 lên 1.4
    
    # Output
    OUTPUT_DIR = "output"
    INPUT_DIR = 'input'
    PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
    
    # Model save
    BEST_MODEL_PATH = "best_traffic_sign_model.pt"
    LAST_MODEL_PATH = "last_traffic_sign_model.pt"
    
    # Training Schedule - Tối ưu cho 150 epochs
    WARMUP_EPOCHS = 3  # Warmup epochs
    COSINE_ANNEALING = True  # Sử dụng cosine annealing
    PATIENCE = 20  # Early stopping patience
    
    # Validation
    VAL_FREQ = 1  # Validate mỗi epoch
    SAVE_FREQ = 10  # Save checkpoint mỗi 10 epochs
    
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
nc: 12  # Updated to 12 classes

# Class names (updated for new dataset with 12 classes)
names:
  0: i.423.b
  1: p.102
  2: p.106.b
  3: p.130
  4: p.131.a
  5: r.308.b
  6: sus
  7: w.201.a
  8: w.203.c
  9: w.207.b
  10: w.207.c
  11: w.209
""" 