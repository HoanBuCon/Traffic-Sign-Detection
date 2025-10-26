import os
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
from typing import List, Union, Tuple
import numpy as np
try:
    from src.training.config import Config
    from src.core.utils import ImageEnhancer, VisualizationUtils, FileUtils
except ModuleNotFoundError:
    import sys as _sys, pathlib as _pathlib
    _repo_root = _pathlib.Path(__file__).resolve().parents[2]
    _sys.path.insert(0, str(_repo_root))
    from src.training.config import Config
    from src.core.utils import ImageEnhancer, VisualizationUtils, FileUtils
import glob
import yaml
import unicodedata

# Hàm tạo thư mục predict mới
def get_new_predict_dir(base_dir=None):
    if base_dir is None:
        base_dir = Config.OUTPUT_DIR
    i = 1
    while True:
        predict_dir = os.path.join(base_dir, f"predict{i}")
        if not os.path.exists(predict_dir):
            os.makedirs(predict_dir)
            return predict_dir
        i += 1

# Khởi tạo biến toàn cục để lưu descriptions từ data.yaml
descriptions_vi = []

def load_descriptions_from_yaml(yaml_path='data.yaml'):
    """
    Load descriptions từ file data.yaml
    
    Args:
        yaml_path: Đường dẫn đến file data.yaml
        
    Returns:
        List descriptions tiếng Việt
    """
    global descriptions_vi
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
            descriptions_vi = data_yaml.get('descriptions', [])
            if not descriptions_vi:
                print(f"[WARNING] No descriptions found in {yaml_path}, using empty list")
                descriptions_vi = []
            else:
                print(f"[INFO] Loaded {len(descriptions_vi)} descriptions from {yaml_path}")
        return descriptions_vi
    except Exception as e:
        print(f"[ERROR] Failed to load descriptions from {yaml_path}: {e}")
        descriptions_vi = []
        return descriptions_vi

def remove_vietnamese_diacritics(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
    text = text.replace('đ', 'd').replace('Đ', 'D')
    text = text.replace('–', '-')
    text = text.replace(' ', '_')
    text = text.replace('*', '')
    text = text.replace('(', '').replace(')', '')
    text = text.replace('/', '_')
    text = text.replace(',', '')
    text = text.replace('.', '')
    text = text.replace('-', '_')
    text = text.replace('__', '_')
    return text

def get_descriptions_no_diacritics():
    """
    Tạo descriptions không dấu từ descriptions_vi
    
    Returns:
        List descriptions không dấu
    """
    global descriptions_vi
    if not descriptions_vi:
        return []
    return [remove_vietnamese_diacritics(desc) for desc in descriptions_vi]

# Tạo dict ánh xạ mã nhãn -> mô tả tiếng Việt
def get_class_names_vi(class_names):
    if isinstance(class_names, dict):
        return {class_names[str(i)]: descriptions_vi[i] for i in range(len(descriptions_vi)) if str(i) in class_names}
    elif isinstance(class_names, list):
        return {class_names[i]: descriptions_vi[i] for i in range(min(len(class_names), len(descriptions_vi)))}
    return {}

class TrafficSignDetector:
    def __init__(self, model_path: str = None, predictions_dir: str = None):
        """
        Initialize the traffic sign detector
        
        Args:
            model_path: Path to the trained model (if None, auto-select latest best.pt in all_weight)
            predictions_dir: Directory to save predictions (if None, auto-create new folder)
        """
        if model_path is None:
            # Tìm thư mục trainX mới nhất trong data/all_weight
            all_weight_dir = Config.ALL_WEIGHT_DIR
            os.makedirs(all_weight_dir, exist_ok=True)
            train_dirs = [d for d in os.listdir(all_weight_dir) if d.startswith('train') and os.path.isdir(os.path.join(all_weight_dir, d))]
            if not train_dirs:
                # Fallback to local pretrained only; do NOT download
                import pathlib as _pl
                repo_root = _pl.Path(__file__).resolve().parents[2]
                # Try src/weights/yolov8m.pt then weights/yolov8m.pt
                candidates = [repo_root / 'src' / 'weight' / 'yolov8m.pt',
                              repo_root / 'src' / 'weights' / 'yolov8m.pt',
                              repo_root / 'weights' / 'yolov8m.pt']
                for c in candidates:
                    if c.exists():
                        model_path = str(c)
                        print(f"[INFO] No trained model found; using local pretrained: {model_path}")
                        break
                else:
                    raise FileNotFoundError(
                        "No trained weights under src/weights/all_weight and no local pretrained found. "
                        "Please place a trained model at src/weights/all_weight/trainX/best.pt or a pretrained at src/weights/yolov8m.pt.")
            else:
                # Sắp xếp theo số thứ tự tăng dần
                train_dirs_sorted = sorted(train_dirs, key=lambda x: int(x.replace('train', '')) if x.replace('train', '').isdigit() else 0)
                latest_train_dir = os.path.join(all_weight_dir, train_dirs_sorted[-1])
                best_pt_path = os.path.join(latest_train_dir, 'best.pt')
                if not os.path.exists(best_pt_path):
                    print(f"[WARN] No best.pt found in {latest_train_dir}, using last fallback if available")
                    last_pt_path = os.path.join(latest_train_dir, 'last.pt')
                    if os.path.exists(last_pt_path):
                        best_pt_path = last_pt_path
                model_path = best_pt_path
                print(f"[INFO] Using weight: {model_path}")
        self.model = YOLO(model_path)
        self.image_enhancer = ImageEnhancer()
        self.config = Config
        if predictions_dir is None:
            self.predictions_dir = get_new_predict_dir(self.config.OUTPUT_DIR)
        else:
            self.predictions_dir = predictions_dir
        print(f"[INFO] Saving predictions to: {self.predictions_dir}")
        
        # Đọc class_names từ data.yaml (auto-detect path)
        from src.core.utils import find_data_yaml
        data_yaml_path = find_data_yaml()
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
            self.class_names = data_yaml.get('names', {})
            
        # Load descriptions từ data.yaml
        load_descriptions_from_yaml(data_yaml_path)
        self.descriptions_vi_no_diacritics = get_descriptions_no_diacritics()
        
    def enhance_image_for_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Apply multiple enhancement techniques for better inference
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        if not self.config.ENABLE_IMAGE_ENHANCEMENT:
            return image
            
        # Apply denoising
        image = self.image_enhancer.denoise_image(image)
        
        # Apply sharpening
        image = self.image_enhancer.sharpen_image(image)
        
        # Enhance contrast and brightness
        image = self.image_enhancer.enhance_image(
            image, 
            enhancement_level=self.config.CONTRAST_ENHANCEMENT
        )
        
        # Apply gamma correction
        image = self.image_enhancer.adjust_gamma(
            image, 
            gamma=self.config.GAMMA_CORRECTION
        )
        
        return image
    
    def predict_image(self, image_path: str, save_result: bool = True) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect traffic signs in an image
        
        Args:
            image_path: Path to the input image
            save_result: Whether to save the visualization
            
        Returns:
            Tuple of (annotated image, detections)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Enhance image for better inference
        enhanced_image = self.enhance_image_for_inference(image)
        
        # Run inference
        results = self.model.predict(
            enhanced_image,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.NMS_THRESHOLD,
            max_det=self.config.MAX_DETECTIONS,
            verbose=self.config.VERBOSE
        )
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_idx = int(box.cls)
                if isinstance(self.class_names, dict):
                    class_label = self.class_names.get(class_idx, self.class_names.get(str(class_idx), str(class_idx)))
                elif isinstance(self.class_names, list) and class_idx < len(self.class_names):
                    class_label = self.class_names[class_idx]
                else:
                    class_label = str(class_idx)
                
                # Sử dụng description có dấu cho hiển thị và không dấu cho lưu file
                global descriptions_vi
                class_label_vi_display = descriptions_vi[class_idx] if class_idx < len(descriptions_vi) else class_label
                class_label_vi_filename = self.descriptions_vi_no_diacritics[class_idx] if class_idx < len(self.descriptions_vi_no_diacritics) else class_label
                
                detection = {
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf),
                    'class_id': class_idx,
                    'class_label': class_label,
                    'class_label_vi': class_label_vi_display,  # Dùng để hiển thị
                    'class_label_vi_filename': class_label_vi_filename  # Dùng cho tên file
                }
                detections.append(detection)
        
        # Draw detections
        annotated_image = VisualizationUtils.draw_detections(
            image,
            detections,
            class_names=self.class_names,
            confidence_threshold=self.config.CONFIDENCE_THRESHOLD
        )
        
        # Save result if requested
        if save_result:
            output_filename = os.path.basename(image_path)
            VisualizationUtils.save_detection_result(
                annotated_image,
                self.predictions_dir,
                output_filename,
                detections
            )
        
        return annotated_image, detections
    
    def predict_directory(self, input_dir: str = Config.INPUT_DIR) -> None:
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing input images
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.predictions_dir, exist_ok=True)
        
        # Get all image files
        image_files = FileUtils.get_image_files(input_dir)
        
        # Process each image
        for image_path in image_files:
            try:
                print(f"Processing {image_path}...")
                self.predict_image(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")

def main():
    """Main function to run inference"""
    # Create necessary directories
    Config.create_directories()
    
    # Initialize detector
    detector = TrafficSignDetector()
    
    # Process all images in custom_images directory
    detector.predict_directory()

if __name__ == "__main__":
    main() 