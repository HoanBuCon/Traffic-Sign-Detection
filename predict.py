import os
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
from typing import List, Union, Tuple
import numpy as np
from config import Config
from utils import ImageEnhancer, VisualizationUtils, FileUtils
import glob
import yaml
<<<<<<< HEAD
=======
import unicodedata
>>>>>>> d9a6e9e11355e0f42059eaa302533f0e4301fe25

# Hàm tạo thư mục predict mới
def get_new_predict_dir(base_dir="output"):
    i = 1
    while True:
        predict_dir = os.path.join(base_dir, f"predict{i}")
        if not os.path.exists(predict_dir):
            os.makedirs(predict_dir)
            return predict_dir
        i += 1

<<<<<<< HEAD
=======
# Thêm ánh xạ mã nhãn sang mô tả tiếng Việt
descriptions_vi = [
    "Đường người đi bộ cắt ngang",
    "Đường giao nhau (ngã ba bên phải)",
    "Cấm đi ngược chiều",
    "Phải đi vòng sang bên phải",
    "Giao nhau với đường đồng cấp",
    "Giao nhau với đường không ưu tiên",
    "Chỗ ngoặt nguy hiểm vòng bên trái",
    "Cấm rẽ trái",
    "Bến xe buýt",
    "Nơi giao nhau chạy theo vòng xuyến",
    "Cấm dừng và đỗ xe",
    "Chỗ quay xe",
    "Biển gộp làn đường theo phương tiện",
    "Đi chậm",
    "Cấm xe tải",
    "Đường bị thu hẹp về phía phải",
    "Giới hạn chiều cao",
    "Cấm quay đầu",
    "Cấm ô tô khách và ô tô tải",
    "Cấm rẽ phải và quay đầu",
    "Cấm ô tô",
    "Đường bị thu hẹp về phía trái",
    "Gồ giảm tốc phía trước",
    "Cấm xe hai và ba bánh",
    "Kiểm tra",
    "Chỉ dành cho xe máy*",
    "Chướng ngoại vật phía trước",
    "Trẻ em",
    "Xe tải và xe công*",
    "Cấm mô tô và xe máy",
    "Chỉ dành cho xe tải*",
    "Đường có camera giám sát",
    "Cấm rẽ phải",
    "Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải",
    "Cấm xe sơ-mi rơ-moóc",
    "Cấm rẽ trái và phải",
    "Cấm đi thẳng và rẽ phải",
    "Đường giao nhau (ngã ba bên trái)",
    "Giới hạn tốc độ (50km/h)",
    "Giới hạn tốc độ (60km/h)",
    "Giới hạn tốc độ (80km/h)",
    "Giới hạn tốc độ (40km/h)",
    "Các xe chỉ được rẽ trái",
    "Chiều cao tĩnh không thực tế",
    "Nguy hiểm khác",
    "Đường một chiều",
    "Cấm đỗ xe",
    "Cấm ô tô quay đầu xe (được rẽ trái)",
    "Giao nhau với đường sắt có rào chắn",
    "Cấm rẽ trái và quay đầu xe",
    "Chỗ ngoặt nguy hiểm vòng bên phải",
    "Chú ý chướng ngại vật – vòng tránh sang bên phải"
]

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

descriptions_vi_no_diacritics = [remove_vietnamese_diacritics(desc) for desc in descriptions_vi]

# Tạo dict ánh xạ mã nhãn -> mô tả tiếng Việt
def get_class_names_vi(class_names):
    if isinstance(class_names, dict):
        return {class_names[str(i)]: descriptions_vi[i] for i in range(len(descriptions_vi)) if str(i) in class_names}
    elif isinstance(class_names, list):
        return {class_names[i]: descriptions_vi[i] for i in range(min(len(class_names), len(descriptions_vi)))}
    return {}

>>>>>>> d9a6e9e11355e0f42059eaa302533f0e4301fe25
class TrafficSignDetector:
    def __init__(self, model_path: str = None, predictions_dir: str = None):
        """
        Initialize the traffic sign detector
        
        Args:
            model_path: Path to the trained model (if None, auto-select latest best.pt in all_weight)
            predictions_dir: Directory to save predictions (if None, auto-create new folder)
        """
        if model_path is None:
            # Tìm thư mục trainX mới nhất trong all_weight
            all_weight_dir = 'all_weight'
            train_dirs = [d for d in os.listdir(all_weight_dir) if d.startswith('train') and os.path.isdir(os.path.join(all_weight_dir, d))]
            if not train_dirs:
                raise FileNotFoundError("No trained model found in all_weight! Please train the model first.")
            # Sắp xếp theo số thứ tự tăng dần
            train_dirs_sorted = sorted(train_dirs, key=lambda x: int(x.replace('train', '')) if x.replace('train', '').isdigit() else 0)
            latest_train_dir = os.path.join(all_weight_dir, train_dirs_sorted[-1])
            best_pt_path = os.path.join(latest_train_dir, 'best.pt')
            if not os.path.exists(best_pt_path):
                raise FileNotFoundError(f"No best.pt found in {latest_train_dir}!")
            model_path = best_pt_path
            print(f"[INFO] Using latest best.pt: {model_path}")
        self.model = YOLO(model_path)
        self.image_enhancer = ImageEnhancer()
        self.config = Config
        if predictions_dir is None:
            self.predictions_dir = get_new_predict_dir(self.config.OUTPUT_DIR)
        else:
            self.predictions_dir = predictions_dir
        print(f"[INFO] Saving predictions to: {self.predictions_dir}")
        
        # Đọc class_names từ data.yaml
        with open('data.yaml', 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
            self.class_names = data_yaml.get('names', {})
        
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
<<<<<<< HEAD
                # Lấy key mã nhãn từ dict theo chỉ số
                if isinstance(self.class_names, dict):
                    keys = list(self.class_names.keys())
                    if class_idx < len(keys):
                        class_id = keys[class_idx]
                    else:
                        class_id = str(class_idx)
                else:
                    class_id = str(class_idx)
                detection = {
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf),
                    'class_id': class_id
=======
                class_label = self.class_names[class_idx] if isinstance(self.class_names, list) and class_idx < len(self.class_names) else str(class_idx)
                class_label_vi = descriptions_vi_no_diacritics[class_idx] if class_idx < len(descriptions_vi_no_diacritics) else class_label
                detection = {
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf),
                    'class_id': class_idx,
                    'class_label': class_label,
                    'class_label_vi': class_label_vi
>>>>>>> d9a6e9e11355e0f42059eaa302533f0e4301fe25
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
<<<<<<< HEAD
            output_path = os.path.join(self.predictions_dir, output_filename)
=======
>>>>>>> d9a6e9e11355e0f42059eaa302533f0e4301fe25
            VisualizationUtils.save_detection_result(
                annotated_image,
                self.predictions_dir,
                output_filename,
<<<<<<< HEAD
                detections,
                class_names=self.class_names
=======
                detections
>>>>>>> d9a6e9e11355e0f42059eaa302533f0e4301fe25
            )
        
        return annotated_image, detections
    
<<<<<<< HEAD
    def predict_directory(self, input_dir: str = Config.CUSTOM_IMAGES_DIR) -> None:
=======
    def predict_directory(self, input_dir: str = Config.INPUT_DIR) -> None:
>>>>>>> d9a6e9e11355e0f42059eaa302533f0e4301fe25
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