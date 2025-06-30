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
import unicodedata

# Hàm tạo thư mục predict mới
def get_new_predict_dir(base_dir="output"):
    i = 1
    while True:
        predict_dir = os.path.join(base_dir, f"predict{i}")
        if not os.path.exists(predict_dir):
            os.makedirs(predict_dir)
            return predict_dir
        i += 1

# Thêm ánh xạ class -> mô tả tiếng Việt đầy đủ
class_vi_map = {
    "W.224": "Đường hẹp cả hai phía",
    "W.205c": "Đường bị hẹp về phía bên trái",
    "P.102": "Cấm đi ngược chiều",
    "R.302a": "Hướng phải đi thẳng",
    "W.205a": "Đường bị hẹp về phía bên phải",
    "W.207": "Đường giao nhau",
    "W.201a": "Chỗ ngoặt nguy hiểm vòng bên trái",
    "P.123a": "Cấm rẽ trái",
    "I.434a": "Biển tên đường (kiểu 1)",
    "R.303": "Hướng phải rẽ",
    "P.130": "Cấm dừng xe và đỗ xe",
    "I.409": "Chỉ dẫn khu công nghiệp",
    "R.415a": "Khu vực cấm đỗ xe",
    "W.245a": "Đường có ổ gà, lồi lõm",
    "P.106a*Xe tải": "Cấm xe tải",
    "W.203c": "Đường người đi bộ cắt ngang",
    "P.117*": "Cấm xe đạp",
    "P.124a*": "Cấm xe máy",
    "P.107": "Cấm quay đầu xe",
    "P.124d": "Cấm xe khách và xe tải",
    "P.103a": "Cấm ô tô",
    "W.203b": "Đường người đi bộ cắt ngang",
    "W.221b": "Đường có vật cản",
    "P.111": "Cấm vượt",
    "P.129": "Cấm bóp còi",
    "S.505a*Xe_may": "Biển phụ áp dụng cho xe máy",
    "W.246a": "Nguy hiểm khác",
    "W.225": "Đường trơn",
    "S.505a*Xe_tai_va_cong": "Biển phụ áp dụng cho xe tải và container",
    "P.104": "Cấm xe kéo",
    "S.505a*Xe_tai": "Biển phụ áp dụng cho xe tải",
    "Camera": "Camera giao thông",
    "P.123b": "Cấm rẽ phải",
    "W.202b": "Chỗ ngoặt nguy hiểm vòng bên phải",
    "B.8a": "Biển báo chỉ hướng",
    "P.137": "Hạn chế chiều cao",
    "P.139": "Hạn chế chiều rộng",
    "W.205b": "Đường bị hẹp cả hai phía",
    "P.127*50": "Giới hạn tốc độ tối đa 50 km/h",
    "P.127*60": "Giới hạn tốc độ tối đa 60 km/h",
    "P.127*80": "Giới hạn tốc độ tối đa 80 km/h",
    "P.127*40": "Giới hạn tốc độ tối đa 40 km/h",
    "R.301e": "Hướng đi ưu tiên",
    "W.239b*": "Nguy hiểm do súc vật",
    "W.233": "Gió ngang",
    "I.407a": "Chỉ dẫn giao lộ",
    "P.131a": "Cấm đỗ xe",
    "P.124b1": "Cấm xe tải",
    "W.210": "Giao nhau với đường sắt có rào chắn",
    "P.124c": "Cấm xe mô tô ba bánh",
    "W.201b": "Chỗ ngoặt nguy hiểm vòng bên phải",
    "W.246c": "Chú ý chướng ngại vật",
    "DP.135": "Hết hạn chế tốc độ",
    "P.103b": "Cấm ô tô khách",
    "P.103c": "Cấm ô tô tải",
    "P.106a": "Cấm xe tải",
    "P.106b": "Cấm xe tải trên 2,5 tấn",
    "P.107a": "Cấm quay đầu xe (trừ xe máy và xe đạp)",
    "P.112": "Cấm xe kéo moóc",
    "P.115": "Cấm xe người kéo",
    "P.117": "Cấm xe đạp",
    "P.124a": "Cấm xe máy",
    "P.124b": "Cấm xe mô tô",
    "P.125": "Cấm xe công nông",
    "P.127": "Giới hạn tốc độ tối đa",
    "P.128": "Cấm sử dụng đèn chiếu xa",
    "P.245a": "Cấm dừng xe",
    "R.301a": "Đường ưu tiên",
    "R.301c": "Hướng đi ưu tiên",
    "R.301d": "Hết đường ưu tiên",
    "R.302b": "Hướng phải rẽ trái",
    "R.407a": "Đi chậm",
    "R.409": "Hết hạn chế tốc độ tối thiểu",
    "R.425": "Hướng đi cho xe thô sơ",
    "R.434": "Khu vực cấm dừng xe",
    "S.509a": "Biển phụ chỉ dẫn khoảng cách",
    "W.202a": "Chỗ ngoặt nguy hiểm vòng bên trái",
    "W.205d": "Đường bị hẹp về phía bên phải",
    "W.207a": "Giao nhau với đường không ưu tiên",
    "W.207b": "Giao nhau với đường ưu tiên",
    "W.207c": "Giao nhau với đường cùng cấp",
    "W.208": "Giao nhau với đường sắt không có rào chắn",
    "W.209": "Cầu hẹp",
    "W.219": "Đường đôi",
    "W.227": "Đường cấm",
    "W.235": "Máy bay bay thấp"
}

# Đọc danh sách class đúng thứ tự từ data.yaml
with open('data.yaml', 'r', encoding='utf-8') as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml['names']

# descriptions_vi mới: đúng thứ tự class_names, ưu tiên giữ mô tả cũ nếu có, bổ sung nghĩa mới nếu chưa có, nếu vẫn chưa có thì để 'Chưa có mô tả'
old_descriptions = [
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
descriptions_vi = []
for i, class_name in enumerate(class_names):
    if i < len(old_descriptions):
        descriptions_vi.append(old_descriptions[i])
    elif class_name in class_vi_map:
        descriptions_vi.append(class_vi_map[class_name])
    else:
        descriptions_vi.append("Chưa có mô tả")

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
    
    def predict_image(self, image_path: str, save_result: bool = True, image_id: int = None) -> Tuple[np.ndarray, List[dict]]:
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
                class_label = self.class_names[class_idx] if isinstance(self.class_names, list) and class_idx < len(self.class_names) else str(class_idx)
                class_label_vi = descriptions_vi_no_diacritics[class_idx] if class_idx < len(descriptions_vi_no_diacritics) else class_label
                detection = {
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf),
                    'class_id': class_idx,
                    'class_label': class_label,
                    'class_label_vi': class_label_vi
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
            if image_id is not None:
                output_filename = f"{image_id}-{output_filename}"
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
        for idx, image_path in enumerate(image_files, 1):
            try:
                print(f"Processing {image_path}...")
                self.predict_image(image_path, save_result=True, image_id=idx)
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