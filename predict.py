import os
import cv2
import time
from ultralytics import YOLO
import numpy as np
import yaml
from config import Config
from utils import ImageEnhancer, VisualizationUtils
import unicodedata

# 12 class biển báo giao thông với mô tả tiếng Việt
class_vi_map = {
    "i.423.b": "Biển chỉ dẫn khoảng cách",
    "p.102": "Cấm đi ngược chiều", 
    "p.106.b": "Cấm xe tải trên 2,5 tấn",
    "p.130": "Cấm dừng xe và đỗ xe",
    "p.131.a": "Cấm đỗ xe",
    "r.308.b": "Hướng đi ưu tiên",
    "sus": "Biển báo nghi ngờ",
    "w.201.a": "Chỗ ngoặt nguy hiểm vòng bên trái",
    "w.203.c": "Đường người đi bộ cắt ngang",
    "w.207.b": "Giao nhau với đường ưu tiên",
    "w.207.c": "Giao nhau với đường cùng cấp",
    "w.209": "Cầu hẹp"
}

# Đọc danh sách class từ data.yaml
with open('data.yaml', 'r', encoding='utf-8') as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml['names']

# Tạo descriptions_vi theo thứ tự class_names
descriptions_vi = []
for class_name in class_names:
    if class_name in class_vi_map:
        descriptions_vi.append(class_vi_map[class_name])
    else:
        descriptions_vi.append(f"Biển báo {class_name}")

# Tạo descriptions không dấu để hiển thị
descriptions_vi_no_diacritics = []
for desc in descriptions_vi:
    text = unicodedata.normalize('NFD', desc)
    text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
    text = text.replace('đ', 'd').replace('Đ', 'D')
    descriptions_vi_no_diacritics.append(text)

def remove_vietnamese_diacritics(text):
    """Loại bỏ dấu tiếng Việt"""
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return text

class TrafficSignPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            # Tìm model mới nhất trong training_history
            training_history_dir = 'training_history'
            if not os.path.exists(training_history_dir):
                raise FileNotFoundError("Không tìm thấy thư mục training_history! Hãy train model trước.")
            
            train_dirs = [d for d in os.listdir(training_history_dir) if d.startswith('train') and os.path.isdir(os.path.join(training_history_dir, d))]
            if not train_dirs:
                raise FileNotFoundError("Không tìm thấy model đã train trong training_history! Hãy train model trước.")
            
            train_dirs_sorted = sorted(train_dirs, key=lambda x: int(x.replace('train', '')) if x.replace('train', '').isdigit() else 0)
            latest_train_dir = os.path.join(training_history_dir, train_dirs_sorted[-1])
            weights_dir = os.path.join(latest_train_dir, 'weights')
            best_pt_path = os.path.join(weights_dir, 'best.pt')
            
            if not os.path.exists(best_pt_path):
                raise FileNotFoundError(f"Không tìm thấy best.pt trong {weights_dir}!")
            
            model_path = best_pt_path
            print(f"[INFO] Sử dụng model: {model_path}")
        
        self.model = YOLO(model_path)
        self.image_enhancer = ImageEnhancer()
        self.config = Config()
        self.class_names = class_names
        self.descriptions_vi = descriptions_vi
        self.descriptions_vi_no_diacritics = descriptions_vi_no_diacritics

    def enhance_image_for_inference(self, image: np.ndarray) -> np.ndarray:
        """Cải thiện ảnh trước khi dự đoán"""
        if not self.config.ENABLE_IMAGE_ENHANCEMENT:
            return image
        
        image = self.image_enhancer.denoise_image(image)
        image = self.image_enhancer.sharpen_image(image)
        image = self.image_enhancer.enhance_image(image, enhancement_level=self.config.CONTRAST_ENHANCEMENT)
        image = self.image_enhancer.adjust_gamma(image, gamma=self.config.GAMMA_CORRECTION)
        return image

    def predict_image(self, image_path: str):
        """Dự đoán một ảnh"""
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Không thể đọc ảnh: {image_path}")
            return None
        
        # Cải thiện ảnh
        enhanced = self.enhance_image_for_inference(image)
        
        # Dự đoán
        results = self.model.predict(
            enhanced,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.NMS_THRESHOLD,
            max_det=self.config.MAX_DETECTIONS,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_idx = int(box.cls)
                if class_idx < len(self.class_names):
                    class_label = self.class_names[class_idx]
                    class_label_vi = self.descriptions_vi[class_idx]
                    class_label_vi_no_diacritics = self.descriptions_vi_no_diacritics[class_idx]
                else:
                    class_label = f"class_{class_idx}"
                    class_label_vi = f"Lớp {class_idx}"
                    class_label_vi_no_diacritics = f"Lop {class_idx}"
                
                detection = {
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf),
                    'class_id': class_idx,
                    'class_label': class_label,
                    'class_label_vi': class_label_vi,
                    'class_label_vi_no_diacritics': class_label_vi_no_diacritics
                }
                detections.append(detection)
        
        return image, detections

    def draw_detections(self, image: np.ndarray, detections):
        """Vẽ kết quả dự đoán lên ảnh"""
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            confidence = det['confidence']
            class_label = det['class_label']
            class_label_vi = det['class_label_vi']
            
            # Màu sắc dựa trên confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Xanh lá - cao
            elif confidence > 0.6:
                color = (0, 255, 255)  # Vàng - trung bình
            else:
                color = (0, 0, 255)  # Đỏ - thấp
            
            # Vẽ bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ label
            label = f"{class_label} | {class_label_vi} | {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Vẽ background cho text
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (255, 255, 255), 2)
            
            # Vẽ text
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image

    def process_input_folder(self):
        """Xử lý tất cả ảnh trong folder input"""
        input_dir = 'input'
        output_dir = 'output'
        
        # Tạo thư mục output nếu chưa có
        os.makedirs(output_dir, exist_ok=True)
        
        # Kiểm tra thư mục input
        if not os.path.exists(input_dir):
            print(f"Thư mục {input_dir} không tồn tại!")
            return
        
        # Lấy danh sách file ảnh
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        if not image_files:
            print(f"Không tìm thấy file ảnh nào trong thư mục {input_dir}!")
            return
        
        print(f"[INFO] Tìm thấy {len(image_files)} file ảnh để xử lý")
        
        # Xử lý từng ảnh
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Đang xử lý: {image_file}")
            
            image_path = os.path.join(input_dir, image_file)
            
            # Dự đoán
            result = self.predict_image(image_path)
            if result is None:
                continue
            
            image, detections = result
            
            # Vẽ kết quả
            result_image = self.draw_detections(image.copy(), detections)
            
            # Lưu ảnh kết quả
            output_filename = f"result_{image_file}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, result_image)
            
            # In thông tin kết quả
            print(f"  ✅ Phát hiện {len(detections)} biển báo")
            for det in detections:
                print(f"     - {det['class_label_vi']} (confidence: {det['confidence']:.2f})")
            print(f"  💾 Đã lưu: {output_path}")
        
        print(f"\n🎉 Hoàn thành! Đã xử lý {len(image_files)} ảnh.")
        print(f"📁 Kết quả được lưu trong thư mục: {output_dir}")

def main():
    """Hàm chính"""
    print("🚦 TRAFFIC SIGN DETECTION - BATCH PREDICTION")
    print("=" * 50)
    
    try:
        # Khởi tạo predictor
        predictor = TrafficSignPredictor()
        
        # Xử lý folder input
        predictor.process_input_folder()
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        print("\n💡 Hướng dẫn sử dụng:")
        print("1. Đặt ảnh cần dự đoán vào thư mục 'input'")
        print("2. Chạy: python predict.py")
        print("3. Kết quả sẽ được lưu trong thư mục 'output'")

if __name__ == "__main__":
    main() 