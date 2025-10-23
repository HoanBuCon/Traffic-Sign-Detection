import os
import sys
import cv2
import time
from ultralytics import YOLO
import numpy as np
import yaml
from config import Config
from utils import ImageEnhancer, VisualizationUtils
import unicodedata
from collections import deque, Counter

# Thêm thư mục gốc của project vào sys.path để xử lý relative imports
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Danh sách nhãn tiếng Việt (copy từ predict.py)
descriptions_vi = [
  "Đường kẻ dành cho người đi bộ",           # i.423.b
  "Cấm đi ngược chiều",                      # p.102
  "Cấm xe tải trên N tấn",                   # p.106.b
  "Cấm dừng và đỗ xe",                       # p.130
  "Cấm đỗ xe",                               # p.131.a
  "Tuyến đường cầu vượt cắt qua",            # r.308.b
  "TEXT SIGN",                               # sus
  "Cảnh báo khúc cua nguy hiểm bên trái",    # w.201.a
  "Cảnh báo đường hẹp",                      # w.203.c
  "Cảnh báo giao nhau với đường không ưu tiên bên phải",  # w.207.b
  "Cảnh báo giao nhau với đường không ưu tiên bên trái",  # w.207.c
  "Cảnh báo đến khu vực đèn tín hiệu giao thông"          # w.209
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

class RealTimeTrafficSignDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            # Tìm model mới nhất trong all_weight
            all_weight_dir = 'all_weight'
            train_dirs = [d for d in os.listdir(all_weight_dir) if d.startswith('train') and os.path.isdir(os.path.join(all_weight_dir, d))]
            if not train_dirs:
                raise FileNotFoundError("Không tìm thấy model đã train trong all_weight! Hãy train model trước.")
            train_dirs_sorted = sorted(train_dirs, key=lambda x: int(x.replace('train', '')) if x.replace('train', '').isdigit() else 0)
            latest_train_dir = os.path.join(all_weight_dir, train_dirs_sorted[-1])
            best_pt_path = os.path.join(latest_train_dir, 'best.pt')
            if not os.path.exists(best_pt_path):
                raise FileNotFoundError(f"Không tìm thấy best.pt trong {latest_train_dir}!")
            model_path = best_pt_path
            print(f"[INFO] Sử dụng model: {model_path}")
        self.model = YOLO(model_path)
        self.image_enhancer = ImageEnhancer()
        self.config = Config
        # Đọc class_names từ data.yaml
        data_yaml_path = 'data.yaml'
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
            self.class_names = data_yaml.get('names', {})
        self.label_buffers = {}  # Buffer cho từng object theo class_idx, key là class_idx, value là deque
        self.buffer_size = 5     # Số frame để voting

    def smooth_label(self, class_idx, obj_id):
        if obj_id not in self.label_buffers:
            self.label_buffers[obj_id] = deque(maxlen=self.buffer_size)
        self.label_buffers[obj_id].append(class_idx)
        most_common = Counter(self.label_buffers[obj_id]).most_common(1)
        return most_common[0][0] if most_common else class_idx

    def enhance_image_for_inference(self, image: np.ndarray) -> np.ndarray:
        if not self.config.ENABLE_IMAGE_ENHANCEMENT:
            return image
        image = self.image_enhancer.denoise_image(image)
        image = self.image_enhancer.sharpen_image(image)
        image = self.image_enhancer.enhance_image(image, enhancement_level=self.config.CONTRAST_ENHANCEMENT)
        image = self.image_enhancer.adjust_gamma(image, gamma=self.config.GAMMA_CORRECTION)
        return image

    def predict_frame(self, frame: np.ndarray):
        enhanced = self.enhance_image_for_inference(frame)
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
                # Làm mượt nhãn theo class_idx (vì không tracking object)
                class_idx_smooth = self.smooth_label(class_idx, class_idx)
                class_label = self.class_names[class_idx_smooth] if isinstance(self.class_names, list) and class_idx_smooth < len(self.class_names) else str(class_idx_smooth)
                class_label_vi = descriptions_vi_no_diacritics[class_idx_smooth] if class_idx_smooth < len(descriptions_vi_no_diacritics) else class_label
                detection = {
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf),
                    'class_id': class_idx_smooth,
                    'class_label': class_label,
                    'class_label_vi': class_label_vi
                }
                detections.append(detection)
        return detections

    def draw_and_show(self, frame: np.ndarray, detections):
        # Vẽ kết quả lên ảnh
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['class_label']} | {det['class_label_vi']} | {det['confidence']:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow('Traffic Sign Detection - Press q to quit', frame)

    def run_webcam(self, cam_id=0, save_video=True):
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print("Không mở được camera!")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 20
        out = None
        if save_video:
            output_dir = 'real_time_output'
            os.makedirs(output_dir, exist_ok=True)
            video_filename = os.path.join(output_dir, f"result_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            print(f"[INFO] Video sẽ được lưu tại: {video_filename}")
        print("Nhấn 'q' để thoát.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không lấy được frame từ camera!")
                break
            detections = self.predict_frame(frame)
            frame_draw = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                label = f"{det['class_label']} | {det['class_label_vi']} | {det['confidence']:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_draw, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow('Traffic Sign Detection - Press q to quit', frame_draw)
            if out is not None:
                out.write(frame_draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        if out is not None:
            print(f"[INFO] Video đã lưu tại: {video_filename}")

def main():
    detector = RealTimeTrafficSignDetector()
    detector.run_webcam(save_video=True)

if __name__ == "__main__":
    main()