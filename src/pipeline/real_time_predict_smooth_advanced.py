import os
import cv2
import time
from ultralytics import YOLO
import numpy as np
import yaml
try:
    from src.training.config import Config
    from src.core.utils import ImageEnhancer
    from src.sort import Sort
except ModuleNotFoundError:
    import sys as _sys, pathlib as _pathlib
    _repo_root = _pathlib.Path(__file__).resolve().parents[2]
    _sys.path.insert(0, str(_repo_root))
    from src.training.config import Config
    from src.core.utils import ImageEnhancer
    from src.sort import Sort
import unicodedata
from collections import deque, Counter

def bbox_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    # Intersection area
    inter_area = max(0, inter_rect_x2 - inter_rect_x1 + 1) * \
                 max(0, inter_rect_y2 - inter_rect_y1 + 1)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / float(b1_area + b2_area - inter_area)
    return iou

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

class RealTimeTrafficSignDetectorAdvanced:
    def __init__(self, model_path=None):
        if model_path is None:
            all_weight_dir = Config.ALL_WEIGHT_DIR
            os.makedirs(all_weight_dir, exist_ok=True)
            train_dirs = [d for d in os.listdir(all_weight_dir) if d.startswith('train') and os.path.isdir(os.path.join(all_weight_dir, d))]
            if not train_dirs:
                import pathlib as _pl
                repo_root = _pl.Path(__file__).resolve().parents[2]
                candidates = [repo_root / 'src' / 'weight' / 'yolov8m.pt',
                              repo_root / 'src' / 'weights' / 'yolov8m.pt',
                              repo_root / 'weights' / 'yolov8m.pt']
                for c in candidates:
                    if c.exists():
                        model_path = str(c)
                        print(f"[INFO] Không có model đã train; dùng pretrained local: {model_path}")
                        break
                else:
                    raise FileNotFoundError(
                        "Không tìm thấy weight local. Hãy đặt weight tại src/weights/all_weight/trainX/best.pt hoặc src/weights/yolov8m.pt")
            else:
                train_dirs_sorted = sorted(train_dirs, key=lambda x: int(x.replace('train', '')) if x.replace('train', '').isdigit() else 0)
                latest_train_dir = os.path.join(all_weight_dir, train_dirs_sorted[-1])
                best_pt_path = os.path.join(latest_train_dir, 'best.pt')
                if not os.path.exists(best_pt_path):
                    last_pt_path = os.path.join(latest_train_dir, 'last.pt')
                    if os.path.exists(last_pt_path):
                        best_pt_path = last_pt_path
                model_path = best_pt_path
                print(f"[INFO] Sử dụng model: {model_path}")
        self.model = YOLO(model_path)
        self.image_enhancer = ImageEnhancer()
        self.config = Config
        from src.core.utils import find_data_yaml
        data_yaml_path = find_data_yaml()
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
            self.class_names = data_yaml.get('names', {})
            self.descriptions_vi = data_yaml.get('descriptions', []) or []
        self.descriptions_vi_no_diacritics = [remove_vietnamese_diacritics(desc) for desc in self.descriptions_vi]
        self.tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3) # Tăng max_age, min_hits
        self.label_buffers = {}  # key: object_id, value: deque
        self.buffer_size = 10    # Tăng kích thước buffer

    def smooth_label(self, class_idx, object_id, confidence):
        if object_id not in self.label_buffers:
            self.label_buffers[object_id] = deque(maxlen=self.buffer_size)
        
        # Chỉ thêm vào buffer nếu confidence đủ cao
        if confidence >= self.config.CONFIDENCE_THRESHOLD - 0.1: # Cho phép hơi thấp hơn ngưỡng chung một chút
            self.label_buffers[object_id].append(class_idx)
        
        # Xóa các buffer của object không còn tồn tại
        current_track_ids = set(trk.id for trk in self.tracker.trackers)
        for obj_id in list(self.label_buffers.keys()):
            if obj_id not in current_track_ids:
                del self.label_buffers[obj_id]

        most_common = Counter(self.label_buffers.get(object_id, [])).most_common(1)
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
        dets = []  # [x1, y1, x2, y2, score]
        det_class_map = {} # map detection index to its class_idx and confidence

        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                bbox = box.xyxy[0].tolist()
                score = float(box.conf)
                class_idx = int(box.cls)
                dets.append(bbox + [score])
                det_class_map[len(dets) - 1] = {'class_idx': class_idx, 'confidence': score}

        if len(dets) == 0:
            trackers = np.empty((0, 5))
        else:
            trackers = self.tracker.update(np.array(dets))
        
        detections = []
        for trk in trackers:
            x1, y1, x2, y2, object_id = trk.astype(int)
            
            # Tìm detection gốc có IOU cao nhất với tracker này
            best_iou = -1
            matched_det_idx = -1
            for i, det_bbox_score in enumerate(dets):
                det_bbox = det_bbox_score[:4]
                current_iou = bbox_iou([x1,y1,x2,y2], det_bbox)
                if current_iou > best_iou:
                    best_iou = current_iou
                    matched_det_idx = i
            
            class_idx = 0 # default
            confidence = 0.0 # default
            if matched_det_idx != -1 and best_iou > 0.1: # Chỉ lấy class nếu IOU đủ lớn
                class_idx = det_class_map[matched_det_idx]['class_idx']
                confidence = det_class_map[matched_det_idx]['confidence']
            
            class_idx_smooth = self.smooth_label(class_idx, object_id, confidence)
            if isinstance(self.class_names, dict):
                class_label = self.class_names.get(class_idx_smooth, str(class_idx_smooth))
                class_id_code = class_label
            elif isinstance(self.class_names, list) and class_idx_smooth < len(self.class_names):
                class_label = self.class_names[class_idx_smooth]
                class_id_code = class_label
            else:
                class_label = str(class_idx_smooth)
                class_id_code = class_label
            class_label_vi = self.descriptions_vi_no_diacritics[class_idx_smooth] if class_idx_smooth < len(self.descriptions_vi_no_diacritics) else class_label

            detection = {
                'object_id': object_id,
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': class_idx_smooth,
                'class_label': class_label,
                'class_label_vi': class_label_vi,
                'class_id_code': class_id_code
            }
            detections.append(detection)

        return detections

    def draw_and_show(self, frame: np.ndarray, detections):
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"ID:{det['object_id']} | {det['class_id_code']} | {det['class_label_vi']} | {det['confidence']:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow('Traffic Sign Detection - SORT Smoothing', frame)

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
            output_dir = Config.REAL_TIME_OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            video_filename = os.path.join(output_dir, f"result_sort_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
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
            self.draw_and_show(frame_draw, detections)
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
    detector = RealTimeTrafficSignDetectorAdvanced()
    detector.run_webcam(save_video=True)

if __name__ == "__main__":
    main() 