import os
import sys
import cv2
import time
from ultralytics import YOLO
import numpy as np
import yaml
from config import Config
from utils import ImageEnhancer
import unicodedata
from collections import deque, Counter
from src.sort import Sort
import torch
from transformers import AutoModel, AutoProcessor

# Thêm thư mục gốc của project vào sys.path để xử lý absolute imports
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

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
descriptions_vi_no_diacritics = [remove_vietnamese_diacritics(desc) for desc in descriptions_vi]

class_ids = [
    "W.301", "W.302a", "P.101a", "P.123a", "W.207", "W.208", "W.212b", "P.124a", "S.507", "W.224", "P.131a",
    "S.407", "R.411", "P.135", "P.106a", "W.233a", "P.117a", "P.125", "P.108", "P.124b", "P.102", "W.233b",
    "W.235", "P.109", "S.501", "R.412", "R.412a", "W.211", "W.210", "P.106b", "P.111b", "R.413", "S.510",
    "P.124c", "W.212a", "P.111a", "P.132", "P.134", "W.302b", "P.127", "P.128", "P.129", "P.126", "R.407",
    "P.117b", "W.245", "R.407a", "P.130", "P.131b", "P.110", "W.222", "P.124d", "W.212c", "W.212d", "W.212e"
]

class RealTimeTrafficSignDetectorAdvanced:
    def __init__(self, model_path=None):
        if model_path is None:
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
        with open('data.yaml', 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
            self.class_names = data_yaml.get('names', {})
        self.tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3) # Tăng max_age, min_hits
        self.label_buffers = {}  # key: object_id, value: deque
        self.buffer_size = 10    # Tăng kích thước buffer

        self._load_nlp_model()
        # Các lớp biển báo mà chúng ta muốn chạy OCR để đọc chữ
        # Sử dụng tên không dấu đã được xử lý
        self.TARGET_CLASSES_FOR_NLP = [
            "Bien_gop_lan_duong_theo_phuong_tien",
            "Nguy_hiem_khac",
            "Chieu_cao_tinh_khong_thuc_te",
            "Cam_o_to_quay_dau_xe_(duoc_re_trai)"
        ]

    def _load_nlp_model(self):
        """Tải mô hình NLP (Vintern) và processor."""
        print("[INFO] Đang tải mô hình NLP (Vintern)...")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.nlp_processor = AutoProcessor.from_pretrained("5CD-AI/Vintern-1B-v3_5", trust_remote_code=True)
            self.nlp_model = AutoModel.from_pretrained("5CD-AI/Vintern-1B-v3_5", trust_remote_code=True).to(self.device)
            print(f"[INFO] Đã tải xong mô hình NLP và chạy trên {self.device}.")
        except Exception as e:
            print(f"[ERROR] Không thể tải mô hình NLP: {e}")
            self.nlp_model = None
            self.nlp_processor = None

    def _get_text_from_sign(self, sign_image: np.ndarray) -> str:
        """Sử dụng Vintern để đọc văn bản từ ảnh biển báo."""
        if self.nlp_model is None or self.nlp_processor is None:
            return ""

        try:
            # Chuyển BGR (cv2) sang RGB (transformers)
            rgb_image = cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB)
            # Xử lý ảnh và đưa qua model
            inputs = self.nlp_processor(images=rgb_image, text="<BOS>", return_tensors="pt").to(self.device)
            generated_ids = self.nlp_model.generate(**inputs, max_length=64)
            generated_text = self.nlp_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Loại bỏ các token không cần thiết
            cleaned_text = generated_text.replace("<BOS>", "").replace("<EOS>", "").strip()
            return cleaned_text
        except Exception as e:
            print(f"[ERROR] Lỗi khi xử lý OCR: {e}")
            return ""

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
            class_label = self.class_names[class_idx_smooth] if isinstance(self.class_names, list) and class_idx_smooth < len(self.class_names) else str(class_idx_smooth)
            class_label_vi = descriptions_vi_no_diacritics[class_idx_smooth] if class_idx_smooth < len(descriptions_vi_no_diacritics) else class_label
            class_id_code = class_ids[class_idx_smooth] if class_idx_smooth < len(class_ids) else str(class_idx_smooth)

            detection = {
                'object_id': object_id,
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': class_idx_smooth,
                'class_label': class_label,
                'class_label_vi': class_label_vi,
                'class_id_code': class_id_code,
                'ocr_text': None # Khởi tạo giá trị ocr_text
            }
            
            if class_label_vi in self.TARGET_CLASSES_FOR_NLP:
                # Cắt ảnh biển báo từ frame gốc để có chất lượng tốt nhất
                # Thêm padding nhỏ để đảm bảo không mất chữ ở viền
                padding = 5
                crop_x1 = max(0, x1 - padding)
                crop_y1 = max(0, y1 - padding)
                crop_x2 = min(frame.shape[1], x2 + padding)
                crop_y2 = min(frame.shape[0], y2 + padding)

                if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                    sign_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    ocr_text = self._get_text_from_sign(sign_crop)
                    if ocr_text:
                        detection['ocr_text'] = ocr_text

            detections.append(detection)

        return detections

    def draw_and_show(self, frame: np.ndarray, detections):
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"ID:{det['object_id']} | {det['class_id_code']} | {det['class_label_vi']} | {det['confidence']:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Hiển thị kết quả OCR nếu có
            if det.get('ocr_text'):
                ocr_label = f"OCR: {det['ocr_text']}"
                # Vị trí hiển thị text OCR, nằm bên dưới bounding box
                text_y_pos = y2 + 20
                (text_width, text_height), baseline = cv2.getTextSize(ocr_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                # Vẽ background cho text OCR để dễ đọc hơn
                cv2.rectangle(frame, (x1, text_y_pos - text_height - baseline), (x1 + text_width, text_y_pos + baseline), (0, 0, 0), -1)
                cv2.putText(frame, ocr_label, (x1, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
            output_dir = 'real_time_output'
            os.makedirs(output_dir, exist_ok=True)
            video_filename = os.path.join(output_dir, f"result_sort_mobile_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
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
    # --- SỬ DỤNG CAMERA ANDROID (IP WEBCAM) ---
    # 1. Cài đặt ứng dụng "IP Webcam" trên điện thoại Android của bạn.
    # 2. Đảm bảo điện thoại và máy tính của bạn được kết nối vào cùng một mạng Wi-Fi.
    # 3. Mở ứng dụng "IP Webcam" trên điện thoại, cuộn xuống và nhấn "Start server".
    # 4. Ứng dụng sẽ hiển thị một địa chỉ IP (ví dụ: http://192.168.1.100:8080).
    # 5. Sao chép địa chỉ đó, dán vào biến `camera_url` bên dưới và thêm "/video" vào cuối.
    # 6. Nếu bạn muốn quay lại dùng webcam của máy tính, hãy đặt camera_url = 0

    # Thay thế URL bên dưới bằng địa chỉ từ ứng dụng IP Webcam của bạn
    # Ví dụ: camera_url = "http://192.168.1.100:8080/video"
    camera_url = "http://172.172.10.236:8080/video"  # Dùng camera từ điện thoại

    detector = RealTimeTrafficSignDetectorAdvanced()
    detector.run_webcam(cam_id=camera_url, save_video=True)

if __name__ == "__main__":
    main() 