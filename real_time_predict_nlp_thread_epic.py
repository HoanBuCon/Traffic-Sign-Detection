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
from PIL import Image, ImageFont, ImageDraw
import torchvision.transforms as T
from transformers import AutoModel, AutoTokenizer
import threading
import queue
import logging
import datetime

# ==== CONFIG TỐI ƯU ====
EPIC_CONFIG = {
    'OCR_QUEUE_SIZE': 16,           # Số task OCR tối đa trong queue
    'OCR_NUM_WORKERS': 2,           # Số thread OCR song song
    'OCR_IOU_THRESHOLD': 0.4,       # IOU để quyết định OCR lại
    'OCR_FRAME_REFRESH': 20,        # Số frame phải OCR lại
    'SHOW_FPS': True,
    'LOG_OCR_TO_FILE': True,
    'OCR_LOG_FILE': 'ocr_log.txt',
    'SAVE_VIDEO': True,
    'PAUSE_HOTKEY': 'p',
    'SNAPSHOT_HOTKEY': 's',
    'EXIT_HOTKEY': 'q',
}

# ==== LOGGING ====
if EPIC_CONFIG['LOG_OCR_TO_FILE']:
    logging.basicConfig(filename=EPIC_CONFIG['OCR_LOG_FILE'], level=logging.INFO, format='%(asctime)s %(message)s')

# ==== Vintern pipeline & util giữ nguyên như cũ ====
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_for_vintern(image, input_size=448, max_num=4):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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

# ==== Danh sách class giữ nguyên như cũ (cắt bớt cho gọn) ====
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

def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    inter_area = max(0, inter_rect_x2 - inter_rect_x1 + 1) * max(0, inter_rect_y2 - inter_rect_y1 + 1)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / float(b1_area + b2_area - inter_area)
    return iou

# ==== CLASS CHÍNH ====
class RealTimeTrafficSignDetectorNLPThreadEpic:
    def __init__(self, model_path=None):
        print("[EPIC] Init class RealTimeTrafficSignDetectorNLPThreadEpic")
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
        self.tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
        # Queue OCR không giới hạn
        self.ocr_queue = queue.Queue()
        self.ocr_cache = {}
        self.ocr_queue_sent = set()
        self.label_buffers = {}
        self.buffer_size = 10
        self.TARGET_CLASSES_FOR_NLP = [
            "Bien_gop_lan_duong_theo_phuong_tien",
            "Nguy_hiem_khac",
            "Chieu_cao_tinh_khong_thuc_te",
            "Cam_o_to_quay_dau_xe_(duoc_re_trai)"
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_nlp_model()
        # Chỉ khởi động 1 thread OCR
        threading.Thread(target=self.ocr_worker, daemon=True).start()
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        self.frame_in_second = 0
        self.paused = False
        # Khởi động nhiều thread OCR (ví dụ: 4 thread)
        self.num_ocr_threads = 4
        for _ in range(self.num_ocr_threads):
            threading.Thread(target=self.ocr_worker, daemon=True).start()

    def _load_nlp_model(self):
        print("[INFO] Đang tải mô hình Vintern NLP...")
        try:
            self.vintern_model = AutoModel.from_pretrained(
                "5CD-AI/Vintern-1B-v3_5",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=False,
            ).eval().to(self.device)
            self.vintern_tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-1B-v3_5", trust_remote_code=True, use_fast=False)
            print(f"[INFO] Đã tải xong Vintern NLP và chạy trên {self.device}.")
        except Exception as e:
            print(f"[ERROR] Không thể tải Vintern NLP: {e}")
            self.vintern_model = None
            self.vintern_tokenizer = None

    def ocr_worker(self):
        while True:
            try:
                object_id, sign_crop = self.ocr_queue.get()
                ocr_text = self._get_text_from_sign(sign_crop)
                if ocr_text:
                    self.ocr_cache[object_id] = ocr_text
                    if EPIC_CONFIG['LOG_OCR_TO_FILE']:
                        logging.info(f"{object_id}|{ocr_text}")
                self.ocr_queue.task_done()
            except Exception as e:
                print(f"[ERROR] OCR thread: {e}")

    def _get_text_from_sign(self, sign_image: np.ndarray) -> str:
        if self.vintern_model is None or self.vintern_tokenizer is None:
            return ""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB))
            pixel_values = load_image_for_vintern(pil_image, input_size=448, max_num=4).to(torch.float16).to(self.device)
            generation_config = dict(max_new_tokens=128, do_sample=False, num_beams=3, repetition_penalty=1.5, pad_token_id=self.vintern_tokenizer.eos_token_id)
            question = "<image>\nChỉ trả về nội dung chữ trên biển báo, không giải thích gì thêm."
            response, history = self.vintern_model.chat(self.vintern_tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            response = response.strip().split('\n')[0]
            response = response.replace('**', '').replace('`', '').strip()
            return response
        except Exception as e:
            print(f"[ERROR] Lỗi khi xử lý OCR Vintern: {e}")
            return ""

    def smooth_label(self, class_idx, object_id, confidence):
        if object_id not in self.label_buffers:
            self.label_buffers[object_id] = deque(maxlen=self.buffer_size)
        if confidence >= self.config.CONFIDENCE_THRESHOLD - 0.1:
            self.label_buffers[object_id].append(class_idx)
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
        self.frame_count += 1
        enhanced = self.enhance_image_for_inference(frame)
        results = self.model.predict(
            enhanced,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.NMS_THRESHOLD,
            max_det=self.config.MAX_DETECTIONS,
            verbose=False
        )
        dets = []
        det_class_map = {}
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
            best_iou = -1
            matched_det_idx = -1
            for i, det_bbox_score in enumerate(dets):
                det_bbox = det_bbox_score[:4]
                current_iou = bbox_iou([x1,y1,x2,y2], det_bbox)
                if current_iou > best_iou:
                    best_iou = current_iou
                    matched_det_idx = i
            class_idx = 0
            confidence = 0.0
            if matched_det_idx != -1 and best_iou > 0.1:
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
                'ocr_text': None
            }
            if class_label_vi in self.TARGET_CLASSES_FOR_NLP:
                padding = 5
                crop_x1 = max(0, x1 - padding)
                crop_y1 = max(0, y1 - padding)
                crop_x2 = min(frame.shape[1], x2 + padding)
                crop_y2 = min(frame.shape[0], y2 + padding)
                if object_id in self.ocr_cache:
                    detection['ocr_text'] = self.ocr_cache[object_id]
                else:
                    if (object_id, ) not in self.ocr_queue_sent:
                        if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                            sign_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            self.ocr_queue.put((object_id, sign_crop.copy()))
                            self.ocr_queue_sent.add((object_id, ))
                    detection['ocr_text'] = "..."
            detections.append(detection)
        return detections

    def draw_and_show(self, frame: np.ndarray, detections):
        # Vẽ bounding box bằng OpenCV trước
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Chuyển frame sang PIL để vẽ text Unicode
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font_path = os.path.join(os.path.dirname(__file__), "arial.ttf")
        if not os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/arial.ttf"
        try:
            font = ImageFont.truetype(font_path, 20)
        except:
            font = ImageFont.load_default()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"ID:{det['object_id']} | {det['class_id_code']} | {det['class_label_vi']} | {det['confidence']:.2f}"
            draw.text((x1, y1 - 25), label, font=font, fill=(0,255,0))
            if det.get('ocr_text') and det['ocr_text'] != '...':
                ocr_label = f"OCR: {det['ocr_text']}"
                draw.text((x1, y2 + 5), ocr_label, font=font, fill=(255,255,255))
                print(f"{det['object_id']} | {det['class_id_code']} | {det['class_label_vi']} | {det['ocr_text']}")
        if 'SHOW_FPS' in EPIC_CONFIG and EPIC_CONFIG['SHOW_FPS']:
            draw.text((10, 10), f"FPS: {self.fps}", font=font, fill=(255,0,0))
            draw.text((10, 40), f"Track: {len(detections)}", font=font, fill=(255,0,0))
            draw.text((10, 70), f"OCR queue: {self.ocr_queue.qsize()}", font=font, fill=(255,0,0))
        frame_show = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow('Traffic Sign Detection - SORT Smoothing EPIC', frame_show)

    def run_webcam(self, cam_id=0, save_video=None):
        if save_video is None:
            save_video = EPIC_CONFIG['SAVE_VIDEO']
        try:
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
                video_filename = os.path.join(output_dir, f"result_sort_epic_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
                print(f"[INFO] Video sẽ được lưu tại: {video_filename}")
            print(f"Nhấn '{EPIC_CONFIG['EXIT_HOTKEY']}' để thoát, '{EPIC_CONFIG['PAUSE_HOTKEY']}' tạm dừng, '{EPIC_CONFIG['SNAPSHOT_HOTKEY']}' lưu ảnh.")
            while True:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Không lấy được frame từ camera!")
                        break
                    detections = self.predict_frame(frame)
                    frame_draw = frame.copy()
                    self.draw_and_show(frame_draw, detections)
                    if out is not None:
                        out.write(frame_draw)
                    # FPS tính toán
                    self.frame_in_second += 1
                    now = time.time()
                    if now - self.last_fps_time >= 1.0:
                        self.fps = self.frame_in_second
                        self.frame_in_second = 0
                        self.last_fps_time = now
                key = cv2.waitKey(1) & 0xFF
                if key == ord(EPIC_CONFIG['EXIT_HOTKEY']):
                    break
                elif key == ord(EPIC_CONFIG['PAUSE_HOTKEY']):
                    self.paused = not self.paused
                    print("[EPIC] Đã tạm dừng" if self.paused else "[EPIC] Tiếp tục")
                elif key == ord(EPIC_CONFIG['SNAPSHOT_HOTKEY']):
                    snapshot_path = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(snapshot_path, frame_draw)
                    print(f"[EPIC] Đã lưu ảnh: {snapshot_path}")
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            if out is not None:
                print(f"[INFO] Video đã lưu tại: {video_filename}")
        except Exception as e:
            import traceback
            print("[ERROR] Lỗi khi chạy webcam:", e)
            traceback.print_exc()

def main():
    print("[EPIC] Khởi động chương trình NLP Thread EPIC oách xà lách vkl")
    detector = RealTimeTrafficSignDetectorNLPThreadEpic()
    print("[EPIC] Đã tạo detector")
    detector.run_webcam()
    print("[EPIC] Kết thúc chương trình NLP Thread EPIC oách xà lách vkl")

if __name__ == "__main__":
    main() 