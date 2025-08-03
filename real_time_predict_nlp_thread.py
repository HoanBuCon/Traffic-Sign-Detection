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
import hashlib

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

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

# Load descriptions từ data.yaml
def load_descriptions_from_yaml(yaml_path='data.yaml'):
    """Load descriptions từ file data.yaml"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
            descriptions = data_yaml.get('descriptions', [])
            if not descriptions:
                print(f"[WARNING] No descriptions found in {yaml_path}")
                return []
            else:
                print(f"[INFO] Loaded {len(descriptions)} descriptions from {yaml_path}")
            return descriptions
    except Exception as e:
        print(f"[ERROR] Failed to load descriptions from {yaml_path}: {e}")
        return []

# Load descriptions từ data.yaml
descriptions_vi = load_descriptions_from_yaml()
descriptions_vi_no_diacritics = [remove_vietnamese_diacritics(desc) for desc in descriptions_vi]

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

class RealTimeTrafficSignDetectorNLPThread:
    def __init__(self, model_path=None):
        print("[DEBUG] Init class RealTimeTrafficSignDetectorNLPThread")
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
        print("[DEBUG] Before init OCR queue/cache/label_buffers")
        self.ocr_queue = queue.Queue()
        self.ocr_cache = {}
        self.ocr_queue_sent = set()
        self.ocr_bbox_cache = {}  # Cache để lưu bbox cuối cùng của mỗi object
        self.ocr_image_hash_cache = {}  # Cache để lưu hash của ảnh đã OCR
        self.ocr_timestamp_cache = {}  # Cache timestamp để hết hạn
        self.label_buffers = {}
        self.buffer_size = 10
        self.ocr_cache_timeout = 30.0  # Cache timeout 30 giây
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[DEBUG] Before load Vintern NLP model")
        self._load_nlp_model()
        print("[DEBUG] Before start OCR thread")
        threading.Thread(target=self.ocr_worker, daemon=True).start()
        print("[DEBUG] After start OCR thread")

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

    def _compute_image_hash(self, image: np.ndarray) -> str:
        """Tính hash của ảnh để xác định nội dung duy nhất"""
        try:
            # Resize ảnh về kích thước cố định để tránh ảnh hưởng của scale
            resized = cv2.resize(image, (64, 64))
            # Chuyển sang grayscale để giảm ảnh hưởng của màu sắc
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # Tính hash MD5
            img_hash = hashlib.md5(gray.tobytes()).hexdigest()
            return img_hash
        except Exception as e:
            print(f"[ERROR] Không thể tính hash ảnh: {e}")
            return ""

    def _is_cache_valid(self, object_id: int, current_bbox: list, image_hash: str) -> bool:
        """Kiểm tra cache có còn hợp lệ không"""
        current_time = time.time()
        
        # Kiểm tra timeout
        if object_id in self.ocr_timestamp_cache:
            if current_time - self.ocr_timestamp_cache[object_id] > self.ocr_cache_timeout:
                return False
        
        # Kiểm tra bbox thay đổi
        if object_id in self.ocr_bbox_cache:
            old_bbox = self.ocr_bbox_cache[object_id]
            iou_bbox = bbox_iou(current_bbox, old_bbox)
            if iou_bbox < 0.7:
                return False
        
        # Kiểm tra nội dung ảnh thay đổi (quan trọng nhất)
        if object_id in self.ocr_image_hash_cache:
            old_hash = self.ocr_image_hash_cache[object_id]
            if old_hash != image_hash:
                return False
        
        return True

    def _clear_object_cache(self, object_id: int):
        """Xóa toàn bộ cache của một object"""
        caches_to_clear = [
            self.ocr_cache,
            self.ocr_bbox_cache, 
            self.ocr_image_hash_cache,
            self.ocr_timestamp_cache
        ]
        
        for cache_dict in caches_to_clear:
            if object_id in cache_dict:
                del cache_dict[object_id]
        
        self.ocr_queue_sent.discard((object_id, ))

    def _update_object_cache(self, object_id: int, bbox: list, image_hash: str, ocr_result: str = None):
        """Cập nhật cache cho object"""
        current_time = time.time()
        self.ocr_bbox_cache[object_id] = bbox
        self.ocr_image_hash_cache[object_id] = image_hash
        self.ocr_timestamp_cache[object_id] = current_time
        
        if ocr_result is not None:
            self.ocr_cache[object_id] = ocr_result

    def _cleanup_expired_cache(self):
        """Dọn dẹp cache hết hạn"""
        current_time = time.time()
        expired_objects = []
        
        for object_id, timestamp in self.ocr_timestamp_cache.items():
            if current_time - timestamp > self.ocr_cache_timeout:
                expired_objects.append(object_id)
        
        for object_id in expired_objects:
            self._clear_object_cache(object_id)

    def ocr_worker(self):
        while True:
            object_id, sign_crop = self.ocr_queue.get()
            try:
                # Tính hash của ảnh để đảm bảo tính nhất quán
                image_hash = self._compute_image_hash(sign_crop)
                
                # Kiểm tra xem object_id có còn hợp lệ không
                if object_id in self.ocr_image_hash_cache:
                    cached_hash = self.ocr_image_hash_cache[object_id]
                    if cached_hash != image_hash:
                        continue  # Skip OCR nếu hash không khớp
                
                ocr_text = self._get_text_from_sign(sign_crop)
                if ocr_text and len(ocr_text.strip()) > 0:
                    # Cập nhật cache với kết quả OCR
                    self.ocr_cache[object_id] = ocr_text
                    print(f"[OCR DEBUG] Cached OCR result for object {object_id}: {ocr_text}")
            except Exception as e:
                print(f"[ERROR] OCR thread: {e}")
            finally:
                self.ocr_queue.task_done()

    def _get_text_from_sign(self, sign_image: np.ndarray) -> str:
        if self.vintern_model is None or self.vintern_tokenizer is None:
            return ""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB))
            pixel_values = load_image_for_vintern(pil_image, input_size=448, max_num=4).to(torch.float16).to(self.device)
            generation_config = dict(max_new_tokens=200, do_sample=False, num_beams=3, repetition_penalty=1.3, pad_token_id=self.vintern_tokenizer.eos_token_id)
            
            # Prompt tối ưu để đọc được cả chữ và hướng mũi tên theo cách tự nhiên
            question = """<image>
            Bạn là trợ lý GPS. Hãy đọc biển báo và thông báo cho người lái xe:

            • Biển báo có những hướng đi nào theo mũi tên ? (đi thẳng / rẽ trái / rẽ phải)
            • Ở mỗi hướng đó, biển báo ghi đến địa điểm nào ? Khoảng cách bao nhiêu ?

            Yêu cầu:
            - Chỉ nói các hướng có mũi tên thật sự được ghi trên biển.
            - Ghi rõ: Hướng - Địa điểm - Khoảng cách (nếu có).
            - Mỗi hướng một dòng, ngắn gọn.
            """
            
            response, history = self.vintern_model.chat(self.vintern_tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            
            # Xử lý response để giữ lại thông tin quan trọng
            response = response.strip()
            response = response.replace('**', '').replace('`', '').replace('*', '')
            
            # Tách các dòng và lọc thông tin hữu ích
            all_lines = []
            for line in response.split('\n'):
                line = line.strip()
                if line and len(line) >= 3 and len(line) <= 120:
                    # Loại bỏ các dòng mô tả không cần thiết
                    if not any(skip in line.lower() for skip in [
                        'biển báo', 'hình ảnh', 'theo thứ tự', 'liệt kê', 'đọc được',
                        'nhìn thấy', 'có thể thấy', 'xuất hiện', 'tôi thấy', 'dựa vào'
                    ]):
                        # Làm sạch dòng
                        clean_line = line.lstrip('- ').lstrip('• ').lstrip('1234567890. ').strip()
                        if clean_line and len(clean_line) >= 2:
                            all_lines.append(clean_line)
            
            # Kết hợp các dòng bằng " | "
            if all_lines:
                result = ' | '.join(all_lines[:4])  # Lấy tối đa 4 dòng đầu tiên
                print(f"[OCR DEBUG] Found {len(all_lines)} valid lines: {result}")
                return result
            else:
                # Fallback: lấy response gốc và làm sạch
                fallback = response.replace('\n', ' ').strip()
                if len(fallback) > 200:
                    fallback = fallback[:200] + "..."
                return fallback
                
        except Exception as e:
            print(f"[ERROR] Lỗi khi xử lý OCR Vintern: {e}")
            return ""

    def smooth_label(self, class_idx, object_id, confidence):
        if object_id not in self.label_buffers:
            self.label_buffers[object_id] = deque(maxlen=self.buffer_size)
        if confidence >= self.config.CONFIDENCE_THRESHOLD - 0.1:
            self.label_buffers[object_id].append(class_idx)
        
        # Làm sạch cache cho các object đã biến mất
        current_track_ids = set(trk.id for trk in self.tracker.trackers)
        for obj_id in list(self.label_buffers.keys()):
            if obj_id not in current_track_ids:
                del self.label_buffers[obj_id]
                # Xóa toàn bộ cache liên quan đến object này
                self._clear_object_cache(obj_id)
        
        # Dọn dẹp cache hết hạn
        self._cleanup_expired_cache()
        
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
            detection = {
                'object_id': object_id,
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': class_idx_smooth,
                'class_label': class_label,
                'class_label_vi': class_label_vi,
                'ocr_text': None
            }
            if class_label == "sus":
                padding = 5
                crop_x1 = max(0, x1 - padding)
                crop_y1 = max(0, y1 - padding)
                crop_x2 = min(frame.shape[1], x2 + padding)
                crop_y2 = min(frame.shape[0], y2 + padding)
                
                # Kiểm tra crop hợp lệ
                if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
                    detection['ocr_text'] = "Invalid crop"
                    detections.append(detection)
                    continue
                
                current_bbox = [crop_x1, crop_y1, crop_x2, crop_y2]
                sign_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Tính hash của ảnh crop để xác định nội dung
                image_hash = self._compute_image_hash(sign_crop)
                if not image_hash:
                    detection['ocr_text'] = "Hash error"
                    detections.append(detection)
                    continue
                
                # Kiểm tra cache có còn hợp lệ không
                cache_valid = self._is_cache_valid(object_id, current_bbox, image_hash)
                
                if cache_valid and object_id in self.ocr_cache:
                    # Sử dụng cache hợp lệ
                    detection['ocr_text'] = self.ocr_cache[object_id]
                else:
                    # Cache không hợp lệ hoặc không tồn tại
                    if not cache_valid:
                        self._clear_object_cache(object_id)
                    
                    # Cập nhật cache với thông tin mới
                    self._update_object_cache(object_id, current_bbox, image_hash)
                    
                    # Gửi tới OCR queue nếu chưa gửi
                    if (object_id, ) not in self.ocr_queue_sent:
                        self.ocr_queue.put((object_id, sign_crop.copy()))
                        self.ocr_queue_sent.add((object_id, ))
                    
                    detection['ocr_text'] = "⏳ Processing..."
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
            # Lấy mã biển báo từ class_names thay vì dùng class_id_code hardcode
            class_code = det['class_label'] if 'class_label' in det else f"Class_{det['class_id']}"
            description_no_diacritics = det['class_label_vi'] if 'class_label_vi' in det else class_code
            confidence = det['confidence']
            
            # Format: Mã biển | Description_tieng_viet_khong_dai | Độ tin cậy
            label = f"{class_code} | {description_no_diacritics} | {confidence:.1%}"
            draw.text((x1, y1 - 25), label, font=font, fill=(0,255,0))
            
            # Chỉ hiển thị OCR khi có kết quả thành công (không phải "..." hay "⏳ Processing...")
            if (det.get('ocr_text') and 
                det['ocr_text'] != "..." and 
                det['ocr_text'] != "⏳ Processing..." and
                not det['ocr_text'].startswith("Invalid") and
                not det['ocr_text'].startswith("Hash error") and
                len(det['ocr_text'].strip()) > 0):
                ocr_label = f"OCR: {det['ocr_text']}"
                draw.text((x1, y2 + 5), ocr_label, font=font, fill=(255,255,255))
                print(f"{class_code} | {description_no_diacritics} | {det['ocr_text']}")
        frame_show = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow('Traffic Sign Detection - SORT Smoothing', frame_show)

    def run_webcam(self, cam_id=0, save_video=True):
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
        except Exception as e:
            import traceback
            print("[ERROR] Lỗi khi chạy webcam:", e)
            traceback.print_exc()

def main():
    print("[DEBUG] Khởi động chương trình")
    detector = RealTimeTrafficSignDetectorNLPThread()
    print("[DEBUG] Đã tạo detector")
    detector.run_webcam(save_video=True)
    print("[DEBUG] Kết thúc chương trình")

if __name__ == "__main__":
    main() 