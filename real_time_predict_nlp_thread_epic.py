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
import math
from scipy.spatial.distance import euclidean

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

# ==== COMPUTER VISION ARROW DETECTION ====
class ArrowDetector:
    """Phân tích hướng mũi tên bằng Computer Vision"""
    
    def __init__(self):
        self.arrow_templates = self._create_arrow_templates()
    
    def _create_arrow_templates(self):
        """Tạo template mũi tên cho template matching"""
        templates = {}
        # Template mũi tên trái (30x20)
        left_arrow = np.zeros((20, 30), dtype=np.uint8)
        cv2.arrowedLine(left_arrow, (25, 10), (5, 10), 255, 2, tipLength=0.3)
        templates['left'] = left_arrow
        
        # Template mũi tên phải
        right_arrow = np.zeros((20, 30), dtype=np.uint8)
        cv2.arrowedLine(right_arrow, (5, 10), (25, 10), 255, 2, tipLength=0.3)
        templates['right'] = right_arrow
        
        # Template mũi tên thẳng
        straight_arrow = np.zeros((30, 20), dtype=np.uint8)
        cv2.arrowedLine(straight_arrow, (10, 25), (10, 5), 255, 2, tipLength=0.3)
        templates['straight'] = straight_arrow
        
        return templates
    
    def detect_arrow_direction(self, sign_image: np.ndarray) -> dict:
        """
        Phân tích hướng mũi tên bằng nhiều phương pháp
        Returns: {'direction': 'left/right/straight', 'confidence': float, 'angle': float}
        """
        try:
            # Chuyển sang grayscale
            if len(sign_image.shape) == 3:
                gray = cv2.cvtColor(sign_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = sign_image.copy()
            
            # Phương pháp 1: Template Matching
            template_result = self._template_matching(gray)
            
            # Phương pháp 2: Contour Analysis
            contour_result = self._contour_analysis(gray)
            
            # Phương pháp 3: Edge Direction Analysis
            edge_result = self._edge_direction_analysis(gray)
            
            # Kết hợp kết quả từ 3 phương pháp
            final_result = self._combine_results([template_result, contour_result, edge_result])
            
            return final_result
            
        except Exception as e:
            print(f"[ARROW] Error detecting arrow: {e}")
            return {'direction': 'unknown', 'confidence': 0.0, 'angle': 0.0}
    
    def _template_matching(self, gray_image):
        """Template matching với các mũi tên chuẩn"""
        best_match = {'direction': 'unknown', 'confidence': 0.0, 'angle': 0.0}
        
        for direction, template in self.arrow_templates.items():
            # Resize template cho phù hợp
            h, w = template.shape
            resized_template = cv2.resize(template, (min(gray_image.shape[1]//3, w), min(gray_image.shape[0]//3, h)))
            
            # Template matching
            result = cv2.matchTemplate(gray_image, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_match['confidence']:
                best_match = {
                    'direction': direction,
                    'confidence': max_val,
                    'angle': self._direction_to_angle(direction)
                }
        
        return best_match
    
    def _contour_analysis(self, gray_image):
        """Phân tích contour để tìm hình dạng mũi tên"""
        # Threshold và tìm contours
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_arrow = {'direction': 'unknown', 'confidence': 0.0, 'angle': 0.0}
        
        for contour in contours:
            # Lọc contour quá nhỏ
            if cv2.contourArea(contour) < 50:
                continue
            
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Phân tích hình dạng mũi tên
            if len(approx) >= 5:  # Mũi tên thường có 5-7 điểm
                arrow_analysis = self._analyze_arrow_shape(approx)
                if arrow_analysis['confidence'] > best_arrow['confidence']:
                    best_arrow = arrow_analysis
        
        return best_arrow
    
    def _analyze_arrow_shape(self, approx_contour):
        """Phân tích hình dạng để xác định hướng mũi tên"""
        points = approx_contour.reshape(-1, 2)
        
        # Tìm điểm xa nhất (có thể là đầu mũi tên)
        center = np.mean(points, axis=0)
        distances = [euclidean(point, center) for point in points]
        tip_idx = np.argmax(distances)
        tip_point = points[tip_idx]
        
        # Tính vector từ center đến tip
        direction_vector = tip_point - center
        angle = math.atan2(direction_vector[1], direction_vector[0]) * 180 / math.pi
        
        # Xác định hướng dựa trên góc
        if -45 <= angle <= 45:
            direction = 'right'
        elif 135 <= angle <= 180 or -180 <= angle <= -135:
            direction = 'left'
        elif 45 < angle < 135:
            direction = 'straight'
        else:
            direction = 'unknown'
        
        # Confidence dựa trên độ đối xứng và tỷ lệ
        confidence = min(1.0, len(points) / 10.0)  # Càng nhiều điểm càng tốt
        
        return {'direction': direction, 'confidence': confidence, 'angle': angle}
    
    def _edge_direction_analysis(self, gray_image):
        """Phân tích hướng edges để xác định mũi tên"""
        # Sobel edge detection
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Tính hướng gradient
        angles = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        magnitudes = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Lọc edges mạnh
        strong_edges = magnitudes > np.percentile(magnitudes, 80)
        strong_angles = angles[strong_edges]
        
        if len(strong_angles) == 0:
            return {'direction': 'unknown', 'confidence': 0.0, 'angle': 0.0}
        
        # Phân tích histogram góc
        hist, bin_edges = np.histogram(strong_angles, bins=36, range=(-180, 180))
        dominant_angle_idx = np.argmax(hist)
        dominant_angle = (bin_edges[dominant_angle_idx] + bin_edges[dominant_angle_idx + 1]) / 2
        
        # Xác định hướng
        if -45 <= dominant_angle <= 45:
            direction = 'right'
        elif 135 <= dominant_angle <= 180 or -180 <= dominant_angle <= -135:
            direction = 'left'
        elif 45 < dominant_angle < 135:
            direction = 'straight'
        else:
            direction = 'unknown'
        
        confidence = hist[dominant_angle_idx] / len(strong_angles)
        
        return {'direction': direction, 'confidence': confidence, 'angle': dominant_angle}
    
    def _combine_results(self, results):
        """Kết hợp kết quả từ nhiều phương pháp"""
        # Weighted voting
        direction_votes = {}
        total_confidence = 0
        
        for result in results:
            direction = result['direction']
            confidence = result['confidence']
            
            if direction != 'unknown':
                if direction not in direction_votes:
                    direction_votes[direction] = 0
                direction_votes[direction] += confidence
                total_confidence += confidence
        
        if not direction_votes:
            return {'direction': 'unknown', 'confidence': 0.0, 'angle': 0.0}
        
        # Chọn hướng có votes cao nhất
        best_direction = max(direction_votes, key=direction_votes.get)
        best_confidence = direction_votes[best_direction] / max(total_confidence, 1)
        best_angle = self._direction_to_angle(best_direction)
        
        return {
            'direction': best_direction,
            'confidence': best_confidence,
            'angle': best_angle
        }
    
    def _direction_to_angle(self, direction):
        """Chuyển hướng thành góc"""
        angle_map = {
            'right': 0,
            'straight': 90,
            'left': 180,
            'unknown': 0
        }
        return angle_map.get(direction, 0)

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Khởi tạo Computer Vision Arrow Detector
        self.arrow_detector = ArrowDetector()
        print("[EPIC] Arrow detector initialized")
        
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
        """Hybrid approach: Computer Vision + NLP"""
        if self.vintern_model is None or self.vintern_tokenizer is None:
            return ""
        try:
            # Bước 1: Computer Vision phân tích hướng mũi tên
            arrow_result = self.arrow_detector.detect_arrow_direction(sign_image)
            cv_direction = arrow_result.get('direction', 'unknown')
            cv_confidence = arrow_result.get('confidence', 0.0)
            cv_angle = arrow_result.get('angle', 0.0)
            
            print(f"[ARROW CV] Direction: {cv_direction}, Confidence: {cv_confidence:.2f}, Angle: {cv_angle:.1f}°")
            
            # Bước 2: NLP đọc text và xác nhận hướng
            pil_image = Image.fromarray(cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB))
            pixel_values = load_image_for_vintern(pil_image, input_size=448, max_num=4).to(torch.float16).to(self.device)
            generation_config = dict(max_new_tokens=150, do_sample=False, num_beams=3, repetition_penalty=1.3, pad_token_id=self.vintern_tokenizer.eos_token_id)
            
            # Prompt được tối ưu cho việc đọc text và địa điểm
            question = f"""<image>
Bạn là trợ lý GPS. Hãy quan sát biển báo chỉ dẫn và hướng dẫn người lái xe theo **các mũi tên thật sự có trên biển**.

Yêu cầu:
1. Chỉ mô tả hướng đi nếu trên biển có mũi tên thật.
2. Nếu có khoảng cách ghi trên biển, hãy thêm vào hướng dẫn. Nếu không có, chỉ cần nói hướng và địa điểm.
3. Tránh sao chép bất kỳ câu mẫu nào. Hãy tạo câu tự nhiên như đang dẫn đường, không cần tuân theo cấu trúc cố định.
4. Kiểm tra kỹ hình dạng mũi tên để xác định đúng hướng: đi thẳng, rẽ trái, rẽ phải.

Ghi nhớ:
- Không phát minh thêm hướng đi nếu không thấy mũi tên.
- Không bỏ sót khoảng cách nếu biển báo có ghi.

Hãy bắt đầu chỉ đường, mỗi hướng là một dòng riêng."""
            
            response, history = self.vintern_model.chat(self.vintern_tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            
            # Bước 3: Xử lý và kết hợp kết quả
            nlp_result = self._process_nlp_response(response)
            
            # Bước 4: Fusion hai kết quả
            final_result = self._fuse_cv_nlp_results(cv_direction, cv_confidence, nlp_result)
            
            return final_result
                
        except Exception as e:
            print(f"[ERROR] Lỗi khi xử lý hybrid OCR: {e}")
            return ""
    
    def _process_nlp_response(self, response: str) -> str:
        """Xử lý response từ NLP model"""
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
                    'nhìn thấy', 'có thể thấy', 'xuất hiện', 'tôi thấy', 'dựa vào',
                    'computer vision', 'đã phát hiện', 'xác nhận'
                ]):
                    # Làm sạch dòng
                    clean_line = line.lstrip('- ').lstrip('• ').lstrip('1234567890. ').strip()
                    if clean_line and len(clean_line) >= 2:
                        all_lines.append(clean_line)
        
        # Kết hợp các dòng bằng " | "
        if all_lines:
            result = ' | '.join(all_lines[:3])  # Lấy tối đa 3 dòng đầu tiên
            return result
        else:
            # Fallback: lấy response gốc và làm sạch
            fallback = response.replace('\n', ' ').strip()
            if len(fallback) > 150:
                fallback = fallback[:150] + "..."
            return fallback
    
    def _fuse_cv_nlp_results(self, cv_direction: str, cv_confidence: float, nlp_result: str) -> str:
        """Kết hợp kết quả Computer Vision và NLP"""
        
        # Nếu CV có confidence cao, ưu tiên CV
        if cv_confidence >= 0.7 and cv_direction != 'unknown':
            direction_map = {
                'left': 'Rẽ trái',
                'right': 'Rẽ phải', 
                'straight': 'Đi thẳng'
            }
            cv_text = direction_map.get(cv_direction, '')
            
            # Trích xuất địa điểm từ NLP nếu có
            location = self._extract_location_from_nlp(nlp_result)
            if location:
                return f"{cv_text} đến {location}"
            else:
                return f"{cv_text} (CV: {cv_confidence:.1%})"
        
        # Nếu CV confidence thấp, ưu tiên NLP
        elif cv_confidence < 0.7:
            print(f"[FUSION] CV confidence thấp ({cv_confidence:.2f}), sử dụng NLP: {nlp_result}")
            return nlp_result
        
        # Fallback
        else:
            return nlp_result if nlp_result else f"Hướng {cv_direction} (CV)"
    
    def _extract_location_from_nlp(self, nlp_text: str) -> str:
        """Trích xuất tên địa điểm từ kết quả NLP"""
        if not nlp_text:
            return ""
        
        # Tìm pattern "đến [địa điểm]"
        import re
        patterns = [
            r'đến\s+([^|]+)',
            r'về\s+([^|]+)', 
            r'tới\s+([^|]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, nlp_text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Loại bỏ khoảng cách nếu có
                location = re.sub(r'\d+[km\s]*', '', location).strip()
                return location
        
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
            detection = {
                'object_id': object_id,
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': class_idx_smooth,
                'class_label': class_label,
                'class_label_vi': class_label_vi,
                'ocr_text': None
            }
            # Chỉ sử dụng NLP cho biển báo "sus" (TEXT SIGN) khi YOLO detect được
            if class_label == "sus":  # Chỉ khi YOLO nhận diện được class "sus"
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
            # Lấy mã biển báo từ class_names thay vì dùng class_id_code hardcode
            class_code = det['class_label'] if 'class_label' in det else f"Class_{det['class_id']}"
            description_no_diacritics = det['class_label_vi'] if 'class_label_vi' in det else class_code
            confidence = det['confidence']
            
            # Format: Mã biển | Description_tieng_viet_khong_dai | Độ tin cậy
            label = f"{class_code} | {description_no_diacritics} | {confidence:.1%}"
            draw.text((x1, y1 - 25), label, font=font, fill=(0,255,0))
            
            if det.get('ocr_text') and det['ocr_text'] != '...':
                ocr_label = f"OCR: {det['ocr_text']}"
                draw.text((x1, y2 + 5), ocr_label, font=font, fill=(255,255,255))
                print(f"{class_code} | {description_no_diacritics} | {det['ocr_text']}")
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