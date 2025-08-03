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

# Thêm thư mục gốc của project vào sys.path để xử lý absolute imports
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Vintern custom pipeline ---
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

def load_image_for_vintern(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# --- Xử lý tiếng Việt không dấu ---
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

class RealTimeTrafficSignDetectorNLP:
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
        self.tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
        self.label_buffers = {}
        self.buffer_size = 10
        self._load_nlp_model()
        self.ocr_cache = {}  # key: object_id, value: {'ocr_text': ..., 'bbox': ..., 'last_update': ...}
        self.frame_count = 0

    def _load_nlp_model(self):
        print("[INFO] Đang tải mô hình Vintern NLP...")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

    def _get_text_from_sign(self, sign_image: np.ndarray) -> str:
        if self.vintern_model is None or self.vintern_tokenizer is None:
            return ""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB))
            pixel_values = load_image_for_vintern(pil_image, input_size=448, max_num=4).to(torch.float16).to(self.device)
            generation_config = dict(max_new_tokens=200, do_sample=False, num_beams=4, repetition_penalty=1.2, pad_token_id=self.vintern_tokenizer.eos_token_id)
            
            # Prompt tối ưu để đọc được nhiều địa danh nhưng ngắn gọn hơn
            question = """<image>
Hãy đọc biển báo chỉ dẫn này như một người dẫn đường. Chỉ mô tả hướng đi khi trên biển có mũi tên chỉ dẫn rõ ràng.

Cách xác định hướng mũi tên:
• Nếu mũi tên **thẳng đứng**, không lệch → Hướng: Đi thẳng
• Nếu mũi tên **chỉ sang trái rõ ràng** → Hướng: Rẽ trái
• Nếu mũi tên **chỉ sang phải rõ ràng** → Hướng: Rẽ phải
• Nếu mũi tên **cong hoặc vòng** → mô tả là: "Rẽ vòng [trái/phải] đến [địa điểm]"

Yêu cầu:
- Chỉ mô tả hướng có mũi tên thật.
- Khoảng cách ghi nếu có, nếu không thì bỏ qua.
- Trả lời ngắn gọn, mỗi hướng một dòng.

Ví dụ:
- Đi thẳng 2km đến Tân Cương
- Rẽ trái 16km đến Đại Từ
- Rẽ phải 91km đến Bắc Kạn"""
            
            response, history = self.vintern_model.chat(self.vintern_tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            
            # Xử lý response để giữ lại nhiều dòng thông tin
            response = response.strip()
            response = response.replace('**', '').replace('`', '').replace('*', '')
            
            # Loại bỏ các dòng giải thích không cần thiết
            lines = []
            for line in response.split('\n'):
                line = line.strip()
                # Loại bỏ những dòng giải thích dài dòng
                if line and len(line) < 100 and not any(skip_word in line.lower() for skip_word in 
                    ['theo như', 'dựa vào', 'từ hình ảnh', 'tôi thấy', 'biển báo này', 'hình ảnh không', 'không cung cấp']):
                    lines.append(line)
            
            # Kết hợp các dòng bằng dấu " | " để hiển thị đầy đủ
            if lines:
                result = ' | '.join(lines[:3])  # Giới hạn tối đa 3 phần để tránh quá dài
                return result
            else:
                # Fallback về phương pháp cũ nếu không có kết quả tốt
                first_line = response.split('\n')[0] if response else ""
                return first_line[:100] if len(first_line) > 100 else first_line
                
        except Exception as e:
            print(f"[ERROR] Lỗi khi xử lý OCR Vintern: {e}")
            return ""

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
        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            iou_val = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
            return iou_val
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
                bbox = [crop_x1, crop_y1, crop_x2, crop_y2]
                need_ocr = False
                if object_id not in self.ocr_cache:
                    need_ocr = True
                else:
                    cache = self.ocr_cache[object_id]
                    if iou(bbox, cache['bbox']) < 0.4 or (self.frame_count - cache['last_update'] > 20):
                        need_ocr = True
                if need_ocr and crop_y2 > crop_y1 and crop_x2 > crop_x1:
                    sign_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    ocr_text = self._get_text_from_sign(sign_crop)
                    if ocr_text:
                        detection['ocr_text'] = ocr_text
                        self.ocr_cache[object_id] = {'ocr_text': ocr_text, 'bbox': bbox, 'last_update': self.frame_count}
                elif object_id in self.ocr_cache:
                    detection['ocr_text'] = self.ocr_cache[object_id]['ocr_text']
            detections.append(detection)
        return detections

    def draw_and_show(self, frame: np.ndarray, detections):
        import os
        from PIL import ImageFont, ImageDraw, Image
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
            
            if det.get('ocr_text'):
                ocr_label = f"OCR: {det['ocr_text']}"
                draw.text((x1, y2 + 5), ocr_label, font=font, fill=(255,255,255))
                print(f"{class_code} | {description_no_diacritics} | {det['ocr_text']}")
        # Chuyển lại sang cv2 để hiển thị
        frame_show = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow('Traffic Sign Detection - SORT Smoothing', frame_show)

# --- Copy các hàm bbox_iou, main, ... từ real_time_predict_smooth_advanced.py ---
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

def main():
    detector = RealTimeTrafficSignDetectorNLP()
    detector.run_webcam(save_video=True)

if __name__ == "__main__":
    main() 