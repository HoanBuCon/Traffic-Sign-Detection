"""
Test script để kiểm tra khả năng OCR cải thiện của Vintern
"""
import cv2
import os
import sys
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T

# Import các hàm cần thiết
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

def load_image_for_vintern(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class ImprovedOCRTester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_nlp_model()
    
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

    def test_old_ocr(self, sign_image: np.ndarray) -> str:
        """Test OCR với prompt cũ"""
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
            print(f"[ERROR] Lỗi OCR cũ: {e}")
            return ""

    def test_improved_ocr(self, sign_image: np.ndarray) -> str:
        """Test OCR với prompt cải thiện"""
        if self.vintern_model is None or self.vintern_tokenizer is None:
            return ""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB))
            pixel_values = load_image_for_vintern(pil_image, input_size=448, max_num=6).to(torch.float16).to(self.device)
            generation_config = dict(max_new_tokens=256, do_sample=False, num_beams=5, repetition_penalty=1.3, pad_token_id=self.vintern_tokenizer.eos_token_id)
            
            # Prompt cải thiện để đọc được nhiều địa danh
            question = """<image>
Hãy đọc TẤT CẢ các chữ và địa danh trên biển báo này, bao gồm:
- Tất cả tên địa danh, tên đường, tên quận/huyện
- Tất cả các mũi tên và hướng chỉ dẫn
- Số km nếu có
Liệt kê từng dòng một cách đầy đủ, không bỏ sót."""
            
            response, history = self.vintern_model.chat(self.vintern_tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            
            # Xử lý response để giữ lại nhiều dòng thông tin
            response = response.strip()
            response = response.replace('**', '').replace('`', '').replace('*', '')
            
            # Loại bỏ các dòng không cần thiết nhưng giữ lại thông tin địa danh
            lines = []
            for line in response.split('\n'):
                line = line.strip()
                if line and not line.lower().startswith(('theo như', 'dựa vào', 'từ hình ảnh', 'tôi thấy', 'biển báo này')):
                    lines.append(line)
            
            # Kết hợp các dòng bằng dấu " | " để hiển thị đầy đủ
            if lines:
                result = ' | '.join(lines[:4])  # Giới hạn tối đa 4 phần để tránh quá dài
                return result
            else:
                return response.split('\n')[0] if response else ""
                
        except Exception as e:
            print(f"[ERROR] Lỗi OCR cải thiện: {e}")
            return ""

    def test_image(self, image_path: str):
        """Test cả 2 phương pháp OCR trên 1 ảnh"""
        if not os.path.exists(image_path):
            print(f"[ERROR] File không tồn tại: {image_path}")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Không thể đọc ảnh: {image_path}")
            return
        
        print(f"\n[TEST] Đang test ảnh: {os.path.basename(image_path)}")
        print("="*60)
        
        # Test OCR cũ
        print("\n[OCR CŨ] Kết quả:")
        old_result = self.test_old_ocr(image)
        print(f"  {old_result}")
        
        # Test OCR cải thiện
        print("\n[OCR CẢI THIỆN] Kết quả:")
        improved_result = self.test_improved_ocr(image)
        print(f"  {improved_result}")
        
        print("\n" + "="*60)

def main():
    print("[INFO] Khởi tạo OCR Tester...")
    tester = ImprovedOCRTester()
    
    # Test trên các ảnh trong thư mục input
    input_dir = "input"
    if os.path.exists(input_dir):
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            print(f"\n[INFO] Tìm thấy {len(image_files)} ảnh để test")
            for img_file in image_files[:3]:  # Test 3 ảnh đầu tiên
                image_path = os.path.join(input_dir, img_file)
                tester.test_image(image_path)
        else:
            print("[WARNING] Không tìm thấy ảnh nào trong thư mục input")
    else:
        print("[WARNING] Thư mục input không tồn tại")

if __name__ == "__main__":
    main()
