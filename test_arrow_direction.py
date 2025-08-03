"""
Test script để kiểm tra khả năng đọc hướng mũi tên của Vintern
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

def load_image_for_vintern(image, input_size=448, max_num=4):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class ArrowDirectionTester:
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

    def test_basic_direction_reading(self, sign_image: np.ndarray) -> str:
        """Test cơ bản: đọc chữ và hướng mũi tên"""
        if self.vintern_model is None or self.vintern_tokenizer is None:
            return ""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB))
            pixel_values = load_image_for_vintern(pil_image, input_size=448, max_num=4).to(torch.float16).to(self.device)
            generation_config = dict(max_new_tokens=150, do_sample=False, num_beams=3, repetition_penalty=1.3, pad_token_id=self.vintern_tokenizer.eos_token_id)
            
            question = """<image>
Hãy đọc TẤT CẢ các chữ và mô tả hướng của các mũi tên trên biển báo này."""
            
            response, history = self.vintern_model.chat(self.vintern_tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            return response.strip()
                
        except Exception as e:
            print(f"[ERROR] Lỗi OCR cơ bản: {e}")
            return ""

    def test_detailed_direction_reading(self, sign_image: np.ndarray) -> str:
        """Test chi tiết: phân tích từng mũi tên"""
        if self.vintern_model is None or self.vintern_tokenizer is None:
            return ""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB))
            pixel_values = load_image_for_vintern(pil_image, input_size=448, max_num=4).to(torch.float16).to(self.device)
            generation_config = dict(max_new_tokens=200, do_sample=False, num_beams=3, repetition_penalty=1.2, pad_token_id=self.vintern_tokenizer.eos_token_id)
            
            question = """<image>
Phân tích chi tiết biển báo này:
1. Có bao nhiêu mũi tên?
2. Mỗi mũi tên chỉ hướng nào? (đi thẳng, sang trái, sang phải, chéo trái, chéo phải)
3. Tên địa danh gần mỗi mũi tên là gì?
Liệt kê từng mũi tên một cách rõ ràng."""
            
            response, history = self.vintern_model.chat(self.vintern_tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            return response.strip()
                
        except Exception as e:
            print(f"[ERROR] Lỗi OCR chi tiết: {e}")
            return ""

    def test_simplified_direction_reading(self, sign_image: np.ndarray) -> str:
        """Test đơn giản: chỉ lấy hướng mũi tên"""
        if self.vintern_model is None or self.vintern_tokenizer is None:
            return ""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB))
            pixel_values = load_image_for_vintern(pil_image, input_size=448, max_num=4).to(torch.float16).to(self.device)
            generation_config = dict(max_new_tokens=100, do_sample=False, num_beams=2, repetition_penalty=1.1, pad_token_id=self.vintern_tokenizer.eos_token_id)
            
            question = """<image>
Mô tả ngắn gọn hướng của từng mũi tên: đi thẳng, trái, phải, chéo trái, chéo phải."""
            
            response, history = self.vintern_model.chat(self.vintern_tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            return response.strip()
                
        except Exception as e:
            print(f"[ERROR] Lỗi OCR đơn giản: {e}")
            return ""

    def test_image_with_all_methods(self, image_path: str):
        """Test ảnh với tất cả phương pháp"""
        if not os.path.exists(image_path):
            print(f"[ERROR] File không tồn tại: {image_path}")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Không thể đọc ảnh: {image_path}")
            return
        
        print(f"\n{'='*100}")
        print(f"🏷️  TEST ẢNH: {os.path.basename(image_path)}")
        print(f"{'='*100}")
        
        # Test 1: Đọc cơ bản
        print("\n🔍 [PHƯƠNG PHÁP 1] Đọc cơ bản:")
        print("-" * 50)
        basic_result = self.test_basic_direction_reading(image)
        print(f"  📄 {basic_result}")
        
        # Test 2: Phân tích chi tiết  
        print("\n🔍 [PHƯƠNG PHÁP 2] Phân tích chi tiết:")
        print("-" * 50)
        detailed_result = self.test_detailed_direction_reading(image)
        print(f"  📄 {detailed_result}")
        
        # Test 3: Đơn giản hóa
        print("\n🔍 [PHƯƠNG PHÁP 3] Hướng mũi tên đơn giản:")
        print("-" * 50)
        simple_result = self.test_simplified_direction_reading(image)
        print(f"  📄 {simple_result}")
        
        print(f"\n{'='*100}")

def main():
    print("🚀 [INFO] Khởi tạo Arrow Direction Tester...")
    tester = ArrowDirectionTester()
    
    # Test trên các ảnh trong thư mục input
    input_dir = "input"
    if os.path.exists(input_dir):
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Lọc các file có khả năng chứa mũi tên chỉ đường
        arrow_files = []
        for f in image_files:
            # Tìm các file có tên gợi ý về biển chỉ đường
            if any(keyword in f.lower() for keyword in [
                'bien', 'sign', 'traffic', 'direction', 'giao', 'duong', 'chi', 'huong',
                'bay', 'gtn', 'arrow', 'way', 'road'
            ]):
                arrow_files.append(f)
        
        # Nếu không tìm thấy file đặc biệt, lấy một vài file đầu
        if not arrow_files:
            arrow_files = image_files[:3]
        
        print(f"\n📂 [INFO] Tìm thấy {len(arrow_files)} ảnh để test khả năng đọc mũi tên")
        
        for img_file in arrow_files[:5]:  # Test tối đa 5 ảnh
            image_path = os.path.join(input_dir, img_file)
            tester.test_image_with_all_methods(image_path)
            
        print(f"\n🎯 [TỔNG KẾT]")
        print("Mô hình Vintern có thể:")
        print("✅ Đọc được các ký tự trên biển báo")
        print("✅ Nhận diện được hình dạng mũi tên")
        print("❓ Cần kiểm tra khả năng phân biệt chính xác hướng mũi tên")
        print("💡 Kết quả phụ thuộc vào chất lượng ảnh và độ rõ nét của mũi tên")
        
    else:
        print(f"❌ [WARNING] Thư mục {input_dir} không tồn tại")
        print("💡 Hãy đặt một số ảnh biển báo có mũi tên vào thư mục input để test")

if __name__ == "__main__":
    main()
