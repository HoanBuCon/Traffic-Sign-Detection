import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import DataAugmentation, ImageEnhancer
from config import Config
import os
from pathlib import Path

def demo_augmentation(image_path: str, output_dir: str = "augmentation_demo"):
    """
    Demo các kỹ thuật augmentation
    
    Args:
        image_path: Đường dẫn đến ảnh gốc
        output_dir: Thư mục lưu ảnh demo
    """
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
    
    # Chuyển BGR sang RGB để hiển thị
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Khởi tạo các class
    data_augmenter = DataAugmentation(image_size=640)
    image_enhancer = ImageEnhancer()
    
    # Tạo danh sách ảnh để hiển thị
    images = []
    titles = []
    
    # Ảnh gốc
    images.append(image_rgb)
    titles.append("Original")
    
    # Enhanced image
    enhanced = image_enhancer.enhance_image(image)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    images.append(enhanced_rgb)
    titles.append("Enhanced")
    
    # Denoised image
    denoised = image_enhancer.denoise_image(image)
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    images.append(denoised_rgb)
    titles.append("Denoised")
    
    # Sharpened image
    sharpened = image_enhancer.sharpen_image(image)
    sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    images.append(sharpened_rgb)
    titles.append("Sharpened")
    
    # Low-light enhanced
    low_light_enhanced = image_enhancer.enhance_low_light(image)
    low_light_rgb = cv2.cvtColor(low_light_enhanced, cv2.COLOR_BGR2RGB)
    images.append(low_light_rgb)
    titles.append("Low-light Enhanced")
    
    # Gamma corrected
    gamma_corrected = image_enhancer.adjust_gamma(image, gamma=1.2)
    gamma_rgb = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)
    images.append(gamma_rgb)
    titles.append("Gamma Corrected (1.2)")
    
    # Tạo các augmented samples
    for i in range(6):
        try:
            # Augment với bbox giả (nếu có)
            aug_image, aug_bboxes, aug_labels = data_augmenter.augment(
                image, [], [], is_training=True
            )
            
            # Chuyển về RGB
            aug_rgb = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
            images.append(aug_rgb)
            titles.append(f"Augmented {i+1}")
            
            # Lưu ảnh riêng lẻ
            cv2.imwrite(os.path.join(output_dir, f"augmented_{i+1}.jpg"), aug_image)
            
        except Exception as e:
            print(f"Lỗi augmentation {i+1}: {e}")
            continue
    
    # Hiển thị grid
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.ravel()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=12)
            axes[i].axis('off')
    
    # Ẩn các subplot không sử dụng
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "augmentation_demo_grid.jpg"), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Demo augmentation đã được lưu tại: {output_dir}")

def demo_batch_augmentation(image_dir: str, output_dir: str = "batch_augmentation_demo"):
    """
    Demo augmentation cho nhiều ảnh
    
    Args:
        image_dir: Thư mục chứa ảnh
        output_dir: Thư mục lưu kết quả
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Tìm tất cả ảnh trong thư mục
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"Không tìm thấy ảnh nào trong {image_dir}")
        return
    
    print(f"Tìm thấy {len(image_files)} ảnh")
    
    # Khởi tạo augmenter
    data_augmenter = DataAugmentation(image_size=640)
    image_enhancer = ImageEnhancer()
    
    # Xử lý từng ảnh
    for i, image_file in enumerate(image_files[:5]):  # Chỉ xử lý 5 ảnh đầu
        print(f"Đang xử lý ảnh {i+1}/{min(5, len(image_files))}: {image_file.name}")
        
        # Đọc ảnh
        image = cv2.imread(str(image_file))
        if image is None:
            continue
        
        # Tạo thư mục cho ảnh này
        image_output_dir = os.path.join(output_dir, image_file.stem)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Lưu ảnh gốc
        cv2.imwrite(os.path.join(image_output_dir, "original.jpg"), image)
        
        # Enhanced
        enhanced = image_enhancer.enhance_image(image)
        cv2.imwrite(os.path.join(image_output_dir, "enhanced.jpg"), enhanced)
        
        # Tạo 3 augmented samples
        for j in range(3):
            try:
                aug_image, _, _ = data_augmenter.augment(image, [], [], is_training=True)
                cv2.imwrite(os.path.join(image_output_dir, f"augmented_{j+1}.jpg"), aug_image)
            except Exception as e:
                print(f"Lỗi augmentation {j+1} cho {image_file.name}: {e}")
                continue
    
    print(f"Batch augmentation demo đã được lưu tại: {output_dir}")

def main():
    """Main function"""
    print("=== Demo Data Augmentation ===")
    
    # Demo với ảnh mẫu
    sample_image = "input/sample.jpg"  # Thay đổi đường dẫn nếu cần
    
    if os.path.exists(sample_image):
        print(f"Demo với ảnh: {sample_image}")
        demo_augmentation(sample_image)
    else:
        print(f"Không tìm thấy ảnh mẫu: {sample_image}")
        print("Vui lòng đặt ảnh mẫu vào thư mục input/")
    
    # Demo batch augmentation
    input_dir = "input"
    if os.path.exists(input_dir):
        print(f"\nDemo batch augmentation với thư mục: {input_dir}")
        demo_batch_augmentation(input_dir)
    else:
        print(f"Không tìm thấy thư mục input: {input_dir}")

if __name__ == "__main__":
    main() 