import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from typing import List, Tuple, Optional
import os
from pathlib import Path
from skimage import exposure
from scipy.ndimage import gaussian_filter
import json
import unicodedata

class ImageEnhancer:
    """Class for enhancing image quality for better detection"""
    
    @staticmethod
    def enhance_image(image: np.ndarray, enhancement_level: float = 1.2) -> np.ndarray:
        """
        Enhance image quality for better detection
        
        Args:
            image: Input image as numpy array
            enhancement_level: Enhancement factor (1.0 = no change)
            
        Returns:
            Enhanced image
        """
        # Convert to PIL Image for enhancement
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = contrast_enhancer.enhance(enhancement_level)
        
        # Enhance sharpness
        sharpness_enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = sharpness_enhancer.enhance(enhancement_level)
        
        # Enhance brightness slightly
        brightness_enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = brightness_enhancer.enhance(1.1)
        
        # Convert back to numpy array
        enhanced_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return enhanced_image
    
    @staticmethod
    def denoise_image(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Remove noise from image using multiple techniques
        
        Args:
            image: Input image
            strength: Denoising strength (0.0 to 1.0)
            
        Returns:
            Denoised image
        """
        # Apply bilateral filter to preserve edges while removing noise
        d = int(strength * 15)  # Diameter of pixel neighborhood
        sigma_color = strength * 75
        sigma_space = strength * 75
        denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        # Apply additional non-local means denoising for stronger effect
        if strength > 0.7:
            h = 10  # Filter strength
            denoised = cv2.fastNlMeansDenoisingColored(denoised, None, h, h, 7, 21)
        
        return denoised
    
    @staticmethod
    def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Sharpen image using multiple techniques
        
        Args:
            image: Input image
            strength: Sharpening strength multiplier
            
        Returns:
            Sharpened image
        """
        # Create sharpening kernel
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]]) * strength
        
        # Apply kernel
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Apply unsharp mask for additional sharpening
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(sharpened, 1.5, gaussian, -0.5, 0)
        
        return sharpened
    
    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Adjust gamma correction with additional contrast enhancement
        
        Args:
            image: Input image
            gamma: Gamma value (1.0 = no change)
            
        Returns:
            Gamma-corrected image
        """
        # Apply gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(image, table)
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    @staticmethod
    def enhance_low_light(image: np.ndarray) -> np.ndarray:
        """
        Enhance low-light images
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to lightness channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Adjust gamma for better visibility
        enhanced = ImageEnhancer.adjust_gamma(enhanced, 1.2)
        
        return enhanced
    
    @staticmethod
    def reduce_blur(image: np.ndarray) -> np.ndarray:
        """
        Reduce blur in images using deconvolution
        
        Args:
            image: Input image
            
        Returns:
            Deblurred image
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Wiener deconvolution
        psf = np.ones((5, 5)) / 25  # Point spread function
        deblurred = np.zeros_like(image)
        
        # Process each channel
        for i in range(3):
            channel = image[:,:,i]
            deblurred_channel = exposure.wiener(channel, psf)
            deblurred[:,:,i] = (deblurred_channel * 255).astype(np.uint8)
        
        return deblurred

class DataAugmentation:
    """Class for data augmentation during training"""
    
    def __init__(self, image_size: int = 640):
        self.image_size = image_size
        self.transform = A.Compose([
            A.Resize(height=image_size, width=image_size, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        ])
    
    def augment(self, image: np.ndarray, bboxes: List[List[float]] = None) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Apply augmentation to image and bounding boxes
        
        Args:
            image: Input image
            bboxes: Bounding boxes in YOLO format [class, x_center, y_center, width, height]
            
        Returns:
            Augmented image and bounding boxes
        """
        if bboxes is None:
            bboxes = []
        
        # Convert YOLO format to Albumentations format
        albumentations_bboxes = []
        for bbox in bboxes:
            if len(bbox) >= 4:
                class_id, x_center, y_center, width, height = bbox[:4]
                x_min = (x_center - width/2) * image.shape[1]
                y_min = (y_center - height/2) * image.shape[0]
                x_max = (x_center + width/2) * image.shape[1]
                y_max = (y_center + height/2) * image.shape[0]
                albumentations_bboxes.append([x_min, y_min, x_max, y_max, class_id])
        
        # Apply augmentation
        if albumentations_bboxes:
            transformed = self.transform(image=image, bboxes=albumentations_bboxes)
            augmented_image = transformed['image']
            augmented_bboxes = transformed['bboxes']
            
            # Convert back to YOLO format
            yolo_bboxes = []
            for bbox in augmented_bboxes:
                x_min, y_min, x_max, y_max, class_id = bbox
                x_center = (x_min + x_max) / 2 / augmented_image.shape[1]
                y_center = (y_min + y_max) / 2 / augmented_image.shape[0]
                width = (x_max - x_min) / augmented_image.shape[1]
                height = (y_max - y_min) / augmented_image.shape[0]
                yolo_bboxes.append([class_id, x_center, y_center, width, height])
        else:
            transformed = self.transform(image=image)
            augmented_image = transformed['image']
            yolo_bboxes = []
        
        return augmented_image, yolo_bboxes

class VisualizationUtils:
    """Utilities for visualization"""
    
    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[dict], 
                       class_names = None, 
                       confidence_threshold: float = 0.25) -> np.ndarray:
        """
        Draw detection results on image with improved visualization
        Args:
            image: Input image
            detections: List of detection dictionaries
            class_names: Dict of class names (key: mã nhãn, value: tên không dấu)
            confidence_threshold: Minimum confidence to display
        Returns:
            Image with detections drawn
        """
        result_image = image.copy()
        color_map = {
            'speed_limit': (0, 255, 0),
            'warning': (0, 165, 255),
            'prohibition': (0, 0, 255),
            'mandatory': (255, 0, 0),
            'other': (128, 128, 128)
        }
        for detection in detections:
            if detection['confidence'] < confidence_threshold:
                continue
            bbox = detection['bbox']
            class_id = detection['class_id']
            confidence = detection['confidence']
            x1, y1, x2, y2 = map(int, bbox)
            # Lấy tên lớp từ dict
            if isinstance(class_names, dict):
                class_name = class_names.get(class_id, f"Class {class_id}")
            elif isinstance(class_names, list) and isinstance(class_id, int) and class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"Class {class_id}"
            color = color_map['other']
            for category, c in color_map.items():
                if category in class_name.lower():
                    color = c
                    break
            thickness = max(1, int(3 * confidence))
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
            cv2.rectangle(result_image, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1),
                         color, -1)
            cv2.putText(result_image, label, 
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                       thickness)
        return result_image
    
    @staticmethod
    def save_detection_result(image: np.ndarray, output_path: str, 
                            filename: str, detections: List[dict] = None, class_names=None, class_names_vi=None):
        """
        Save detection result with metadata (JSON)
        Args:
            image: Image with drawn detections
            output_path: Directory to save results
            filename: Output filename
            detections: List of detection dictionaries
            class_names: List hoặc Dict ánh xạ số -> mã nhãn
            class_names_vi: Dict ánh xạ mã nhãn -> mô tả tiếng Việt
        """
        images_dir = os.path.join(output_path, 'images')
        json_dir = os.path.join(output_path, 'json')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        image_path = os.path.join(images_dir, filename)
        cv2.imwrite(image_path, image)
        if detections:
            base_name = os.path.splitext(filename)[0]
            metadata_path = os.path.join(json_dir, f"{base_name}_detections.json")
            output_json = []
            for det in detections:
                class_id = det.get('class_id')
                class_label = det.get('class_label')
                class_label_vi = det.get('class_label_vi')
                if class_label is None:
                    if isinstance(class_names, list) and int(class_id) < len(class_names):
                        class_label = class_names[int(class_id)]
                    else:
                        class_label = str(class_id)
                if class_label_vi is None:
                    class_label_vi = class_label
                output_json.append({
                    "class_id": class_id,
                    "class_label": class_label,
                    "class_label_vi": class_label_vi,
                    "confidence": det['confidence'],
                    "bbox": det['bbox']
                })
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(output_json, f, ensure_ascii=False, indent=2)

class FileUtils:
    """Utilities for file operations"""
    
    @staticmethod
    def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
        """
        Get all image files in a directory
        
        Args:
            directory: Directory to search
            extensions: List of valid extensions (default: ['.jpg', '.jpeg', '.png'])
            
        Returns:
            List of image file paths
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            
        image_files = []
        for ext in extensions:
            image_files.extend(Path(directory).glob(f"*{ext}"))
            image_files.extend(Path(directory).glob(f"*{ext.upper()}"))
        
        return [str(f) for f in sorted(image_files)]
    
    @staticmethod
    def create_dataset_yaml(dataset_path: str, output_path: str = "dataset.yaml"):
        """
        Create dataset.yaml file
        
        Args:
            dataset_path: Path to dataset
            output_path: Output yaml file path
        """
        yaml_content = f"""# Dataset configuration
path: ./dataset
train: images/train
val: images/val

# Number of classes
nc: 43  # Adjust based on your traffic sign classes

# Class names (example - adjust based on your dataset)
names:
Đường người đi bộ cắt ngang
Đường giao nhau (ngã ba bên phải)
Cấm đi ngược chiều
Phải đi vòng sang bên phải
Giao nhau với đường đồng cấp
Giao nhau với đường không ưu tiên
Chỗ ngoặt nguy hiểm vòng bên trái
Cấm rẽ trái
Bến xe buýt
Nơi giao nhau chạy theo vòng xuyến
Cấm dừng và đỗ xe
Chỗ quay xe
Biển gộp làn đường theo phương tiện
Đi chậm
Cấm xe tải
Đường bị thu hẹp về phía phải
Giới hạn chiều cao
Cấm quay đầu
Cấm ô tô khách và ô tô tải
Cấm rẽ phải và quay đầu
Cấm ô tô
Đường bị thu hẹp về phía trái
Gồ giảm tốc phía trước
Cấm xe hai và ba bánh
Kiểm tra
Chỉ dành cho xe máy*
Chướng ngoại vật phía trước
Trẻ em
Xe tải và xe công*
Cấm mô tô và xe máy
Chỉ dành cho xe tải*
Đường có camera giám sát
Cấm rẽ phải
Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải
Cấm xe sơ-mi rơ-moóc
Cấm rẽ trái và phải
Cấm đi thẳng và rẽ phải
Đường giao nhau (ngã ba bên trái)
Giới hạn tốc độ (50km/h)
Giới hạn tốc độ (60km/h)
Giới hạn tốc độ (80km/h)
Giới hạn tốc độ (40km/h)
Các xe chỉ được rẽ trái
Chiều cao tĩnh không thực tế
Nguy hiểm khác
Đường một chiều
Cấm đỗ xe
Cấm ô tô quay đầu xe (được rẽ trái)
Giao nhau với đường sắt có rào chắn
Cấm rẽ trái và quay đầu xe
Chỗ ngoặt nguy hiểm vòng bên phải
Chú ý chướng ngại vật – vòng tránh sang bên phải
"""
        
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created dataset.yaml at: {output_path}")

def to_ascii_label(s):
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    s = s.replace(' ', '_')
    return s 

IMAGE_DIR = os.path.abspath("./dataset/images")
LABEL_DIR = os.path.abspath("./dataset/labels") 