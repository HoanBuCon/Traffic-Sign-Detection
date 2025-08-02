import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Optional
import os
import yaml
from PIL import Image, ImageEnhance, ImageFilter
import random
import albumentations as A

class ImageEnhancer:
    """Class for enhancing image quality"""
    
    @staticmethod
    def enhance_image(image: np.ndarray, enhancement_level: float = 1.2) -> np.ndarray:
        """
        Enhance image quality using multiple techniques
        
        Args:
            image: Input image as numpy array
            enhancement_level: Enhancement strength (1.0 = no change)
            
        Returns:
            Enhanced image as numpy array
        """
        # Convert to PIL Image for enhancement
        pil_image = Image.fromarray(image)
        
        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = contrast_enhancer.enhance(enhancement_level)
        
        # Enhance brightness
        brightness_enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = brightness_enhancer.enhance(enhancement_level)
        
        # Enhance sharpness
        sharpness_enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = sharpness_enhancer.enhance(enhancement_level)
        
        # Convert back to numpy array
        return np.array(pil_image)
    
    @staticmethod
    def denoise_image(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Remove noise from image using bilateral filter
        
        Args:
            image: Input image
            strength: Denoising strength (0.0 to 1.0)
            
        Returns:
            Denoised image
        """
        # Convert strength to filter parameters
        d = int(15 * strength)  # Diameter of pixel neighborhood
        sigma_color = 75 * strength  # Filter sigma in the color space
        sigma_space = 75 * strength  # Filter sigma in the coordinate space
        
        # Apply bilateral filter
        denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return denoised
    
    @staticmethod
    def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Sharpen image using unsharp mask technique
        
        Args:
            image: Input image
            strength: Sharpening strength (0.0 to 2.0)
            
        Returns:
            Sharpened image
        """
        # Create unsharp mask
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
        
        # Ensure pixel values are in valid range
        unsharp_mask = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
        return unsharp_mask
    
    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Adjust gamma correction of image
        
        Args:
            image: Input image
            gamma: Gamma value (0.1 to 3.0)
            
        Returns:
            Gamma-corrected image
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction
        return cv2.LUT(image, table)
    
    @staticmethod
    def enhance_low_light(image: np.ndarray) -> np.ndarray:
        """
        Enhance low-light images using CLAHE
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    
    @staticmethod
    def reduce_blur(image: np.ndarray) -> np.ndarray:
        """
        Reduce blur using Wiener filter
        
        Args:
            image: Input image
            
        Returns:
            Deblurred image
        """
        # Convert to grayscale for deblurring
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Wiener filter
        kernel_size = 5
        noise_var = 0.01
        deblurred = cv2.filter2D(gray, -1, np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size))
        
        # Convert back to RGB if original was RGB
        if len(image.shape) == 3:
            deblurred = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2RGB)
        
        return deblurred

class DataAugmentation:
    """Class for advanced data augmentation using Albumentations"""
    
    def __init__(self, image_size: int = 640):
        """
        Initialize data augmentation pipeline
        
        Args:
            image_size: Target image size for resizing
        """
        # Create comprehensive augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            # Geometric transformations
            A.OneOf([
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=30,
                    interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                    value=0, p=1.0
                ),
                A.Affine(
                    scale=(0.8, 1.2), rotate=(-30, 30), translate_percent=(-0.1, 0.1),
                    shear=(-10, 10), interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0
                ),
            ], p=0.8),
            
            # Color and brightness augmentations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3,
                    brightness_by_max=True, p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
            ], p=0.8),
            
            # Noise and blur augmentations
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.4),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            # Weather and lighting effects
            A.OneOf([
                A.RandomRain(
                    slant_lower=-10, slant_upper=10,
                    drop_length=20, drop_width=1, drop_color=(200, 200, 200),
                    blur_value=3, brightness_coefficient=0.7,
                    rain_type="drizzle", p=1.0
                ),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1,
                    num_flare_circles_lower=6, num_flare_circles_upper=10,
                    src_radius=400, src_color=(255, 255, 255), p=1.0
                ),
            ], p=0.2),
            
            # Occlusion and cutout
            A.OneOf([
                A.CoarseDropout(
                    max_holes=8, max_height=32, max_width=32,
                    min_holes=1, min_height=8, min_width=8,
                    fill_value=0, p=1.0
                ),
                A.GridDropout(
                    ratio=0.1, unit_size_min=32, unit_size_max=128,
                    holes_number_x=4, holes_number_y=4,
                    shift_x=0, shift_y=0, random_offset=True,
                    fill_value=0, p=1.0
                ),
            ], p=0.3),
            
            # Elastic and optical distortions
            A.OneOf([
                A.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=50,
                    interpolation=1, border_mode=4, value=None,
                    mask_value=None, always_apply=False, approximate=False, p=1.0
                ),
                A.OpticalDistortion(
                    distort_limit=0.2, shift_limit=0.15,
                    interpolation=1, border_mode=4, value=None,
                    mask_value=None, always_apply=False, p=1.0
                ),
            ], p=0.2),
            
            # Resize to target size
            A.Resize(height=image_size, width=image_size, p=1.0),
            
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Create simpler augmentation for validation
        self.validation_pipeline = A.Compose([
            A.Resize(height=image_size, width=image_size, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def augment(self, image: np.ndarray, bboxes: List[List[float]] = None, 
                class_labels: List[int] = None, is_training: bool = True) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Apply data augmentation to image and bounding boxes
        
        Args:
            image: Input image
            bboxes: List of bounding boxes in YOLO format [x_center, y_center, width, height]
            class_labels: List of class labels
            is_training: Whether this is for training (True) or validation (False)
            
        Returns:
            Tuple of (augmented_image, augmented_bboxes, augmented_class_labels)
        """
        # Prepare data for augmentation
        if bboxes is None:
            bboxes = []
        if class_labels is None:
            class_labels = []
        
        # Choose pipeline based on training/validation
        pipeline = self.augmentation_pipeline if is_training else self.validation_pipeline
        
        # Apply augmentation
        augmented = pipeline(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        return (
            augmented['image'],
            augmented['bboxes'],
            augmented['class_labels']
        )
    
    def augment_batch(self, images: List[np.ndarray], bboxes_list: List[List[List[float]]] = None,
                     class_labels_list: List[List[int]] = None, is_training: bool = True) -> Tuple[List[np.ndarray], List[List[List[float]]], List[List[int]]]:
        """
        Apply augmentation to a batch of images
        
        Args:
            images: List of input images
            bboxes_list: List of bounding box lists for each image
            class_labels_list: List of class label lists for each image
            is_training: Whether this is for training
            
        Returns:
            Tuple of (augmented_images, augmented_bboxes_list, augmented_class_labels_list)
        """
        if bboxes_list is None:
            bboxes_list = [[] for _ in images]
        if class_labels_list is None:
            class_labels_list = [[] for _ in images]
        
        augmented_images = []
        augmented_bboxes = []
        augmented_labels = []
        
        for image, bboxes, labels in zip(images, bboxes_list, class_labels_list):
            aug_image, aug_bboxes, aug_labels = self.augment(
                image, bboxes, labels, is_training
            )
            augmented_images.append(aug_image)
            augmented_bboxes.append(aug_bboxes)
            augmented_labels.append(aug_labels)
        
        return augmented_images, augmented_bboxes, augmented_labels
    
    def create_augmentation_samples(self, image: np.ndarray, bboxes: List[List[float]] = None,
                                  class_labels: List[int] = None, num_samples: int = 5) -> List[Tuple[np.ndarray, List[List[float]], List[int]]]:
        """
        Create multiple augmented samples from a single image
        
        Args:
            image: Input image
            bboxes: Bounding boxes
            class_labels: Class labels
            num_samples: Number of augmented samples to create
            
        Returns:
            List of (augmented_image, augmented_bboxes, augmented_class_labels) tuples
        """
        samples = []
        for _ in range(num_samples):
            sample = self.augment(image, bboxes, class_labels, is_training=True)
            samples.append(sample)
        
        return samples

class VisualizationUtils:
    """Utility class for visualization functions"""
    
    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[dict], 
                       class_names = None, 
                       confidence_threshold: float = 0.25) -> np.ndarray:
        """
        Draw detection results on image
        
        Args:
            image: Input image
            detections: List of detection dictionaries with 'bbox', 'confidence', 'class_id'
            class_names: List of class names
            confidence_threshold: Minimum confidence to display
            
        Returns:
            Image with detections drawn
        """
        # Create a copy of the image
        result_image = image.copy()
        
        # Define colors for different classes
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),    # Dark Red
            (0, 128, 0),    # Dark Green
            (0, 0, 128),    # Dark Blue
            (128, 128, 0),  # Olive
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
        ]
        
        for detection in detections:
            if detection['confidence'] < confidence_threshold:
                continue
                
            bbox = detection['bbox']
            class_id = detection['class_id']
            confidence = detection['confidence']
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Convert bbox from [x_center, y_center, width, height] to [x1, y1, x2, y2]
            x_center, y_center, width, height = bbox
            x1 = int((x_center - width/2) * image.shape[1])
            y1 = int((y_center - height/2) * image.shape[0])
            x2 = int((x_center + width/2) * image.shape[1])
            y2 = int((y_center + height/2) * image.shape[0])
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            if class_names and class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"Class {class_id}"
            
            label = f"{class_name}: {confidence:.2f}"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_image, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image
    
    @staticmethod
    def save_detection_result(image: np.ndarray, output_path: str, 
                           filename: str, detections: List[dict] = None, class_names=None, class_names_vi=None):
        """
        Save detection result with visualization
        
        Args:
            image: Input image
            output_path: Output directory path
            filename: Output filename
            detections: List of detections
            class_names: English class names
            class_names_vi: Vietnamese class names
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Draw detections if provided
        if detections:
            result_image = VisualizationUtils.draw_detections(image, detections, class_names)
        else:
            result_image = image
        
        # Save the result image
        output_file = os.path.join(output_path, filename)
        cv2.imwrite(output_file, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        # Save detection data as text file
        if detections:
            txt_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
            txt_file = os.path.join(output_path, txt_filename)
            
            with open(txt_file, 'w', encoding='utf-8') as f:
                for detection in detections:
                    bbox = detection['bbox']
                    class_id = detection['class_id']
                    confidence = detection['confidence']
                    
                    # Get class name
                    if class_names_vi and class_id < len(class_names_vi):
                        class_name = class_names_vi[class_id]
                    elif class_names and class_id < len(class_names):
                        class_name = class_names[class_id]
                    else:
                        class_name = f"Class_{class_id}"
                    
                    # Write detection info
                    f.write(f"Class: {class_name}, Confidence: {confidence:.3f}, BBox: {bbox}\n")

class FileUtils:
    """Utility class for file operations"""
    
    @staticmethod
    def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
        """
        Get all image files from directory
        
        Args:
            directory: Directory path
            extensions: List of file extensions to include
            
        Returns:
            List of image file paths
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        image_files = []
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(directory, filename))
        
        return sorted(image_files)
    
    @staticmethod
    def create_dataset_yaml(dataset_path: str, output_path: str = "dataset.yaml"):
        """
        Create YAML configuration file for dataset
        
        Args:
            dataset_path: Path to dataset directory
            output_path: Output YAML file path
        """
        # Define dataset structure
        dataset_config = {
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 12,  # Number of classes
            'names': [
                'i.423.b', 'p.102', 'p.106.b', 'p.130', 'p.131.a',
                'r.308.b', 'sus', 'w.201.a', 'w.203.c', 'w.207.b',
                'w.207.c', 'w.209'
            ]
        }
        
        # Write YAML file
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Dataset YAML created: {output_path}")

def to_ascii_label(s):
    """Convert string to ASCII-safe label"""
    return ''.join(c for c in s if c.isalnum() or c in (' ', '-', '_')).rstrip() 