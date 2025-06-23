import os
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
from typing import List, Union, Tuple
import numpy as np
from config import Config
from utils import ImageEnhancer, VisualizationUtils, FileUtils

class TrafficSignDetector:
    def __init__(self, model_path: str = Config.BEST_MODEL_PATH):
        """
        Initialize the traffic sign detector
        
        Args:
            model_path: Path to the trained model
        """
        self.model = YOLO(model_path)
        self.image_enhancer = ImageEnhancer()
        self.config = Config
        
    def enhance_image_for_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Apply multiple enhancement techniques for better inference
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        if not self.config.ENABLE_IMAGE_ENHANCEMENT:
            return image
            
        # Apply denoising
        image = self.image_enhancer.denoise_image(image)
        
        # Apply sharpening
        image = self.image_enhancer.sharpen_image(image)
        
        # Enhance contrast and brightness
        image = self.image_enhancer.enhance_image(
            image, 
            enhancement_level=self.config.CONTRAST_ENHANCEMENT
        )
        
        # Apply gamma correction
        image = self.image_enhancer.adjust_gamma(
            image, 
            gamma=self.config.GAMMA_CORRECTION
        )
        
        return image
    
    def predict_image(self, image_path: str, save_result: bool = True) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect traffic signs in an image
        
        Args:
            image_path: Path to the input image
            save_result: Whether to save the visualization
            
        Returns:
            Tuple of (annotated image, detections)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Enhance image for better inference
        enhanced_image = self.enhance_image_for_inference(image)
        
        # Run inference
        results = self.model.predict(
            enhanced_image,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.NMS_THRESHOLD,
            max_det=self.config.MAX_DETECTIONS,
            verbose=self.config.VERBOSE
        )
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].tolist(),  # Convert to absolute coordinates
                    'confidence': float(box.conf),
                    'class_id': int(box.cls)
                }
                detections.append(detection)
        
        # Draw detections
        annotated_image = VisualizationUtils.draw_detections(
            image,
            detections,
            class_names=self.model.names,
            confidence_threshold=self.config.CONFIDENCE_THRESHOLD
        )
        
        # Save result if requested
        if save_result:
            output_filename = os.path.basename(image_path)
            output_path = os.path.join(self.config.PREDICTIONS_DIR, output_filename)
            VisualizationUtils.save_detection_result(
                annotated_image,
                self.config.PREDICTIONS_DIR,
                output_filename,
                detections
            )
        
        return annotated_image, detections
    
    def predict_directory(self, input_dir: str = Config.CUSTOM_IMAGES_DIR) -> None:
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing input images
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.config.PREDICTIONS_DIR, exist_ok=True)
        
        # Get all image files
        image_files = FileUtils.get_image_files(input_dir)
        
        # Process each image
        for image_path in image_files:
            try:
                print(f"Processing {image_path}...")
                self.predict_image(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")

def main():
    """Main function to run inference"""
    # Create necessary directories
    Config.create_directories()
    
    # Initialize detector
    detector = TrafficSignDetector()
    
    # Process all images in custom_images directory
    detector.predict_directory()

if __name__ == "__main__":
    main() 