import os
import cv2
import numpy as np
import yaml
import shutil
from pathlib import Path
from tqdm import tqdm
import random
from utils import DataAugmentation, ImageEnhancer
from config import Config

class AutoDatasetAugmenter:
    """Automatically augment dataset with advanced techniques"""
    
    def __init__(self, dataset_path: str = "dataset", output_path: str = "augmented_dataset"):
        """
        Initialize dataset augmenter
        
        Args:
            dataset_path: Path to original dataset
            output_path: Path to save augmented dataset
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.data_augmenter = DataAugmentation(image_size=Config.IMAGE_SIZE)
        self.image_enhancer = ImageEnhancer()
        
        # Create output directories
        self.create_output_directories()
        
        # Load class names
        self.load_class_names()
    
    def create_output_directories(self):
        """Create output directory structure"""
        splits = ['train', 'val', 'test']
        for split in splits:
            os.makedirs(os.path.join(self.output_path, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, 'labels', split), exist_ok=True)
    
    def load_class_names(self):
        """Load class names from data.yaml"""
        with open('data.yaml', 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
        self.class_names = data_yaml['names']
        print(f"Loaded {len(self.class_names)} classes: {self.class_names}")
    
    def read_yolo_labels(self, label_path: str) -> tuple:
        """
        Read YOLO format labels
        
        Args:
            label_path: Path to label file
            
        Returns:
            Tuple of (bboxes, class_labels)
        """
        bboxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        return bboxes, class_labels
    
    def write_yolo_labels(self, label_path: str, bboxes: list, class_labels: list):
        """
        Write YOLO format labels
        
        Args:
            label_path: Path to save label file
            bboxes: List of bounding boxes
            class_labels: List of class labels
        """
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def augment_single_image(self, image_path: str, label_path: str, 
                           num_augmentations: int = 5) -> list:
        """
        Augment a single image with multiple techniques
        
        Args:
            image_path: Path to image file
            label_path: Path to label file
            num_augmentations: Number of augmented samples to create
            
        Returns:
            List of augmented samples [(image, bboxes, class_labels), ...]
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return []
        
        # Read labels
        bboxes, class_labels = self.read_yolo_labels(label_path)
        
        # Create augmented samples
        samples = []
        
        # Original image (enhanced)
        enhanced_image = self.image_enhancer.enhance_image(image)
        samples.append((enhanced_image, bboxes, class_labels))
        
        # Create augmented samples
        for i in range(num_augmentations):
            try:
                aug_image, aug_bboxes, aug_labels = self.data_augmenter.augment(
                    image, bboxes, class_labels, is_training=True
                )
                samples.append((aug_image, aug_bboxes, aug_labels))
            except Exception as e:
                print(f"Warning: Augmentation failed for {image_path}: {e}")
                continue
        
        return samples
    
    def process_split(self, split: str, num_augmentations: int = 5):
        """
        Process all images in a split
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            num_augmentations: Number of augmented samples per image
        """
        print(f"\nProcessing {split} split...")
        
        # Get image and label paths
        images_dir = os.path.join(self.dataset_path, 'images', split)
        labels_dir = os.path.join(self.dataset_path, 'labels', split)
        
        if not os.path.exists(images_dir):
            print(f"Warning: Images directory {images_dir} does not exist")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(images_dir).glob(f"*{ext}"))
            image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images in {split} split")
        
        # Process each image
        total_samples = 0
        for image_file in tqdm(image_files, desc=f"Augmenting {split}"):
            image_path = str(image_file)
            label_path = os.path.join(labels_dir, image_file.stem + '.txt')
            
            # Augment image
            samples = self.augment_single_image(image_path, label_path, num_augmentations)
            
            # Save augmented samples
            for i, (aug_image, aug_bboxes, aug_labels) in enumerate(samples):
                # Generate filename
                if i == 0:
                    # Original enhanced image
                    filename = f"{image_file.stem}_enhanced{image_file.suffix}"
                else:
                    # Augmented image
                    filename = f"{image_file.stem}_aug{i}{image_file.suffix}"
                
                # Save image
                output_image_path = os.path.join(self.output_path, 'images', split, filename)
                cv2.imwrite(output_image_path, aug_image)
                
                # Save labels
                output_label_path = os.path.join(self.output_path, 'labels', split, 
                                               filename.replace(image_file.suffix, '.txt'))
                self.write_yolo_labels(output_label_path, aug_bboxes, aug_labels)
                
                total_samples += 1
        
        print(f"Created {total_samples} samples for {split} split")
    
    def create_augmented_data_yaml(self):
        """Create data.yaml for augmented dataset"""
        yaml_content = f"""# Augmented dataset configuration
path: {os.path.abspath(self.output_path)}
train: images/train
val: images/val

# Number of classes
nc: {len(self.class_names)}

# Class names
names:
"""
        for i, class_name in enumerate(self.class_names):
            yaml_content += f"  {i}: {class_name}\n"
        
        # Save data.yaml
        output_yaml_path = os.path.join(self.output_path, 'data.yaml')
        with open(output_yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"Created data.yaml at: {output_yaml_path}")
    
    def augment_dataset(self, train_augmentations: int = 5, val_augmentations: int = 2, 
                       test_augmentations: int = 1):
        """
        Augment entire dataset
        
        Args:
            train_augmentations: Number of augmentations per training image
            val_augmentations: Number of augmentations per validation image
            test_augmentations: Number of augmentations per test image
        """
        print("Starting dataset augmentation...")
        print(f"Output directory: {self.output_path}")
        
        # Process each split
        self.process_split('train', train_augmentations)
        self.process_split('val', val_augmentations)
        self.process_split('test', test_augmentations)
        
        # Create data.yaml for augmented dataset
        self.create_augmented_data_yaml()
        
        print(f"\nDataset augmentation completed!")
        print(f"Augmented dataset saved to: {self.output_path}")
        
        # Print statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print dataset statistics"""
        print("\nDataset Statistics:")
        
        for split in ['train', 'val', 'test']:
            images_dir = os.path.join(self.output_path, 'images', split)
            labels_dir = os.path.join(self.output_path, 'labels', split)
            
            if os.path.exists(images_dir):
                num_images = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                num_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
                print(f"{split}: {num_images} images, {num_labels} labels")
            else:
                print(f"{split}: No data")

def main():
    """Main function to run dataset augmentation"""
    # Initialize augmenter
    augmenter = AutoDatasetAugmenter(
        dataset_path="dataset",
        output_path="augmented_dataset"
    )
    
    # Run augmentation
    augmenter.augment_dataset(
        train_augmentations=5,  # 5 augmented samples per training image
        val_augmentations=2,    # 2 augmented samples per validation image
        test_augmentations=1    # 1 augmented sample per test image
    )

if __name__ == "__main__":
    main() 