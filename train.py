import os
import datetime
from ultralytics import YOLO
from config import Config
from utils import DataAugmentation
import torch
from pathlib import Path
import glob
import shutil

class TrafficSignTrainer:
    def __init__(self):
        """Initialize the trainer with configuration"""
        self.config = Config
        self.data_augmentation = DataAugmentation(image_size=self.config.IMAGE_SIZE)
        # Thêm biến đếm số lần train
        self.all_weight_dir = 'all_weight'
        os.makedirs(self.all_weight_dir, exist_ok=True)
        
    def get_next_train_dir(self):
        """Tìm tên thư mục train tiếp theo trong all_weight"""
        existing = [d for d in os.listdir(self.all_weight_dir) if d.startswith('train') and os.path.isdir(os.path.join(self.all_weight_dir, d))]
        nums = [int(d.replace('train', '')) for d in existing if d.replace('train', '').isdigit()]
        next_num = max(nums) + 1 if nums else 1
        return os.path.join(self.all_weight_dir, f'train{next_num}')
        
    def get_latest_weight(self):
        """Find the latest best.pt or last.pt in all_weight/trainX, return its path or None if not found"""
        existing = [d for d in os.listdir(self.all_weight_dir) if d.startswith('train') and os.path.isdir(os.path.join(self.all_weight_dir, d))]
        if not existing:
            return None
        nums = [int(d.replace('train', '')) for d in existing if d.replace('train', '').isdigit()]
        if not nums:
            return None
        latest_num = max(nums)
        latest_dir = os.path.join(self.all_weight_dir, f'train{latest_num}')
        best_path = os.path.join(latest_dir, 'best.pt')
        last_path = os.path.join(latest_dir, 'last.pt')
        if os.path.exists(best_path):
            return best_path
        elif os.path.exists(last_path):
            return last_path
        return None
        
    def setup_training(self):
        """Setup training environment and create necessary files/directories"""
        # Create necessary directories
        self.config.create_directories()
        
        # Generate dataset.yaml only if it doesn't exist
        if not os.path.exists('data.yaml'):
            dataset_content = self.config.get_dataset_yaml()
            with open('data.yaml', 'w') as f:
                f.write(dataset_content)
            print("Created new data.yaml file")
        else:
            print("Using existing data.yaml file")
            
        # Print training configuration
        print("\nTraining Configuration:")
        print(f"Model: {self.config.MODEL_SIZE}")
        print(f"Image Size: {self.config.IMAGE_SIZE}")
        print(f"Batch Size: {self.config.BATCH_SIZE}")
        print(f"Epochs: {self.config.EPOCHS}")
        print(f"Learning Rate: {self.config.LEARNING_RATE}")
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        
    def train(self):
        """Train the YOLOv8 model on traffic sign dataset, resuming from latest weights if available (without overwriting them)"""
        try:
            # Check for latest weights
            latest_weight = self.get_latest_weight()
            temp_resume_path = None
            if latest_weight is not None:
                # Copy to a temp file to avoid overwriting
                temp_resume_path = os.path.join(self.all_weight_dir, 'tmp_resume.pt')
                shutil.copy2(latest_weight, temp_resume_path)
                print(f"[INFO] Resume training from: {latest_weight} (copied to {temp_resume_path})")
                model = YOLO(temp_resume_path)
            else:
                print("[INFO] No previous weights found. Training from scratch.")
                model = YOLO(self.config.MODEL_SIZE)
            
            # Train the model
            results = model.train(
                data='data.yaml',
                epochs=self.config.EPOCHS,
                imgsz=self.config.IMAGE_SIZE,
                batch=self.config.BATCH_SIZE,
                name='traffic_sign_detection',
                patience=50,  # Early stopping patience
                save=True,  # Save best and last checkpoints
                device=0 if torch.cuda.is_available() else 'cpu',
                verbose=True,
                
                # Optimizer parameters
                lr0=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                momentum=self.config.MOMENTUM,
                
                # Augmentation parameters
                flipud=self.config.VERTICAL_FLIP,
                fliplr=self.config.HORIZONTAL_FLIP,
                mosaic=0.5,  # Mosaic augmentation
                mixup=0.3,   # Mixup augmentation
                degrees=self.config.ROTATION,
                
                # Save best model
                save_period=10,  # Save checkpoint every 10 epochs
                project='runs',  # Project name
                exist_ok=True,   # Overwrite existing experiment
                
                # Additional parameters for better convergence
                warmup_epochs=3.0,  # Warmup epochs
                warmup_momentum=0.8,  # Warmup momentum
                warmup_bias_lr=0.1,  # Warmup initial bias lr
                box=7.5,  # Box loss gain
                cls=0.5,  # Classification loss gain
                dfl=1.5,  # DFL loss gain
                close_mosaic=10,  # Close mosaic augmentation for last 10 epochs
            )
            
            # Lưu best.pt và last.pt vào thư mục all_weight/trainX
            best_model_path = os.path.join('runs', 'traffic_sign_detection', 'weights', 'best.pt')
            last_model_path = os.path.join('runs', 'traffic_sign_detection', 'weights', 'last.pt')
            train_dir = self.get_next_train_dir()
            os.makedirs(train_dir, exist_ok=True)
            if os.path.exists(best_model_path):
                dest_best = os.path.join(train_dir, 'best.pt')
                os.replace(best_model_path, dest_best)
                print(f"\nBest model saved to: {dest_best}")
            
            if os.path.exists(last_model_path):
                dest_last = os.path.join(train_dir, 'last.pt')
                os.replace(last_model_path, dest_last)
                print(f"Last model saved to: {dest_last}")
            
            # Print training results
            print("\nTraining Results:")
            print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
            print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
            
            return results
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            raise
    
    def validate(self, model_path: str = None):
        """
        Validate the trained model
        
        Args:
            model_path: Path to the model to validate (default: best model)
        """
        if model_path is None:
            model_path = self.config.BEST_MODEL_PATH
            
        if not os.path.exists(model_path):
            print(f"Model not found at: {model_path}")
            return
            
        try:
            # Load the model
            model = YOLO(model_path)
            
            # Validate
            results = model.val(
                data='data.yaml',
                imgsz=self.config.IMAGE_SIZE,
                batch=self.config.BATCH_SIZE,
                device=0 if torch.cuda.is_available() else 'cpu',
                verbose=True
            )
            
            # Print validation results
            print("\nValidation Results:")
            print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
            print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
            
            return results
            
        except Exception as e:
            print(f"\nError during validation: {str(e)}")
            raise

def main():
    """Main function to run training"""
    trainer = TrafficSignTrainer()
    
    print("Setting up training...")
    trainer.setup_training()
    
    print("\nStarting training...")
    results = trainer.train()
    
    print("\nStarting validation...")
    val_results = trainer.validate()
    
    print("\nTraining and validation completed!")

if __name__ == "__main__":
    main() 