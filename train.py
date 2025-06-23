import os
from ultralytics import YOLO
from config import Config
from utils import DataAugmentation
import torch
from pathlib import Path

class TrafficSignTrainer:
    def __init__(self):
        """Initialize the trainer with configuration"""
        self.config = Config
        self.data_augmentation = DataAugmentation(image_size=self.config.IMAGE_SIZE)
        
    def setup_training(self):
        """Setup training environment and create necessary files/directories"""
        # Create necessary directories
        self.config.create_directories()
        
        # Generate dataset.yaml
        dataset_content = self.config.get_dataset_yaml()
        with open('data.yaml', 'w') as f:
            f.write(dataset_content)
            
        # Print training configuration
        print("\nTraining Configuration:")
        print(f"Model: {self.config.MODEL_SIZE}")
        print(f"Image Size: {self.config.IMAGE_SIZE}")
        print(f"Batch Size: {self.config.BATCH_SIZE}")
        print(f"Epochs: {self.config.EPOCHS}")
        print(f"Learning Rate: {self.config.LEARNING_RATE}")
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        
    def train(self):
        """Train the YOLOv8 model on traffic sign dataset"""
        try:
            # Load the model
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
            
            # Save the trained model
            best_model_path = os.path.join('runs', 'traffic_sign_detection', 'weights', 'best.pt')
            last_model_path = os.path.join('runs', 'traffic_sign_detection', 'weights', 'last.pt')
            
            if os.path.exists(best_model_path):
                os.replace(best_model_path, self.config.BEST_MODEL_PATH)
                print(f"\nBest model saved to: {self.config.BEST_MODEL_PATH}")
            
            if os.path.exists(last_model_path):
                os.replace(last_model_path, self.config.LAST_MODEL_PATH)
                print(f"Last model saved to: {self.config.LAST_MODEL_PATH}")
            
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