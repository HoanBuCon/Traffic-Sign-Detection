import os
import datetime
from ultralytics import YOLO
from config import Config
from utils import DataAugmentation
import torch
from pathlib import Path
import glob
import shutil
import re
import json

class TrafficSignTrainer:
    def __init__(self):
        """Initialize the trainer with configuration"""
        self.config = Config
        self.data_augmentation = DataAugmentation(image_size=self.config.IMAGE_SIZE)
        
        # Tạo thư mục chính để lưu lịch sử training
        self.training_history_dir = 'training_history'
        os.makedirs(self.training_history_dir, exist_ok=True)
        
        # Tạo file log để theo dõi các lần training
        self.training_log_file = os.path.join(self.training_history_dir, 'training_log.json')
        self.load_training_log()
        
    def load_training_log(self):
        """Load training log from JSON file"""
        if os.path.exists(self.training_log_file):
            with open(self.training_log_file, 'r', encoding='utf-8') as f:
                self.training_log = json.load(f)
        else:
            self.training_log = {
                'total_training_sessions': 0,
                'sessions': []
            }
    
    def save_training_log(self):
        """Save training log to JSON file"""
        with open(self.training_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_log, f, indent=2, ensure_ascii=False)
    
    def get_next_train_dir(self, continue_from=None):
        """Tạo thư mục training mới với cấu trúc có tổ chức"""
        # Tìm số training session tiếp theo
        next_session_num = self.training_log['total_training_sessions'] + 1
        
        # Tạo tên thư mục
        if continue_from is not None:
            train_dir_name = f"train{next_session_num}_continue_from_{continue_from}"
        else:
            train_dir_name = f"train{next_session_num}"
        
        # Tạo đường dẫn đầy đủ
        train_dir = os.path.join(self.training_history_dir, train_dir_name)
        
        # Tạo cấu trúc thư mục
        self.create_training_directory_structure(train_dir)
        
        return train_dir, next_session_num
    
    def create_training_directory_structure(self, train_dir):
        """Tạo cấu trúc thư mục cho một session training"""
        # Thư mục chính
        os.makedirs(train_dir, exist_ok=True)
        
        # Thư mục con
        subdirs = [
            'weights',           # Lưu trọng số model
            'plots',            # Lưu biểu đồ training
            'logs',             # Lưu log files
            'configs',          # Lưu cấu hình training
            'results',          # Lưu kết quả validation
            'samples',          # Lưu sample predictions
            'augmented_data',   # Lưu dữ liệu đã augment (nếu có)
            'checkpoints'       # Lưu checkpoints
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
    
    def get_latest_weight(self):
        """Tìm trọng số mới nhất từ các session training trước"""
        if not self.training_log['sessions']:
            return None
        
        # Lấy session cuối cùng
        latest_session = self.training_log['sessions'][-1]
        session_dir = latest_session['session_dir']
        
        # Tìm file trọng số tốt nhất
        weights_dir = os.path.join(session_dir, 'weights')
        best_weight_path = os.path.join(weights_dir, 'best.pt')
        last_weight_path = os.path.join(weights_dir, 'last.pt')
        
        if os.path.exists(best_weight_path):
            return best_weight_path
        elif os.path.exists(last_weight_path):
            return last_weight_path
        
        return None
    
    def create_training_config_file(self, train_dir, session_info):
        """Tạo file cấu hình training cho session này"""
        config_file = os.path.join(train_dir, 'configs', 'training_config.json')
        
        config_data = {
            'session_info': session_info,
            'model_config': {
                'model_size': self.config.MODEL_SIZE,
                'image_size': self.config.IMAGE_SIZE,
                'batch_size': self.config.BATCH_SIZE,
                'epochs': self.config.EPOCHS,
                'learning_rate': self.config.LEARNING_RATE,
                'weight_decay': self.config.WEIGHT_DECAY,
                'momentum': self.config.MOMENTUM
            },
            'augmentation_config': {
                'augmentation_enabled': self.config.AUGMENTATION,
                'augmentation_strength': self.config.AUGMENTATION_STRENGTH,
                'rotation': self.config.ROTATION,
                'horizontal_flip': self.config.HORIZONTAL_FLIP,
                'vertical_flip': self.config.VERTICAL_FLIP,
                'brightness_limit': self.config.BRIGHTNESS_LIMIT,
                'contrast_limit': self.config.CONTRAST_LIMIT,
                'hue_limit': self.config.HUE_LIMIT
            },
            'dataset_config': {
                'dataset_path': self.config.DATASET_PATH,
                'num_classes': 12,
                'class_names': [
                    'i.423.b', 'p.102', 'p.106.b', 'p.130', 'p.131.a',
                    'r.308.b', 'sus', 'w.201.a', 'w.203.c', 'w.207.b',
                    'w.207.c', 'w.209'
                ]
            },
            'training_timestamp': datetime.datetime.now().isoformat(),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'gpu_info': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        return config_file
    
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
        print(f"Augmentation: {self.config.AUGMENTATION}")
        print(f"Augmentation Strength: {self.config.AUGMENTATION_STRENGTH}")
        
    def train(self, start_fresh=True):
        """Train the YOLOv8 model with organized directory structure"""
        try:
            # Always start fresh if requested
            latest_weight = None
            continue_from_session = None
            
            if not start_fresh:
                # Check for latest weights only if not starting fresh
                latest_weight = self.get_latest_weight()
                if latest_weight is not None:
                    # Tìm session number từ đường dẫn
                    session_match = re.search(r'train(\d+)', latest_weight)
                    if session_match:
                        continue_from_session = session_match.group(1)
                    print(f"[INFO] Found previous weights: {latest_weight}")
                    print(f"[INFO] Continuing from session: {continue_from_session}")
            
            # Tạo thư mục training mới
            train_dir, session_num = self.get_next_train_dir(continue_from=continue_from_session)
            
            # Tạo thông tin session
            session_info = {
                'session_number': session_num,
                'session_dir': train_dir,
                'start_time': datetime.datetime.now().isoformat(),
                'continue_from': continue_from_session,
                'status': 'started'
            }
            
            # Tạo file cấu hình
            config_file = self.create_training_config_file(train_dir, session_info)
            print(f"[INFO] Training session {session_num} created at: {train_dir}")
            print(f"[INFO] Config saved to: {config_file}")
            
            # Load model - always start fresh
            print("[INFO] Starting training from scratch")
            model = YOLO(self.config.MODEL_SIZE)
            
            # Train the model with advanced augmentation
            results = model.train(
                data='data.yaml',
                epochs=self.config.EPOCHS,
                imgsz=self.config.IMAGE_SIZE,
                batch=self.config.BATCH_SIZE,
                name=f'traffic_sign_detection_session_{session_num}',
                patience=50,  # Early stopping patience
                save=True,  # Save best and last checkpoints
                device=0 if torch.cuda.is_available() else 'cpu',
                verbose=True,
                
                # Optimizer parameters
                lr0=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                momentum=self.config.MOMENTUM,
                
                # Advanced Augmentation parameters
                flipud=self.config.VERTICAL_FLIP,
                fliplr=self.config.HORIZONTAL_FLIP,
                mosaic=self.config.MOSAIC_PROBABILITY,  # Mosaic augmentation
                mixup=self.config.MIXUP_PROBABILITY,   # Mixup augmentation
                degrees=self.config.ROTATION,
                
                # Additional geometric augmentations
                translate=0.1,  # Translation up to 10%
                scale=0.2,      # Scale from 0.8 to 1.2
                shear=0.0,      # Shear transformation
                perspective=0.0, # Perspective transformation
                
                # Color and brightness augmentations
                hsv_h=0.015,    # HSV-Hue augmentation
                hsv_s=0.7,      # HSV-Saturation augmentation
                hsv_v=0.4,      # HSV-Value augmentation
                
                # Advanced augmentation settings
                copy_paste=0.0,  # Copy-paste augmentation
                auto_augment='randaugment',  # Auto-augment policy
                
                # Save best model
                save_period=1,   # Save checkpoint every epoch (for short training)
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
                
                # Advanced training settings
                workers=0,          # Number of worker threads for data loading (0 for Windows compatibility)
                
                # Validation settings
                val=True,           # Validate during training
                plots=True,         # Save plots for train/val
                rect=False,         # Rectangular training
                resume=False,       # Resume from last checkpoint
                
                # Additional plotting settings
                save_conf=True,     # Save confidences in --save-txt labels
                save_txt=True,      # Save results to *.txt
                save_json=True,     # Save a COCO-JSON results file
                save_crop=True,     # Save cropped prediction boxes
                
                # Advanced augmentation during training
                augment=True,       # Apply augmentations during training
                seed=0,            # Random seed for reproducibility
                deterministic=True, # Deterministic training
                
                # Additional augmentation techniques
                cache=False,       # Cache images for faster training
                
                # Multi-scale training
                multi_scale=False,  # Vary img-size +/- 50%
            )
            
            # Di chuyển kết quả training vào thư mục session
            runs_dir = os.path.join('runs', f'traffic_sign_detection_session_{session_num}')
            if os.path.exists(runs_dir):
                # Di chuyển weights
                weights_dir = os.path.join(runs_dir, 'weights')
                if os.path.exists(weights_dir):
                    for weight_file in ['best.pt', 'last.pt']:
                        src_path = os.path.join(weights_dir, weight_file)
                        dst_path = os.path.join(train_dir, 'weights', weight_file)
                        if os.path.exists(src_path):
                            shutil.move(src_path, dst_path)
                            print(f"Saved {weight_file} to: {dst_path}")
                
                # Di chuyển plots
                plots_dir = os.path.join(runs_dir, 'plots')
                if os.path.exists(plots_dir):
                    dst_plots_dir = os.path.join(train_dir, 'plots')
                    shutil.move(plots_dir, dst_plots_dir)
                    print(f"Moved plots to: {dst_plots_dir}")
                
                # Di chuyển results
                results_dir = os.path.join(runs_dir, 'results.csv')
                if os.path.exists(results_dir):
                    dst_results_dir = os.path.join(train_dir, 'results', 'training_results.csv')
                    shutil.move(results_dir, dst_results_dir)
                    print(f"Moved results to: {dst_results_dir}")
                
                # Di chuyển các file khác
                for file_name in ['confusion_matrix.png', 'confusion_matrix_normalized.png', 'labels.jpg', 'labels_correlogram.jpg']:
                    src_file = os.path.join(runs_dir, file_name)
                    dst_file = os.path.join(train_dir, 'plots', file_name)
                    if os.path.exists(src_file):
                        shutil.move(src_file, dst_file)
                        print(f"Moved {file_name} to: {dst_file}")
                
                # Xóa thư mục runs cũ
                shutil.rmtree(runs_dir)
            
            # Tạo thêm biểu đồ chi tiết
            self.create_detailed_plots(train_dir, results)
            
            # Cập nhật thông tin session
            session_info.update({
                'end_time': datetime.datetime.now().isoformat(),
                'status': 'completed',
                'best_map50': results.results_dict.get('metrics/mAP50(B)', 0),
                'best_map50_95': results.results_dict.get('metrics/mAP50-95(B)', 0),
                'final_epoch': results.results_dict.get('epoch', 0)
            })
            
            # Thêm vào training log
            self.training_log['sessions'].append(session_info)
            self.training_log['total_training_sessions'] = session_num
            self.save_training_log()
            
            # Print training results
            print("\nTraining Results:")
            print(f"Best mAP50: {session_info['best_map50']:.4f}")
            print(f"Best mAP50-95: {session_info['best_map50_95']:.4f}")
            print(f"Session {session_num} completed successfully!")
            print(f"All files saved to: {train_dir}")
            
            return results
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            # Cập nhật status nếu có lỗi
            if 'session_info' in locals():
                session_info.update({
                    'end_time': datetime.datetime.now().isoformat(),
                    'status': 'failed',
                    'error': str(e)
                })
                self.training_log['sessions'].append(session_info)
                self.save_training_log()
            raise
    
    def validate(self, model_path: str = None):
        """
        Validate the trained model
        
        Args:
            model_path: Path to the model to validate (default: best model from latest session)
        """
        if model_path is None:
            # Tìm model tốt nhất từ session cuối cùng
            if self.training_log['sessions']:
                latest_session = self.training_log['sessions'][-1]
                session_dir = latest_session['session_dir']
                model_path = os.path.join(session_dir, 'weights', 'best.pt')
            else:
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
    
    def list_training_sessions(self):
        """Liệt kê tất cả các session training"""
        print("\n=== Training Sessions History ===")
        if not self.training_log['sessions']:
            print("No training sessions found.")
            return
        
        for session in self.training_log['sessions']:
            print(f"\nSession {session['session_number']}:")
            print(f"  Directory: {session['session_dir']}")
            print(f"  Start Time: {session['start_time']}")
            print(f"  Status: {session['status']}")
            if session['status'] == 'completed':
                print(f"  Best mAP50: {session.get('best_map50', 0):.4f}")
                print(f"  Best mAP50-95: {session.get('best_map50_95', 0):.4f}")
                print(f"  Final Epoch: {session.get('final_epoch', 0)}")
            elif session['status'] == 'failed':
                print(f"  Error: {session.get('error', 'Unknown error')}")
            if session.get('continue_from'):
                print(f"  Continued from: Session {session['continue_from']}")
    
    def create_detailed_plots(self, train_dir, results):
        """Tạo các biểu đồ chi tiết sau khi training"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            
            print("\n📊 Creating detailed plots...")
            
            # Đọc kết quả training
            results_file = os.path.join(train_dir, 'results', 'training_results.csv')
            if not os.path.exists(results_file):
                print("❌ Results file not found!")
                return
            
            df = pd.read_csv(results_file)
            if len(df) == 0:
                print("❌ No training data found!")
                return
            
            plots_dir = os.path.join(train_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # 1. Training curves tổng hợp
            self.create_training_curves(df, plots_dir)
            
            # 2. Performance metrics
            self.create_performance_metrics(df, plots_dir)
            
            # 3. Loss analysis
            self.create_loss_analysis(df, plots_dir)
            
            # 4. Learning rate analysis
            self.create_lr_analysis(df, plots_dir)
            
            # 5. Training summary
            self.create_training_summary(df, plots_dir)
            
            print("✅ Detailed plots created successfully!")
            
        except Exception as e:
            print(f"❌ Error creating detailed plots: {e}")
    
    def create_training_curves(self, df, plots_dir):
        """Tạo biểu đồ đường cong training"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # Training loss
        plt.subplot(2, 3, 1)
        if 'train/box_loss' in df.columns:
            plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='blue', marker='o')
        if 'train/cls_loss' in df.columns:
            plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', color='red', marker='s')
        if 'train/dfl_loss' in df.columns:
            plt.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', color='green', marker='^')
        plt.title('Training Losses', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Validation loss
        plt.subplot(2, 3, 2)
        if 'val/box_loss' in df.columns:
            plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='blue', marker='o')
        if 'val/cls_loss' in df.columns:
            plt.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', color='red', marker='s')
        if 'val/dfl_loss' in df.columns:
            plt.plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', color='green', marker='^')
        plt.title('Validation Losses', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # mAP metrics
        plt.subplot(2, 3, 3)
        if 'metrics/mAP50(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', color='orange', marker='o')
        if 'metrics/mAP50-95(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', color='purple', marker='s')
        plt.title('mAP Metrics', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision and Recall
        plt.subplot(2, 3, 4)
        if 'metrics/precision(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='green', marker='o')
        if 'metrics/recall(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='red', marker='s')
        plt.title('Precision & Recall', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate
        plt.subplot(2, 3, 5)
        if 'lr/pg0' in df.columns:
            plt.plot(df['epoch'], df['lr/pg0'], label='Learning Rate', color='brown', marker='o')
        plt.title('Learning Rate Schedule', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Combined loss
        plt.subplot(2, 3, 6)
        if all(col in df.columns for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']):
            total_train_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
            plt.plot(df['epoch'], total_train_loss, label='Total Train Loss', color='blue', marker='o')
        
        if all(col in df.columns for col in ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']):
            total_val_loss = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
            plt.plot(df['epoch'], total_val_loss, label='Total Val Loss', color='red', marker='s')
        
        plt.title('Total Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_curves_detailed.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_metrics(self, df, plots_dir):
        """Tạo biểu đồ performance metrics"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Metrics comparison
        metrics = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']
        colors = ['orange', 'purple', 'green', 'red']
        markers = ['o', 's', '^', 'D']
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                plt.plot(df['epoch'], df[metric], label=metric.split('/')[-1].replace('(B)', ''), 
                        color=colors[i], marker=markers[i], linewidth=2, markersize=6)
        
        plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_loss_analysis(self, df, plots_dir):
        """Tạo biểu đồ phân tích loss"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        train_losses = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
        val_losses = ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']
        colors = ['blue', 'red', 'green']
        
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            if train_loss in df.columns:
                plt.plot(df['epoch'], df[train_loss], label=f'Train {train_loss.split("/")[-1]}', 
                        color=colors[i], linestyle='-', marker='o', linewidth=2)
            if val_loss in df.columns:
                plt.plot(df['epoch'], df[val_loss], label=f'Val {val_loss.split("/")[-1]}', 
                        color=colors[i], linestyle='--', marker='s', linewidth=2)
        
        plt.title('Training vs Validation Losses', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'loss_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_lr_analysis(self, df, plots_dir):
        """Tạo biểu đồ phân tích learning rate"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        if 'lr/pg0' in df.columns:
            plt.plot(df['epoch'], df['lr/pg0'], color='brown', marker='o', linewidth=2, markersize=6)
            plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'learning_rate_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_training_summary(self, df, plots_dir):
        """Tạo bảng tóm tắt training"""
        import pandas as pd
        
        if len(df) == 0:
            return
        
        last_epoch = df.iloc[-1]
        
        # Create summary data
        summary_data = {
            'Metric': [
                'mAP50', 'mAP50-95', 'Precision', 'Recall',
                'Train Box Loss', 'Train Class Loss', 'Train DFL Loss',
                'Val Box Loss', 'Val Class Loss', 'Val DFL Loss',
                'Total Train Loss', 'Total Val Loss', 'Training Time (s)'
            ],
            'Value': [
                f"{last_epoch['metrics/mAP50(B)']:.4f}",
                f"{last_epoch['metrics/mAP50-95(B)']:.4f}",
                f"{last_epoch['metrics/precision(B)']:.4f}",
                f"{last_epoch['metrics/recall(B)']:.4f}",
                f"{last_epoch['train/box_loss']:.4f}",
                f"{last_epoch['train/cls_loss']:.4f}",
                f"{last_epoch['train/dfl_loss']:.4f}",
                f"{last_epoch['val/box_loss']:.4f}",
                f"{last_epoch['val/cls_loss']:.4f}",
                f"{last_epoch['val/dfl_loss']:.4f}",
                f"{last_epoch['train/box_loss'] + last_epoch['train/cls_loss'] + last_epoch['train/dfl_loss']:.4f}",
                f"{last_epoch['val/box_loss'] + last_epoch['val/cls_loss'] + last_epoch['val/dfl_loss']:.4f}",
                f"{last_epoch['time']:.2f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(plots_dir, 'training_summary_detailed.csv'), index=False)

def main():
    """Main function to run training"""
    trainer = TrafficSignTrainer()
    
    print("Setting up training...")
    trainer.setup_training()
    
    # Hiển thị lịch sử training
    trainer.list_training_sessions()
    
    print("\nStarting training...")
    results = trainer.train(start_fresh=True)  # Luôn bắt đầu training mới
    
    print("\nStarting validation...")
    val_results = trainer.validate()
    
    print("\nTraining and validation completed!")

if __name__ == "__main__":
    main() 