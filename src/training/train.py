import os
import json
import datetime
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
datasets_dir = repo_root / "data" / "dataset"

# Initialize Ultralytics settings once (avoid repeated resets in spawned workers on Windows)
if os.environ.get("YOLO_SETTINGS_INIT", "0") != "1":
    os.environ["YOLO_DATASETS_DIR"] = str(datasets_dir)
    from ultralytics.utils import SETTINGS
    SETTINGS["datasets_dir"] = str(datasets_dir)
    SETTINGS["sync"] = False
    SETTINGS["checks"] = False
    SETTINGS["downloads"] = False
    os.environ["YOLO_SETTINGS_INIT"] = "1"
    print(f"[INFO] YOLO datasets_dir set to: {datasets_dir}")

from ultralytics import YOLO
import torch

try:
    from src.training.config import Config
    from src.core.utils import DataAugmentation
except ModuleNotFoundError:
    import sys as _sys, pathlib as _pathlib
    _repo_root = _pathlib.Path(__file__).resolve().parents[2]
    _sys.path.insert(0, str(_repo_root))
    from src.training.config import Config
    from src.core.utils import DataAugmentation


class TrafficSignTrainer:
    def _build_resolved_dataset_yaml(self, yaml_path: str) -> str:
        import yaml
        from pathlib import Path

        with open(yaml_path, 'r', encoding='utf-8') as f:
            spec = yaml.safe_load(f) or {}

        base_dir = Path(yaml_path).resolve().parent
        ds_root = base_dir / spec.get('path', '') if 'path' in spec else base_dir

        def resolve_split(split_key: str, value: str) -> str:
            p = Path(value)
            if p.is_absolute() and p.exists():
                return str(p)
            for candidate in [
                ds_root / value,
                ds_root / f"images/{split_key}",
                ds_root / f"{split_key}/images"
            ]:
                if candidate.exists():
                    return str(candidate.resolve())
            return str(ds_root.resolve())

        resolved = {
            k: resolve_split(k, spec.get(k, ''))
            for k in ('train', 'val', 'test') if k in spec
        }
        for k in ('nc', 'names', 'descriptions'):
            if k in spec:
                resolved[k] = spec[k]

        out_dir = Path('runs') / 'traffic_sign_detection'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_yaml = out_dir / 'data_resolved.yaml'
        with open(out_yaml, 'w', encoding='utf-8') as f:
            yaml.safe_dump(resolved, f, allow_unicode=True)
        return str(out_yaml.resolve())

    def __init__(self):
        self.config = Config
        self.data_augmentation = DataAugmentation(image_size=self.config.IMAGE_SIZE)
        self.all_weight_dir = Config.ALL_WEIGHT_DIR
        os.makedirs(self.all_weight_dir, exist_ok=True)

    def setup_training(self):
        """Khởi tạo môi trường train (load dataset YAML, in config, vv.)"""
        from src.core.utils import find_data_yaml
        import yaml

        self.config.create_directories()

        # Tìm file data.yaml
        self.data_yaml_path = find_data_yaml()
        if not os.path.exists(self.data_yaml_path):
            raise FileNotFoundError(
                f"Không tìm thấy data.yaml. Đặt tại data/dataset/data.yaml hoặc config.DATA_YAML"
            )

        print(f"Using dataset YAML at: {self.data_yaml_path}")

        # Sinh ra file data_resolved.yaml có đường dẫn tuyệt đối
        self.data_yaml_resolved = self._build_resolved_dataset_yaml(self.data_yaml_path)

        # In config huấn luyện
        print("\nTraining Configuration:")
        print(f"Model: {self.config.MODEL_SIZE}")
        print(f"Image Size: {self.config.IMAGE_SIZE}")
        print(f"Batch Size: {self.config.BATCH_SIZE}")
        print(f"Epochs: {self.config.EPOCHS}")
        print(f"Learning Rate: {self.config.LEARNING_RATE}")
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    def _resolve_base_model_path(self) -> str:
        """Tìm model .pt local, tuyệt đối không tải online"""
        base = self.config.MODEL_SIZE
        from src.training.config import WEIGHT_BASE_DIR
        candidates = [
            Path(WEIGHT_BASE_DIR) / base,
            repo_root / "src" / "weight" / base,
            repo_root / "src" / "weights" / base,
            repo_root / base,
        ]
        for c in candidates:
            if c.exists():
                print(f"Found local YOLO weight: {c.resolve()}")
                return str(c.resolve())
        raise FileNotFoundError(
            f"Không tìm thấy file '{base}' trong bất kỳ vị trí nào sau đây:\n"
            + "\n".join(map(str, candidates))
        )

    def train(self):
        """Train YOLOv8 an toàn, không tải model ngoài"""
        base_model_path = self._resolve_base_model_path()
        print(f"[INFO] Using local pretrained model: {base_model_path}")

        model = YOLO(base_model_path, task="detect")
        model.model.args["pretrained"] = False

        results = model.train(
            data=self.data_yaml_resolved,
            epochs=self.config.EPOCHS,
            imgsz=self.config.IMAGE_SIZE,
            batch=self.config.BATCH_SIZE,
            name="traffic_sign_detection",
            device=0 if torch.cuda.is_available() else "cpu",
            pretrained=False,
            verbose=True,
            lr0=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            momentum=self.config.MOMENTUM,
            flipud=self.config.VERTICAL_FLIP,
            fliplr=self.config.HORIZONTAL_FLIP,
            mosaic=0.5,
            mixup=0.3,
            degrees=self.config.ROTATION,
            workers=self.config.WORKERS,
            save_period=10,
            exist_ok=True,
        )

        best = Path("runs/detect/traffic_sign_detection/weights/best.pt")
        last = Path("runs/detect/traffic_sign_detection/weights/last.pt")
        train_dir = Path(self.all_weight_dir) / f"train_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        train_dir.mkdir(parents=True, exist_ok=True)

        for src, name in [(best, "best.pt"), (last, "last.pt")]:
            if src.exists():
                dst = train_dir / name
                src.replace(dst)
                print(f"Saved {name} → {dst}")

        print("\n✅ Training finished successfully!")
        return results
    
    def validate(self, model_path: str = None):
        """
        Validate the trained model
        
        Args:
            model_path: Path to the model to validate (default: best model)
        """
        if model_path is None:
            model_path = self._get_latest_best_model()
            
        if not os.path.exists(model_path):
            print(f"Model not found at: {model_path}")
            return
            
        try:
            # Load the model
            model = YOLO(model_path)
            
            # Validate
            results = model.val(
                data=self.data_yaml_resolved,
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

    def _get_latest_best_model(self):
        """Tìm file best.pt trong thư mục lưu trữ gần nhất (do hàm train vừa tạo ra)"""
        try:
            # Đường dẫn nơi chứa các folder train_... (định nghĩa trong config)
            base_path = Path(self.all_weight_dir)
            
            if not base_path.exists():
                # Nếu chưa có folder custom, trả về đường dẫn mặc định của YOLO
                return str(Path("runs/detect/traffic_sign_detection/weights/best.pt"))

            # Lấy tất cả thư mục con bắt đầu bằng 'train_'
            dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('train_')]
            
            if not dirs:
                # Nếu không tìm thấy folder train nào, trả về mặc định
                return str(Path("runs/detect/traffic_sign_detection/weights/best.pt"))
            
            # Tìm thư mục có thời gian sửa đổi mới nhất (vừa train xong)
            latest_dir = max(dirs, key=lambda d: d.stat().st_mtime)
            
            return str(latest_dir / "best.pt")
        except Exception as e:
            print(f"Lỗi khi tìm model: {e}")
            return "runs/detect/traffic_sign_detection/weights/best.pt"

    def test(self, model_path: str = None):
        """Hàm test trên tập kiểm thử (Test Set)"""
        if model_path is None:
            model_path = self._get_latest_best_model()
            
        print(f"\n[INFO] Starting Testing on Test Set with model: {model_path}")
        if not os.path.exists(model_path):
            print(f"Không tìm thấy model tại {model_path} để test.")
            return

        model = YOLO(model_path)
        
        # Chạy validation nhưng trên tập dữ liệu 'test'
        # Lưu ý: split='test' yêu cầu file yaml phải khai báo đường dẫn 'test:'
        results = model.val(
            data=self.data_yaml_resolved,
            split='test',  
            imgsz=self.config.IMAGE_SIZE,
            batch=self.config.BATCH_SIZE,
            device=0 if torch.cuda.is_available() else 'cpu',
            verbose=True
        )
        return results

def main():
    """Main function to run training"""
    trainer = TrafficSignTrainer()
    
    print("Setting up training...")
    trainer.setup_training()
    
    print("\nStarting training...")
    results = trainer.train()
    
    print("\nStarting validation...")
    val_results = trainer.validate()
    
    print("\nStarting test on test set...")
    test_results = trainer.test()
    print("\nTraining, validation and test completed!")

if __name__ == "__main__":
    main() 