import torch
import os
from ultralytics import YOLO
import shutil
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def check_disk_space(drive):
    total, used, free = shutil.disk_usage(drive)
    return free / (1024 ** 3)  # Dung lượng trống (GB)

def check_environment():
    print("🔍 Checking environment...")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU. Using CPU (slow).")
    
    c_free = check_disk_space("C:\\")
    d_free = check_disk_space("D:\\")
    print(f"💾 Drive C: {c_free:.2f} GB free")
    print(f"💾 Drive D: {d_free:.2f} GB free")
    if c_free < 1:
        print("🚨 Warning: Drive C is low (<1GB). Free up space or move swap file to D.")
    if d_free < 10:
        print("🚨 Warning: Drive D is low (<10GB). Dataset and logs may fill it up.")

def train_model():
    # 1. KIỂM TRA MÔI TRƯỜNG
    # =================================
    check_disk_space("C:\\")  # Kiểm tra dung lượng ổ C:
    check_environment()  # Gọi hàm kiểm tra môi trường và dung lượng ổ
    # Kiểm tra xem có GPU (như card NVIDIA) để tăng tốc độ huấn luyện không.
    if torch.cuda.is_available():
        print("✅ GPU is available! We will use it for training.")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ GPU is not available. Training will run on CPU, which might be very slow.")
        print("   If you are on Kaggle/Colab, make sure to enable the GPU accelerator.")


    # 2. CÁC THAM SỐ HUẤN LUYỆN
    # =================================
    # Bạn có thể thay đổi các giá trị này để thử nghiệm.
    PRETRAINED_MODEL = 'yolov8n.pt'
    DATA_CONFIG = 'data.yaml'
    EPOCHS = 100
    IMAGE_SIZE = 640  # Giảm từ 640 để tiết kiệm bộ nhớ GPU
    BATCH_SIZE = 8    # Giảm từ 8 để tránh CUDA out of memory
    WORKERS = 4       # Tắt multiprocessing hoặc để là 2 đến 4 để giảm lỗi shared file mapping


    # 3. HUẤN LUYỆN MÔ HÌNH
    # =================================
    # Ghi chú: Để chuyển sang notebook, bạn chỉ cần copy phần code này vào một cell.
    print("\\n🚀 Starting training...")
    try:
        # Tải một mô hình YOLOv8 đã được huấn luyện trước (pretrained).
        model = YOLO(PRETRAINED_MODEL)

        # Bắt đầu huấn luyện mô hình trên dataset của bạn.
        # Kết quả sẽ được lưu trong thư mục `runs/detect/train/`
        results = model.train(
            data=DATA_CONFIG,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            workers=WORKERS,
            name='yolov8_traffic_sign_training', # Tên thư mục lưu kết quả
            resume=False # Huấn luyện tiếp từ lần chạy trước
        )
        print("✅ Training completed successfully!")
        
        # Lấy đường dẫn đến thư mục kết quả mới nhất
        latest_run_dir = max([os.path.join('runs/detect', d) for d in os.listdir('runs/detect') if os.path.isdir(os.path.join('runs/detect', d))], key=os.path.getmtime)
        weights_path = os.path.join(latest_run_dir, 'weights/best.pt')
        print(f"👉 Best model weights saved at: {weights_path}")
        return weights_path

    except Exception as e:
        print(f"❌ An error occurred during training: {e}")
        return None


def run_inference(weights_path):
    # 4. CHẠY NHẬN DIỆN (INFERENCE)
    # =================================
    # Sau khi huấn luyện, bạn có thể dùng đoạn code dưới đây để nhận diện trên ảnh mới.
    if not weights_path:
        print("\\n⏩ Skipping inference because training did not complete successfully.")
        return

    print("\\n🔎 Running inference on a sample image...")
    try:
        # Tải mô hình tốt nhất đã được huấn luyện
        trained_model = YOLO(weights_path)

        # Lấy một ảnh ngẫu nhiên từ tập validation để kiểm tra
        val_image_dir = 'dataset/images/val'
        if os.path.exists(val_image_dir) and len(os.listdir(val_image_dir)) > 0:
            sample_image_name = os.listdir(val_image_dir)[0]
            sample_image_path = os.path.join(val_image_dir, sample_image_name)
            
            print(f"   - Predicting on image: {sample_image_path}")

            # Chạy nhận diện và lưu kết quả
            results = trained_model.predict(source=sample_image_path, save=True)
            print("   - Prediction complete! Result saved in the latest `runs/detect/predict` folder.")
        else:
            print("   - Could not find any validation images to test.")

    except Exception as e:
        print(f"❌ An error occurred during inference: {e}")

if __name__ == '__main__':
    best_weights_path = train_model()
    # Bỏ comment (xóa dấu #) ở dòng dưới để tự động chạy inference sau khi train xong.
    # run_inference(best_weights_path)