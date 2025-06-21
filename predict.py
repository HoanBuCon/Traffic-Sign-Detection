import os
import glob
from ultralytics import YOLO
import argparse

def predict(weights_path, source_path):
    """
    Chạy nhận diện YOLOv8 trên một thư mục ảnh.
    """
    if not os.path.exists(source_path):
        print(f"❌ Lỗi: Thư mục nguồn '{source_path}' không tồn tại.")
        return

    if not any(fname.lower().endswith(('.png', '.jpg', '.jpeg')) for fname in os.listdir(source_path)):
        print(f"❌ Lỗi: Không tìm thấy ảnh nào trong '{source_path}'.")
        print("   Vui lòng thêm ảnh của bạn vào thư mục này rồi chạy lại script.")
        return
        
    print(f"🔎 Đang tải mô hình từ: {weights_path}")
    model = YOLO(weights_path)
    
    print(f"🚀 Bắt đầu nhận diện trên các ảnh trong: {source_path}")
    results = model.predict(
        source=source_path,
        conf=0.25,      # Ngưỡng tin cậy (có thể điều chỉnh)
        save=True,      # Lưu lại ảnh kết quả với các khung bao
        save_txt=True   # Lưu kết quả dưới dạng file text
    )
    
    # predict() là một generator, cần duyệt qua để kích hoạt việc lưu file
    for _ in results:
        pass
        
    print("\\n✅ Quá trình nhận diện hoàn tất!")
    
    # Tìm thư mục predict mới nhất để thông báo cho người dùng
    predict_dirs = glob.glob('runs/detect/predict*')
    latest_predict_dir = max(predict_dirs, key=os.path.getmtime)
    print(f"📂 Kết quả đã được lưu tại: {os.path.abspath(latest_predict_dir)}")


def find_latest_weights():
    """
    Tìm đường dẫn đến file best.pt từ lần huấn luyện gần nhất.
    """
    runs_dir = 'runs/detect'
    if not os.path.exists(runs_dir):
        return None
        
    train_dirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if d.startswith('yolov8_traffic_sign_training')]
    
    if not train_dirs:
        return None
        
    latest_run_dir = max(train_dirs, key=os.path.getmtime)
    weights_path = os.path.join(latest_run_dir, 'weights/best.pt')
    
    if os.path.exists(weights_path):
        return weights_path
    else:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chạy nhận diện YOLOv8 trên ảnh tùy chỉnh.")
    parser.add_argument('--weights', type=str, default=None, help='Đường dẫn đến file trọng số (ví dụ: runs/detect/train/weights/best.pt). Nếu bỏ trống, sẽ tự động tìm file mới nhất.')
    parser.add_argument('--source', type=str, default='custom_images', help='Đường dẫn đến thư mục chứa ảnh của bạn.')
    args = parser.parse_args()

    weights = args.weights
    if weights is None:
        print("🤔 Không có đường dẫn trọng số. Đang tìm mô hình được huấn luyện gần nhất...")
        weights = find_latest_weights()

    if not weights:
        print("❌ Lỗi: Không tìm thấy trọng số đã huấn luyện. Vui lòng chạy train.py trước hoặc cung cấp đường dẫn bằng tham số --weights.")
    else:
        predict(weights, args.source) 