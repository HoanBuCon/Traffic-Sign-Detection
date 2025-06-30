import os
import cv2
import time
from ultralytics import YOLO
import numpy as np
import yaml
from config import Config
from utils import ImageEnhancer, VisualizationUtils
import unicodedata

# Thêm ánh xạ class -> mô tả tiếng Việt đầy đủ
class_vi_map = {
    "W.224": "Đường hẹp cả hai phía",
    "W.205c": "Đường bị hẹp về phía bên trái",
    "P.102": "Cấm đi ngược chiều",
    "R.302a": "Hướng phải đi thẳng",
    "W.205a": "Đường bị hẹp về phía bên phải",
    "W.207": "Đường giao nhau",
    "W.201a": "Chỗ ngoặt nguy hiểm vòng bên trái",
    "P.123a": "Cấm rẽ trái",
    "I.434a": "Biển tên đường (kiểu 1)",
    "R.303": "Hướng phải rẽ",
    "P.130": "Cấm dừng xe và đỗ xe",
    "I.409": "Chỉ dẫn khu công nghiệp",
    "R.415a": "Khu vực cấm đỗ xe",
    "W.245a": "Đường có ổ gà, lồi lõm",
    "P.106a*Xe tải": "Cấm xe tải",
    "W.203c": "Đường người đi bộ cắt ngang",
    "P.117*": "Cấm xe đạp",
    "P.124a*": "Cấm xe máy",
    "P.107": "Cấm quay đầu xe",
    "P.124d": "Cấm xe khách và xe tải",
    "P.103a": "Cấm ô tô",
    "W.203b": "Đường người đi bộ cắt ngang",
    "W.221b": "Đường có vật cản",
    "P.111": "Cấm vượt",
    "P.129": "Cấm bóp còi",
    "S.505a*Xe_may": "Biển phụ áp dụng cho xe máy",
    "W.246a": "Nguy hiểm khác",
    "W.225": "Đường trơn",
    "S.505a*Xe_tai_va_cong": "Biển phụ áp dụng cho xe tải và container",
    "P.104": "Cấm xe kéo",
    "S.505a*Xe_tai": "Biển phụ áp dụng cho xe tải",
    "Camera": "Camera giao thông",
    "P.123b": "Cấm rẽ phải",
    "W.202b": "Chỗ ngoặt nguy hiểm vòng bên phải",
    "B.8a": "Biển báo chỉ hướng",
    "P.137": "Hạn chế chiều cao",
    "P.139": "Hạn chế chiều rộng",
    "W.205b": "Đường bị hẹp cả hai phía",
    "P.127*50": "Giới hạn tốc độ tối đa 50 km/h",
    "P.127*60": "Giới hạn tốc độ tối đa 60 km/h",
    "P.127*80": "Giới hạn tốc độ tối đa 80 km/h",
    "P.127*40": "Giới hạn tốc độ tối đa 40 km/h",
    "R.301e": "Hướng đi ưu tiên",
    "W.239b*": "Nguy hiểm do súc vật",
    "W.233": "Gió ngang",
    "I.407a": "Chỉ dẫn giao lộ",
    "P.131a": "Cấm đỗ xe",
    "P.124b1": "Cấm xe tải",
    "W.210": "Giao nhau với đường sắt có rào chắn",
    "P.124c": "Cấm xe mô tô ba bánh",
    "W.201b": "Chỗ ngoặt nguy hiểm vòng bên phải",
    "W.246c": "Chú ý chướng ngại vật",
    "DP.135": "Hết hạn chế tốc độ",
    "P.103b": "Cấm ô tô khách",
    "P.103c": "Cấm ô tô tải",
    "P.106a": "Cấm xe tải",
    "P.106b": "Cấm xe tải trên 2,5 tấn",
    "P.107a": "Cấm quay đầu xe (trừ xe máy và xe đạp)",
    "P.112": "Cấm xe kéo moóc",
    "P.115": "Cấm xe người kéo",
    "P.117": "Cấm xe đạp",
    "P.124a": "Cấm xe máy",
    "P.124b": "Cấm xe mô tô",
    "P.125": "Cấm xe công nông",
    "P.127": "Giới hạn tốc độ tối đa",
    "P.128": "Cấm sử dụng đèn chiếu xa",
    "P.245a": "Cấm dừng xe",
    "R.301a": "Đường ưu tiên",
    "R.301c": "Hướng đi phải theo",
    "R.301d": "Hướng đi phải theo (bên phải)",
    "R.302b": "Phải đi vòng chướng ngại vật",
    "R.407a": "Đường một chiều",
    "R.409": "Chỗ quay xe",
    "R.425": "Chỉ dẫn bệnh viện",
    "R.434": "Bến xe buýt",
    "S.509a": "Thuyết minh biển chính",
    "W.202a": "Đường ngoặt liên tiếp",
    "W.205d": "Đường giao nhau",
    "W.207a": "Giao nhau với đường không ưu tiên (bên trái và bên phải)",
    "W.207b": "Giao nhau với đường không ưu tiên (bên phải)",
    "W.207c": "Giao nhau với đường không ưu tiên (bên trái)",
    "W.208": "Giao nhau với đường ưu tiên",
    "W.209": "Giao nhau có tín hiệu đèn",
    "W.219": "dốc xuống nguy hiểm",
    "W.227": "Báo hiệu công trường",
    "W.235": "Đường đôi"
}

# Đọc danh sách class đúng thứ tự từ data.yaml
with open('data.yaml', 'r', encoding='utf-8') as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml['names']

# descriptions_vi mới: đúng thứ tự class_names, ưu tiên giữ mô tả cũ nếu có, bổ sung nghĩa mới nếu chưa có, nếu vẫn chưa có thì để 'Chưa có mô tả'
old_descriptions = [
    "Đường người đi bộ cắt ngang",
    "Đường giao nhau (ngã ba bên phải)",
    "Cấm đi ngược chiều",
    "Phải đi vòng sang bên phải",
    "Giao nhau với đường đồng cấp",
    "Giao nhau với đường không ưu tiên",
    "Chỗ ngoặt nguy hiểm vòng bên trái",
    "Cấm rẽ trái",
    "Bến xe buýt",
    "Nơi giao nhau chạy theo vòng xuyến",
    "Cấm dừng và đỗ xe",
    "Chỗ quay xe",
    "Biển gộp làn đường theo phương tiện",
    "Đi chậm",
    "Cấm xe tải",
    "Đường bị thu hẹp về phía phải",
    "Giới hạn chiều cao",
    "Cấm quay đầu",
    "Cấm ô tô khách và ô tô tải",
    "Cấm rẽ phải và quay đầu",
    "Cấm ô tô",
    "Đường bị thu hẹp về phía trái",
    "Gồ giảm tốc phía trước",
    "Cấm xe hai và ba bánh",
    "Kiểm tra",
    "Chỉ dành cho xe máy*",
    "Chướng ngoại vật phía trước",
    "Trẻ em",
    "Xe tải và xe công*",
    "Cấm mô tô và xe máy",
    "Chỉ dành cho xe tải*",
    "Đường có camera giám sát",
    "Cấm rẽ phải",
    "Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải",
    "Cấm xe sơ-mi rơ-moóc",
    "Cấm rẽ trái và phải",
    "Cấm đi thẳng và rẽ phải",
    "Đường giao nhau (ngã ba bên trái)",
    "Giới hạn tốc độ (50km/h)",
    "Giới hạn tốc độ (60km/h)",
    "Giới hạn tốc độ (80km/h)",
    "Giới hạn tốc độ (40km/h)",
    "Các xe chỉ được rẽ trái",
    "Chiều cao tĩnh không thực tế",
    "Nguy hiểm khác",
    "Đường một chiều",
    "Cấm đỗ xe",
    "Cấm ô tô quay đầu xe (được rẽ trái)",
    "Giao nhau với đường sắt có rào chắn",
    "Cấm rẽ trái và quay đầu xe",
    "Chỗ ngoặt nguy hiểm vòng bên phải",
    "Chú ý chướng ngại vật – vòng tránh sang bên phải"
]
descriptions_vi = []
for i, class_name in enumerate(class_names):
    if i < len(old_descriptions):
        descriptions_vi.append(old_descriptions[i])
    elif class_name in class_vi_map:
        descriptions_vi.append(class_vi_map[class_name])
    else:
        descriptions_vi.append("Chưa có mô tả")

def remove_vietnamese_diacritics(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
    text = text.replace('đ', 'd').replace('Đ', 'D')
    text = text.replace('–', '-')
    text = text.replace(' ', '_')
    text = text.replace('*', '')
    text = text.replace('(', '').replace(')', '')
    text = text.replace('/', '_')
    text = text.replace(',', '')
    text = text.replace('.', '')
    text = text.replace('-', '_')
    text = text.replace('__', '_')
    return text

descriptions_vi_no_diacritics = [remove_vietnamese_diacritics(desc) for desc in descriptions_vi]

class RealTimeTrafficSignDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            # Tìm model mới nhất trong all_weight
            all_weight_dir = 'all_weight'
            train_dirs = [d for d in os.listdir(all_weight_dir) if d.startswith('train') and os.path.isdir(os.path.join(all_weight_dir, d))]
            if not train_dirs:
                raise FileNotFoundError("Không tìm thấy model đã train trong all_weight! Hãy train model trước.")
            train_dirs_sorted = sorted(train_dirs, key=lambda x: int(x.replace('train', '')) if x.replace('train', '').isdigit() else 0)
            latest_train_dir = os.path.join(all_weight_dir, train_dirs_sorted[-1])
            best_pt_path = os.path.join(latest_train_dir, 'best.pt')
            if not os.path.exists(best_pt_path):
                raise FileNotFoundError(f"Không tìm thấy best.pt trong {latest_train_dir}!")
            model_path = best_pt_path
            print(f"[INFO] Sử dụng model: {model_path}")
        self.model = YOLO(model_path)
        self.image_enhancer = ImageEnhancer()
        self.config = Config
        # Đọc class_names từ data.yaml
        with open('data.yaml', 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
            self.class_names = data_yaml.get('names', {})

    def enhance_image_for_inference(self, image: np.ndarray) -> np.ndarray:
        if not self.config.ENABLE_IMAGE_ENHANCEMENT:
            return image
        image = self.image_enhancer.denoise_image(image)
        image = self.image_enhancer.sharpen_image(image)
        image = self.image_enhancer.enhance_image(image, enhancement_level=self.config.CONTRAST_ENHANCEMENT)
        image = self.image_enhancer.adjust_gamma(image, gamma=self.config.GAMMA_CORRECTION)
        return image

    def predict_frame(self, frame: np.ndarray):
        enhanced = self.enhance_image_for_inference(frame)
        results = self.model.predict(
            enhanced,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.NMS_THRESHOLD,
            max_det=self.config.MAX_DETECTIONS,
            verbose=False
        )
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_idx = int(box.cls)
                class_label = self.class_names[class_idx] if isinstance(self.class_names, list) and class_idx < len(self.class_names) else str(class_idx)
                class_label_vi = descriptions_vi_no_diacritics[class_idx] if class_idx < len(descriptions_vi_no_diacritics) else class_label
                detection = {
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf),
                    'class_id': class_idx,
                    'class_label': class_label,
                    'class_label_vi': class_label_vi
                }
                detections.append(detection)
        return detections

    def draw_and_show(self, frame: np.ndarray, detections):
        # Vẽ kết quả lên ảnh
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['class_label']} | {det['class_label_vi']} | {det['confidence']:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow('Traffic Sign Detection - Press q to quit', frame)

    def run_webcam(self, cam_id=0, save_video=True):
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print("Không mở được camera!")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 20
        out = None
        if save_video:
            output_dir = 'real_time_output'
            os.makedirs(output_dir, exist_ok=True)
            video_filename = os.path.join(output_dir, f"result_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            print(f"[INFO] Video sẽ được lưu tại: {video_filename}")
        print("Nhấn 'q' để thoát.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không lấy được frame từ camera!")
                break
            detections = self.predict_frame(frame)
            frame_draw = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                label = f"{det['class_label']} | {det['class_label_vi']} | {det['confidence']:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_draw, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow('Traffic Sign Detection - Press q to quit', frame_draw)
            if out is not None:
                out.write(frame_draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        if out is not None:
            print(f"[INFO] Video đã lưu tại: {video_filename}")

def main():
    detector = RealTimeTrafficSignDetector()
    detector.run_webcam(save_video=True)

if __name__ == "__main__":
    main() 