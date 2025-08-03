#!/usr/bin/env python3
"""
Test script để kiểm tra hiển thị bounding box với format mới
"""

import cv2
import numpy as np
from utils import VisualizationUtils

def create_test_image():
    """Tạo ảnh test đơn giản"""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255  # Ảnh trắng
    return img

def create_test_detections():
    """Tạo test detections với format mới"""
    detections = [
        {
            'bbox': [100, 100, 300, 250],
            'confidence': 0.95,
            'class_id': 0,
            'class_label': 'i.423.b',
            'class_label_vi': 'Đường kẻ dành cho người đi bộ',
            'class_label_vi_filename': 'Duong_ke_danh_cho_nguoi_di_bo'
        },
        {
            'bbox': [400, 150, 600, 300],
            'confidence': 0.87,
            'class_id': 1,
            'class_label': 'p.102',
            'class_label_vi': 'Cấm đi ngược chiều',
            'class_label_vi_filename': 'Cam_di_nguoc_chieu'
        },
        {
            'bbox': [150, 350, 350, 500],
            'confidence': 0.92,
            'class_id': 7,
            'class_label': 'w.201.a',
            'class_label_vi': 'Cảnh báo khúc cua nguy hiểm bên trái',
            'class_label_vi_filename': 'Canh_bao_khuc_cua_nguy_hiem_ben_trai'
        }
    ]
    return detections

def main():
    """Test hàm visualization"""
    print("Testing new bounding box format...")
    
    # Tạo test data
    test_image = create_test_image()
    test_detections = create_test_detections()
    
    # Vẽ detections
    result_image = VisualizationUtils.draw_detections(
        test_image, 
        test_detections,
        confidence_threshold=0.5
    )
    
    # Lưu kết quả
    output_path = "test_visualization_result.jpg"
    cv2.imwrite(output_path, result_image)
    
    print(f"Test completed! Result saved to: {output_path}")
    print("\nFormat hiển thị mới:")
    for detection in test_detections:
        class_code = detection['class_label']  # Mã biển như w.201.a
        description = detection['class_label_vi_filename']
        confidence = detection['confidence']
        print(f"  {class_code} | {description} | {confidence:.1%}")
        
    print("\nSo với format cũ (ID:object_id | class_id_code | description | confidence):")
    print("- Bỏ object_id để tập trung vào thông tin biển báo")
    print("- Sử dụng mã biển từ class_names thay vì hardcode class_ids")
    print("- Format độ tin cậy thành % với 1 chữ số thập phân")

if __name__ == "__main__":
    main()
