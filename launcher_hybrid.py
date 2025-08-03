#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launcher cho Traffic Sign Detection với Hybrid CV+NLP
Cho phép người dùng chọn chế độ GUI hoặc No GUI
"""

import sys
import os
from real_time_predict_nlp_hybrid import RealTimeTrafficSignDetectorNLPHybrid

def test_opencv_gui():
    """Test xem OpenCV GUI có hoạt động không"""
    try:
        import cv2
        import numpy as np
        
        # Tạo một ảnh test nhỏ
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(test_img, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Thử hiển thị
        cv2.imshow("GUI Test", test_img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        
        return True
    except Exception as e:
        print(f"[WARNING] OpenCV GUI không khả dụng: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("TRAFFIC SIGN DETECTION - HYBRID CV+NLP LAUNCHER")
    print("="*60)
    
    # Test OpenCV GUI
    gui_available = test_opencv_gui()
    
    if gui_available:
        print("[INFO] OpenCV GUI khả dụng")
        print("\nChọn chế độ chạy:")
        print("1. Với GUI (hiển thị cửa sổ video)")
        print("2. Không GUI (chỉ lưu video)")
        print("3. Tự động (khuyến nghị)")
    else:
        print("[WARNING] OpenCV GUI không khả dụng - sẽ chạy chế độ No GUI")
        mode = "2"
    
    if gui_available:
        while True:
            try:
                choice = input("\nNhập lựa chọn (1/2/3): ").strip()
                if choice in ["1", "2", "3"]:
                    mode = choice
                    break
                else:
                    print("Vui lòng nhập 1, 2 hoặc 3")
            except KeyboardInterrupt:
                print("\n[INFO] Thoát chương trình")
                return
    
    # Xác định chế độ chạy
    if mode == "1":
        show_gui = True
        print("\n[INFO] Chạy với GUI")
    elif mode == "2":
        show_gui = False
        print("\n[INFO] Chạy không GUI")
    else:  # mode == "3"
        show_gui = gui_available
        print(f"\n[INFO] Chạy tự động - GUI: {'Có' if show_gui else 'Không'}")
    
    print("="*60)
    
    try:
        print("[DEBUG] Khởi động detector...")
        detector = RealTimeTrafficSignDetectorNLPHybrid()
        print("[DEBUG] Đã tạo detector thành công")
        
        if show_gui:
            print("[INFO] Nhấn 'q' trong cửa sổ video để thoát")
        else:
            print("[INFO] Nhấn Ctrl+C để dừng chương trình")
            print("[INFO] Video sẽ được lưu trong thư mục 'real_time_output'")
        
        print("[INFO] Bắt đầu phát hiện biển báo...")
        detector.run_webcam(save_video=True, show_gui=show_gui)
        
    except KeyboardInterrupt:
        print("\n[INFO] Chương trình đã được dừng bởi người dùng")
    except Exception as e:
        print(f"[ERROR] Lỗi trong quá trình chạy: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[DEBUG] Kết thúc chương trình")

if __name__ == "__main__":
    main()
