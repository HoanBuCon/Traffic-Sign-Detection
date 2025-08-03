#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phiên bản không GUI của Traffic Sign Detection với Hybrid CV+NLP
Sử dụng khi gặp lỗi OpenCV GUI trên Windows
"""

import sys
import os

# Import main detector
from real_time_predict_nlp_hybrid import RealTimeTrafficSignDetectorNLPHybrid

def main():
    print("[DEBUG] Khởi động chương trình Hybrid CV+NLP (No GUI)")
    
    try:
        detector = RealTimeTrafficSignDetectorNLPHybrid()
        print("[DEBUG] Đã tạo detector Hybrid")
        
        print("\n" + "="*50)
        print("TRAFFIC SIGN DETECTION - HYBRID CV+NLP")
        print("Version: No GUI (Headless mode)")
        print("="*50)
        print("[INFO] Chương trình chạy mà không hiển thị cửa sổ")
        print("[INFO] Video sẽ được lưu trong thư mục 'real_time_output'")
        print("[INFO] Nhấn Ctrl+C để dừng chương trình")
        print("="*50 + "\n")
        
        # Chạy detector mà không hiển thị GUI
        detector.run_webcam(save_video=True, show_gui=False)
        
    except KeyboardInterrupt:
        print("\n[INFO] Chương trình đã được dừng bởi người dùng")
    except Exception as e:
        print(f"[ERROR] Lỗi trong quá trình chạy: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[DEBUG] Kết thúc chương trình Hybrid (No GUI)")

if __name__ == "__main__":
    main()
