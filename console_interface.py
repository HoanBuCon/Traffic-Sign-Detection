#!/usr/bin/env python3
"""
Console Interface cho Traffic Sign Detection
Giao diện console để chạy các thao tác predict
"""

import os
import sys
import subprocess
import time
from traffic_sign_names import print_traffic_sign_info

class TrafficSignConsole:
    def __init__(self):
        self.running = True
        
    def clear_screen(self):
        """Xóa màn hình console"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """In header của ứng dụng"""
        print("=" * 60)
        print("🚦 TRAFFIC SIGN DETECTION - CONSOLE INTERFACE")
        print("=" * 60)
        print("Hệ thống nhận diện biển báo giao thông với 12 loại biển báo")
        print("=" * 60)
    
    def print_menu(self):
        """In menu chính"""
        print("\n📋 MENU CHÍNH:")
        print("1. 🖼️  Nhận diện batch (predict.py)")
        print("2. 📹 Nhận diện real-time (predict2.py)")
        print("3. 📊 Xem thông tin biển báo")
        print("4. 📁 Kiểm tra thư mục input/output")
        print("5. 🔧 Cài đặt hệ thống")
        print("6. 📖 Hướng dẫn sử dụng")
        print("0. ❌ Thoát")
        print("-" * 60)
    
    def check_input_folder(self):
        """Kiểm tra thư mục input"""
        input_dir = 'input'
        if not os.path.exists(input_dir):
            print(f"❌ Thư mục '{input_dir}' không tồn tại!")
            return False
        
        # Đếm số file ảnh
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        if not image_files:
            print(f"❌ Không tìm thấy file ảnh nào trong thư mục '{input_dir}'!")
            print("💡 Hãy đặt ảnh cần nhận diện vào thư mục 'input'")
            return False
        
        print(f"✅ Tìm thấy {len(image_files)} file ảnh trong thư mục 'input'")
        return True
    
    def check_output_folder(self):
        """Kiểm tra thư mục output"""
        output_dir = 'output'
        if not os.path.exists(output_dir):
            print(f"📁 Tạo thư mục '{output_dir}'...")
            os.makedirs(output_dir, exist_ok=True)
            print(f"✅ Đã tạo thư mục '{output_dir}'")
        else:
            print(f"✅ Thư mục '{output_dir}' đã tồn tại")
    
    def run_batch_prediction(self):
        """Chạy nhận diện batch"""
        print("\n🖼️  NHẬN DIỆN BATCH (predict.py)")
        print("-" * 40)
        
        # Kiểm tra thư mục input
        if not self.check_input_folder():
            input("\nNhấn Enter để quay lại menu...")
            return
        
        # Kiểm tra thư mục output
        self.check_output_folder()
        
        print("\n🚀 Bắt đầu nhận diện batch...")
        print("⏳ Đang chạy predict.py...")
        
        try:
            # Chạy predict.py
            result = subprocess.run([sys.executable, 'predict.py'], 
                                  capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                print("✅ Nhận diện batch hoàn thành thành công!")
                print("\n📋 Kết quả:")
                print(result.stdout)
            else:
                print("❌ Lỗi khi chạy predict.py:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")
        
        input("\nNhấn Enter để quay lại menu...")
    
    def run_realtime_prediction(self):
        """Chạy nhận diện real-time"""
        print("\n📹 NHẬN DIỆN REAL-TIME (predict2.py)")
        print("-" * 40)
        
        print("⚠️  Lưu ý:")
        print("- Cần có webcam để chạy real-time")
        print("- Nhấn 'q' để thoát khỏi chế độ real-time")
        print("- Video sẽ được lưu trong thư mục 'real_time_output'")
        
        choice = input("\nBạn có muốn tiếp tục? (y/n): ").lower()
        if choice != 'y':
            return
        
        print("\n🚀 Bắt đầu nhận diện real-time...")
        print("⏳ Đang chạy predict2.py...")
        print("💡 Nhấn 'q' để thoát")
        
        try:
            # Chạy predict2.py
            subprocess.run([sys.executable, 'predict2.py'])
        except KeyboardInterrupt:
            print("\n⏹️  Đã dừng nhận diện real-time")
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")
        
        input("\nNhấn Enter để quay lại menu...")
    
    def show_traffic_sign_info(self):
        """Hiển thị thông tin biển báo"""
        print("\n📊 THÔNG TIN BIỂN BÁO")
        print("-" * 40)
        print_traffic_sign_info()
        input("\nNhấn Enter để quay lại menu...")
    
    def check_folders(self):
        """Kiểm tra các thư mục"""
        print("\n📁 KIỂM TRA THƯ MỤC")
        print("-" * 40)
        
        folders = {
            'input': 'Thư mục chứa ảnh cần nhận diện',
            'output': 'Thư mục lưu kết quả nhận diện',
            'real_time_output': 'Thư mục lưu video real-time',
            'training_history': 'Thư mục chứa model đã train',
            'dataset': 'Thư mục dataset'
        }
        
        for folder, description in folders.items():
            if os.path.exists(folder):
                if folder == 'input':
                    # Đếm file ảnh trong input
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
                    image_files = []
                    for file in os.listdir(folder):
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_files.append(file)
                    print(f"✅ {folder:20s} - {description} ({len(image_files)} ảnh)")
                else:
                    print(f"✅ {folder:20s} - {description}")
            else:
                print(f"❌ {folder:20s} - {description} (KHÔNG TỒN TẠI)")
        
        input("\nNhấn Enter để quay lại menu...")
    
    def system_setup(self):
        """Cài đặt hệ thống"""
        print("\n🔧 CÀI ĐẶT HỆ THỐNG")
        print("-" * 40)
        
        print("1. Tạo thư mục cần thiết")
        print("2. Kiểm tra model")
        print("3. Kiểm tra dependencies")
        print("0. Quay lại")
        
        choice = input("\nChọn tùy chọn: ")
        
        if choice == '1':
            self.create_folders()
        elif choice == '2':
            self.check_model()
        elif choice == '3':
            self.check_dependencies()
    
    def create_folders(self):
        """Tạo các thư mục cần thiết"""
        print("\n📁 Tạo thư mục cần thiết...")
        
        folders = ['input', 'output', 'real_time_output']
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
                print(f"✅ Đã tạo thư mục: {folder}")
            else:
                print(f"✅ Thư mục {folder} đã tồn tại")
        
        input("\nNhấn Enter để tiếp tục...")
    
    def check_model(self):
        """Kiểm tra model"""
        print("\n🤖 Kiểm tra model...")
        
        model_path = 'training_history/train2/weights/best.pt'
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✅ Model tồn tại: {model_path}")
            print(f"📊 Kích thước: {size_mb:.1f} MB")
        else:
            print(f"❌ Không tìm thấy model: {model_path}")
            print("💡 Hãy train model trước khi sử dụng")
        
        input("\nNhấn Enter để tiếp tục...")
    
    def check_dependencies(self):
        """Kiểm tra dependencies"""
        print("\n📦 Kiểm tra dependencies...")
        
        try:
            import cv2
            print("✅ OpenCV (cv2)")
        except ImportError:
            print("❌ OpenCV (cv2) - Cần cài đặt: pip install opencv-python")
        
        try:
            from ultralytics import YOLO
            print("✅ Ultralytics (YOLO)")
        except ImportError:
            print("❌ Ultralytics - Cần cài đặt: pip install ultralytics")
        
        try:
            import yaml
            print("✅ PyYAML")
        except ImportError:
            print("❌ PyYAML - Cần cài đặt: pip install pyyaml")
        
        try:
            import numpy as np
            print("✅ NumPy")
        except ImportError:
            print("❌ NumPy - Cần cài đặt: pip install numpy")
        
        input("\nNhấn Enter để tiếp tục...")
    
    def show_help(self):
        """Hiển thị hướng dẫn sử dụng"""
        print("\n📖 HƯỚNG DẪN SỬ DỤNG")
        print("-" * 40)
        
        print("🚦 TRAFFIC SIGN DETECTION - HƯỚNG DẪN")
        print()
        print("1. NHẬN DIỆN BATCH (predict.py):")
        print("   - Đặt ảnh cần nhận diện vào thư mục 'input'")
        print("   - Chạy tùy chọn 1 từ menu")
        print("   - Kết quả sẽ được lưu trong thư mục 'output'")
        print()
        print("2. NHẬN DIỆN REAL-TIME (predict2.py):")
        print("   - Cần có webcam")
        print("   - Chạy tùy chọn 2 từ menu")
        print("   - Nhấn 'q' để thoát")
        print("   - Video sẽ được lưu trong 'real_time_output'")
        print()
        print("3. CÁC LOẠI BIỂN BÁO ĐƯỢC NHẬN DIỆN:")
        print("   - 4 biển báo cấm (P.102, P.106.b, P.130, P.131.a)")
        print("   - 5 biển báo nguy hiểm (W.201.a, W.203.c, W.207.b, W.207.c, W.209)")
        print("   - 1 biển báo chỉ dẫn (I.423.b)")
        print("   - 1 biển báo hiệu lệnh (R.308.b)")
        print("   - 1 biển báo đặc biệt (SUS)")
        print()
        print("4. CẤU TRÚC THƯ MỤC:")
        print("   input/           - Ảnh cần nhận diện")
        print("   output/          - Kết quả nhận diện batch")
        print("   real_time_output/ - Video real-time")
        print("   training_history/ - Model đã train")
        print("   dataset/         - Dataset training")
        
        input("\nNhấn Enter để quay lại menu...")
    
    def run(self):
        """Chạy giao diện console"""
        while self.running:
            self.clear_screen()
            self.print_header()
            self.print_menu()
            
            try:
                choice = input("Chọn tùy chọn (0-6): ").strip()
                
                if choice == '0':
                    print("\n👋 Tạm biệt! Cảm ơn bạn đã sử dụng Traffic Sign Detection!")
                    self.running = False
                elif choice == '1':
                    self.run_batch_prediction()
                elif choice == '2':
                    self.run_realtime_prediction()
                elif choice == '3':
                    self.show_traffic_sign_info()
                elif choice == '4':
                    self.check_folders()
                elif choice == '5':
                    self.system_setup()
                elif choice == '6':
                    self.show_help()
                else:
                    print("❌ Lựa chọn không hợp lệ! Vui lòng chọn từ 0-6.")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Tạm biệt! Cảm ơn bạn đã sử dụng Traffic Sign Detection!")
                self.running = False
            except Exception as e:
                print(f"\n❌ Lỗi: {str(e)}")
                input("Nhấn Enter để tiếp tục...")

def main():
    """Hàm chính"""
    console = TrafficSignConsole()
    console.run()

if __name__ == "__main__":
    main() 