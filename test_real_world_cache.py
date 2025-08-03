"""
Test script để kiểm tra hiệu quả cache system trong môi trường thực tế
"""
import cv2
import numpy as np
import time
from real_time_predict_nlp_thread import RealTimeTrafficSignDetectorNLPThread

class CachePerformanceMonitor:
    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.ocr_requests = 0
        self.start_time = time.time()
        
    def log_cache_hit(self):
        self.cache_hits += 1
        
    def log_cache_miss(self):
        self.cache_misses += 1
        
    def log_ocr_request(self):
        self.ocr_requests += 1
        
    def get_stats(self):
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        runtime = time.time() - self.start_time
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'ocr_requests': self.ocr_requests,
            'hit_rate': hit_rate,
            'runtime': runtime,
            'requests_per_second': total_requests / runtime if runtime > 0 else 0
        }

class OptimizedDetector(RealTimeTrafficSignDetectorNLPThread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = CachePerformanceMonitor()
        
    def _is_cache_valid(self, object_id, current_bbox, image_hash):
        # Override để thêm monitoring
        is_valid = super()._is_cache_valid(object_id, current_bbox, image_hash)
        
        if is_valid and object_id in self.ocr_cache:
            self.monitor.log_cache_hit()
            print(f"[MONITOR] Cache HIT for object {object_id}")
        else:
            self.monitor.log_cache_miss()
            print(f"[MONITOR] Cache MISS for object {object_id}")
            
        return is_valid
    
    def ocr_worker(self):
        # Override để thêm monitoring
        while True:
            object_id, sign_crop = self.ocr_queue.get()
            try:
                self.monitor.log_ocr_request()
                print(f"[MONITOR] OCR Request #{self.monitor.ocr_requests} for object {object_id}")
                
                # Gọi method gốc
                image_hash = self._compute_image_hash(sign_crop)
                
                if object_id in self.ocr_image_hash_cache:
                    cached_hash = self.ocr_image_hash_cache[object_id]
                    if cached_hash != image_hash:
                        print(f"[MONITOR] Hash mismatch detected, skipping OCR")
                        continue
                
                ocr_text = self._get_text_from_sign(sign_crop)
                if ocr_text and len(ocr_text.strip()) > 0:
                    self.ocr_cache[object_id] = ocr_text
                    print(f"[MONITOR] OCR completed for object {object_id}: {ocr_text}")
                else:
                    print(f"[MONITOR] Empty OCR result for object {object_id}")
            except Exception as e:
                print(f"[ERROR] OCR thread: {e}")
            finally:
                self.ocr_queue.task_done()
    
    def print_performance_stats(self):
        stats = self.monitor.get_stats()
        print(f"\n=== CACHE PERFORMANCE STATS ===")
        print(f"Runtime: {stats['runtime']:.1f}s")
        print(f"Cache Hits: {stats['cache_hits']}")
        print(f"Cache Misses: {stats['cache_misses']}")
        print(f"OCR Requests: {stats['ocr_requests']}")
        print(f"Cache Hit Rate: {stats['hit_rate']:.1f}%")
        print(f"Requests/Second: {stats['requests_per_second']:.1f}")
        print(f"================================\n")

def test_with_video_file(video_path):
    """Test với video file thay vì webcam"""
    print("=== Testing Cache System with Video ===")
    
    detector = OptimizedDetector()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return
    
    frame_count = 0
    stats_interval = 50  # Print stats every 50 frames
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process frame
            detections = detector.predict_frame(frame)
            
            # Print stats periodically
            if frame_count % stats_interval == 0:
                detector.print_performance_stats()
            
            # Show frame (optional)
            if frame_count % 5 == 0:  # Show every 5th frame
                detector.draw_and_show(frame.copy(), detections)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        print(f"\n=== FINAL PERFORMANCE REPORT ===")
        print(f"Total Frames Processed: {frame_count}")
        detector.print_performance_stats()

def test_with_webcam():
    """Test với webcam trực tiếp"""
    print("=== Testing Cache System with Webcam ===")
    print("Press 'q' to quit, 's' to show stats")
    
    detector = OptimizedDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Không thể mở webcam!")
        return
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process frame
            detections = detector.predict_frame(frame)
            detector.draw_and_show(frame.copy(), detections)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                detector.print_performance_stats()
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        print(f"\n=== FINAL PERFORMANCE REPORT ===")
        print(f"Total Frames Processed: {frame_count}")
        detector.print_performance_stats()

def create_test_scenario():
    """Tạo scenario test với các ảnh tĩnh"""
    print("=== Testing Cache System with Static Images ===")
    
    detector = OptimizedDetector()
    
    # Tạo frame 640x480 với biển báo giả
    def create_frame_with_sign(sign_text, position=(100, 100), sign_size=(200, 100)):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background
        
        # Tạo biển báo
        x, y = position
        w, h = sign_size
        
        # Background của biển báo
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
        
        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(sign_text, font, 0.7, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(frame, sign_text, (text_x, text_y), font, 0.7, (0, 0, 0), 2)
        
        return frame
    
    # Test scenarios
    scenarios = [
        # Same sign, multiple frames
        ("Same Sign Frame 1", create_frame_with_sign("Quan 1 - 2km")),
        ("Same Sign Frame 2", create_frame_with_sign("Quan 1 - 2km")),
        ("Same Sign Frame 3", create_frame_with_sign("Quan 1 - 2km")),
        
        # Different sign
        ("Different Sign", create_frame_with_sign("Quan 3 - 5km")),
        ("Different Sign Again", create_frame_with_sign("Quan 3 - 5km")),
        
        # Back to first sign
        ("Back to First", create_frame_with_sign("Quan 1 - 2km")),
        
        # New position
        ("New Position", create_frame_with_sign("Quan 7 - 3km", position=(300, 200))),
    ]
    
    for scenario_name, frame in scenarios:
        print(f"\nProcessing: {scenario_name}")
        detections = detector.predict_frame(frame)
        
        # Show frame
        detector.draw_and_show(frame.copy(), detections)
        cv2.waitKey(1000)  # Wait 1 second
        
        detector.print_performance_stats()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Test with static images scenario")
    print("2. Test with webcam")
    print("3. Test with video file")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        create_test_scenario()
    elif choice == "2":
        test_with_webcam()
    elif choice == "3":
        video_path = input("Enter video file path: ").strip()
        test_with_video_file(video_path)
    else:
        print("Invalid choice. Running static scenario...")
        create_test_scenario()
