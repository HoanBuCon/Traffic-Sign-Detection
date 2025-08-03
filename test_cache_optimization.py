"""
Test script để kiểm tra hiệu quả của hệ thống cache OCR được tối ưu hóa
"""
import cv2
import numpy as np
import time
import hashlib
from real_time_predict_nlp_thread import RealTimeTrafficSignDetectorNLPThread

def create_fake_sign_image(text_content="TEST SIGN", size=(100, 50)):
    """Tạo ảnh biển báo giả để test"""
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    cv2.putText(img, text_content, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img

def test_cache_efficiency():
    """Test hiệu quả cache system"""
    print("=== Test Cache System Optimization ===")
    
    # Khởi tạo detector
    detector = RealTimeTrafficSignDetectorNLPThread()
    
    # Tạo các ảnh test khác nhau
    sign1 = create_fake_sign_image("Cho Xa La 2km", (150, 80))
    sign2 = create_fake_sign_image("Quang Trung 5km", (150, 80))
    sign3 = create_fake_sign_image("Hai Ba Trung 3km", (150, 80))
    
    # Test 1: Cache hit với cùng nội dung
    print("\n--- Test 1: Cache Hit ---")
    hash1_a = detector._compute_image_hash(sign1)
    hash1_b = detector._compute_image_hash(sign1)
    print(f"Hash ảnh 1 lần 1: {hash1_a}")
    print(f"Hash ảnh 1 lần 2: {hash1_b}")
    print(f"Cache hit (same content): {hash1_a == hash1_b}")
    
    # Test 2: Cache miss với nội dung khác nhau
    print("\n--- Test 2: Cache Miss ---")
    hash2 = detector._compute_image_hash(sign2)
    hash3 = detector._compute_image_hash(sign3)
    print(f"Hash ảnh 2: {hash2}")
    print(f"Hash ảnh 3: {hash3}")
    print(f"Different content detected: {hash1_a != hash2 != hash3}")
    
    # Test 3: Kiểm tra cache validation
    print("\n--- Test 3: Cache Validation ---")
    object_id = 100
    bbox1 = [10, 10, 160, 90]
    bbox2 = [15, 15, 165, 95]  # Slight change
    bbox3 = [50, 50, 200, 130]  # Significant change
    
    # Cập nhật cache cho object
    detector._update_object_cache(object_id, bbox1, hash1_a)
    
    # Test validation với bbox thay đổi nhẹ
    valid1 = detector._is_cache_valid(object_id, bbox2, hash1_a)
    print(f"Cache valid with slight bbox change: {valid1}")
    
    # Test validation với bbox thay đổi lớn
    valid2 = detector._is_cache_valid(object_id, bbox3, hash1_a)
    print(f"Cache valid with significant bbox change: {valid2}")
    
    # Test validation với nội dung khác
    valid3 = detector._is_cache_valid(object_id, bbox1, hash2)
    print(f"Cache valid with different content: {valid3}")
    
    # Test 4: Cache timeout
    print("\n--- Test 4: Cache Timeout ---")
    # Giảm timeout xuống 2 giây để test
    detector.ocr_cache_timeout = 2.0
    detector._update_object_cache(object_id, bbox1, hash1_a)
    
    print("Waiting 3 seconds for cache timeout...")
    time.sleep(3)
    
    valid_after_timeout = detector._is_cache_valid(object_id, bbox1, hash1_a)
    print(f"Cache valid after timeout: {valid_after_timeout}")
    
    # Test 5: Performance comparison
    print("\n--- Test 5: Performance Comparison ---")
    num_operations = 1000
    
    # Test performance của hash computation
    start_time = time.time()
    for _ in range(num_operations):
        detector._compute_image_hash(sign1)
    hash_time = time.time() - start_time
    print(f"Hash computation time for {num_operations} operations: {hash_time:.4f}s")
    
    # Test cache validation performance
    detector._update_object_cache(object_id, bbox1, hash1_a)
    start_time = time.time()
    for _ in range(num_operations):
        detector._is_cache_valid(object_id, bbox1, hash1_a)
    validation_time = time.time() - start_time
    print(f"Cache validation time for {num_operations} operations: {validation_time:.4f}s")
    
    print("\n=== Cache System Test Completed ===")

def test_real_scenario():
    """Test với scenario thực tế"""
    print("\n=== Real Scenario Test ===")
    
    detector = RealTimeTrafficSignDetectorNLPThread()
    
    # Simulate camera frames với các biển báo khác nhau
    frames_with_signs = [
        ("Frame 1", create_fake_sign_image("Quan 1 - 2km", (120, 60))),
        ("Frame 2", create_fake_sign_image("Quan 1 - 2km", (120, 60))),  # Same sign
        ("Frame 3", create_fake_sign_image("Quan 3 - 5km", (120, 60))),  # Different sign
        ("Frame 4", create_fake_sign_image("Quan 3 - 5km", (120, 60))),  # Same as frame 3
        ("Frame 5", create_fake_sign_image("Quan 1 - 2km", (120, 60))),  # Back to frame 1
    ]
    
    cache_hits = 0
    cache_misses = 0
    object_id = 200
    
    for frame_name, sign_image in frames_with_signs:
        bbox = [20, 20, 140, 80]
        image_hash = detector._compute_image_hash(sign_image)
        
        # Check if cache is valid
        if detector._is_cache_valid(object_id, bbox, image_hash):
            cache_hits += 1
            print(f"{frame_name}: Cache HIT - hash: {image_hash[:8]}")
        else:
            cache_misses += 1
            detector._clear_object_cache(object_id)
            detector._update_object_cache(object_id, bbox, image_hash)
            print(f"{frame_name}: Cache MISS - hash: {image_hash[:8]}")
    
    print(f"\nCache Performance:")
    print(f"Cache Hits: {cache_hits}")
    print(f"Cache Misses: {cache_misses}")
    print(f"Hit Rate: {cache_hits/(cache_hits + cache_misses)*100:.1f}%")

if __name__ == "__main__":
    test_cache_efficiency()
    test_real_scenario()
