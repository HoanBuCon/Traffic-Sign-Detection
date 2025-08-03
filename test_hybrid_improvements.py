"""
Test file để kiểm tra các cải thiện của hệ thống Hybrid CV+NLP
"""
import cv2
import numpy as np
from real_time_predict_nlp_hybrid import RealTimeTrafficSignDetectorNLPHybrid, ArrowDetector

def test_nlp_parsing():
    """Test function parsing NLP text"""
    print("=== Testing NLP Text Parsing ===")
    detector = RealTimeTrafficSignDetectorNLPHybrid()
    
    test_cases = [
        "CHỢ THÁI 3,3 Km | DÁN 1,6 Km",
        "CHỢ THÁI 3.3 Km | DÁN 1.6 Km", 
        "BỆNH VIỆN 2,5 Km | TRƯỜNG HỌC 1,2 Km",
        "3,3 Km CHỢ THÁI | 1,6 Km DÁN",
        "CHUYÊN MỘT 130 | CHUYÊN MỘT 150"
    ]
    
    for test_text in test_cases:
        print(f"\nInput: {test_text}")
        locations = detector._parse_locations_from_nlp(test_text)
        print(f"Parsed locations: {locations}")
        
        # Test fusion
        cv_directions = {"rẽ trái": 0.4, "rẽ phải": 0.35, "đi thẳng": 0.25}
        result = detector._fuse_cv_nlp_results_multi(cv_directions, 0.8, test_text)
        print(f"Fusion result: {result}")
        print("-" * 50)

def test_arrow_detection():
    """Test Arrow Detection với synthetic images"""
    print("\n=== Testing Arrow Detection ===")
    arrow_detector = ArrowDetector()
    
    # Tạo test images với mũi tên đơn giản
    test_images = []
    
    # Left arrow
    img_left = np.zeros((100, 100), dtype=np.uint8)
    pts = np.array([[70, 30], [30, 50], [70, 70]], dtype=np.int32)
    cv2.fillPoly(img_left, [pts], 255)
    test_images.append(("Left Arrow", img_left))
    
    # Right arrow  
    img_right = np.zeros((100, 100), dtype=np.uint8)
    pts = np.array([[30, 30], [70, 50], [30, 70]], dtype=np.int32)
    cv2.fillPoly(img_right, [pts], 255)
    test_images.append(("Right Arrow", img_right))
    
    # Up arrow
    img_up = np.zeros((100, 100), dtype=np.uint8)
    pts = np.array([[30, 70], [50, 30], [70, 70]], dtype=np.int32)
    cv2.fillPoly(img_up, [pts], 255)
    test_images.append(("Up Arrow", img_up))
    
    for name, img in test_images:
        print(f"\nTesting {name}:")
        directions, confidence = arrow_detector.detect_arrow_direction(img)
        print(f"Detected: {directions} (confidence: {confidence:.3f})")
        
        # Visualize
        cv2.imshow(f"Test - {name}", img)
        cv2.waitKey(1000)
    
    cv2.destroyAllWindows()

def test_direction_keywords():
    """Test direction keyword detection"""
    print("\n=== Testing Direction Keywords ===")
    detector = RealTimeTrafficSignDetectorNLPHybrid()
    
    test_texts = [
        "rẽ trái 3km đến chợ",
        "đi thẳng 5km", 
        "quay phải tại ngã tư",
        "CHỢ THÁI 3,3 Km",  # Không có direction keyword
        "turn left 2km ahead",
        "go straight for 1km"
    ]
    
    for text in test_texts:
        has_keywords = detector._has_direction_keywords(text)
        print(f"'{text}' -> Has direction keywords: {has_keywords}")

def test_improved_fusion():
    """Test improved fusion logic"""
    print("\n=== Testing Improved Fusion Logic ===")
    detector = RealTimeTrafficSignDetectorNLPHybrid()
    
    # Test case 1: Good CV + Good NLP
    cv_dirs1 = {"rẽ trái": 0.45, "rẽ phải": 0.30, "đi thẳng": 0.25}
    nlp_text1 = "CHỢ THÁI 3,3 Km | BỆNH VIỆN 1,6 Km"
    
    result1 = detector._fuse_cv_nlp_results_multi(cv_dirs1, 0.7, nlp_text1)
    print(f"Test 1 - Result: {result1}")
    
    # Test case 2: Poor CV + Good NLP with direction
    cv_dirs2 = {"rẽ trái": 0.32, "rẽ phải": 0.34, "đi thẳng": 0.34}
    nlp_text2 = "rẽ trái đến chợ 3km"
    
    result2 = detector._fuse_cv_nlp_results_multi(cv_dirs2, 0.3, nlp_text2)
    print(f"Test 2 - Result: {result2}")
    
    # Test case 3: Blurry/Invalid NLP
    cv_dirs3 = {"đi thẳng": 0.55, "rẽ trái": 0.25, "rẽ phải": 0.20}
    nlp_text3 = "The text in the image is too blurry to be read accurately."
    
    result3 = detector._fuse_cv_nlp_results_multi(cv_dirs3, 0.6, nlp_text3)
    print(f"Test 3 - Result: {result3}")

if __name__ == "__main__":
    print("Testing Hybrid CV+NLP Improvements...")
    print("=" * 60)
    
    try:
        test_nlp_parsing()
        test_arrow_detection() 
        test_direction_keywords()
        test_improved_fusion()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
