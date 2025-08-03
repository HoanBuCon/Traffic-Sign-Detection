#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Junction Detection Improvements
Kiểm tra cải thiện cho việc phân biệt hướng mũi tên trong biển báo hình chữ T
"""

import cv2
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from real_time_predict_nlp_hybrid import ArrowDetector

def create_test_junction_images():
    """Tạo test images mô phỏng biển báo chữ T"""
    test_images = {}
    
    # Test 1: T-junction với 3 hướng (như mô tả của user)
    # <---|---->
    #       |
    t_junction_1 = np.zeros((100, 120), dtype=np.uint8)
    # Ngang trên (trái-phải)
    cv2.line(t_junction_1, (20, 40), (100, 40), 255, 4)
    # Dọc xuống từ center
    cv2.line(t_junction_1, (60, 40), (60, 80), 255, 4)
    
    # Thêm mũi tên
    # Mũi tên trái
    cv2.arrowedLine(t_junction_1, (35, 40), (25, 40), 255, 2, tipLength=0.3)
    # Mũi tên phải  
    cv2.arrowedLine(t_junction_1, (85, 40), (95, 40), 255, 2, tipLength=0.3)
    # Mũi tên xuống
    cv2.arrowedLine(t_junction_1, (60, 55), (60, 75), 255, 2, tipLength=0.3)
    
    test_images['t_junction_3_way'] = t_junction_1
    
    # Test 2: T-junction với 2 hướng chính
    t_junction_2 = np.zeros((100, 100), dtype=np.uint8)
    # Đường chính ngang
    cv2.line(t_junction_2, (10, 50), (90, 50), 255, 5)
    # Nhánh xuống
    cv2.line(t_junction_2, (50, 50), (50, 85), 255, 4)
    
    # Mũi tên
    cv2.arrowedLine(t_junction_2, (25, 50), (15, 50), 255, 2, tipLength=0.3)  # Trái
    cv2.arrowedLine(t_junction_2, (50, 65), (50, 80), 255, 2, tipLength=0.3)  # Xuống
    
    test_images['t_junction_2_way'] = t_junction_2
    
    # Test 3: Complex junction
    complex_junction = np.zeros((120, 120), dtype=np.uint8)
    center = (60, 60)
    
    # 4 đường từ center
    cv2.line(complex_junction, (60, 20), (60, 100), 255, 4)  # Dọc
    cv2.line(complex_junction, (20, 60), (100, 60), 255, 4)  # Ngang
    
    # Mũi tên 4 hướng
    cv2.arrowedLine(complex_junction, (60, 35), (60, 25), 255, 2, tipLength=0.3)  # Lên
    cv2.arrowedLine(complex_junction, (60, 85), (60, 95), 255, 2, tipLength=0.3)  # Xuống
    cv2.arrowedLine(complex_junction, (35, 60), (25, 60), 255, 2, tipLength=0.3)  # Trái
    cv2.arrowedLine(complex_junction, (85, 60), (95, 60), 255, 2, tipLength=0.3)  # Phải
    
    test_images['complex_4_way'] = complex_junction
    
    return test_images

def test_junction_detection():
    """Test junction detection capabilities"""
    print("🔍 TESTING JUNCTION DETECTION IMPROVEMENTS")
    print("=" * 60)
    
    # Initialize detector
    try:
        detector = ArrowDetector()
        print("✅ ArrowDetector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize ArrowDetector: {e}")
        return
    
    # Create test images
    test_images = create_test_junction_images()
    
    # Test each image
    for test_name, test_image in test_images.items():
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 40)
        
        try:
            # Run detection
            directions, confidence = detector.detect_arrow_direction(test_image)
            
            print(f"Detected directions: {directions}")
            print(f"Overall confidence: {confidence:.3f}")
            
            # Analyze results
            direction_count = len([d for d in directions.keys() if d != "unknown"])
            print(f"Direction count: {direction_count}")
            
            if direction_count >= 2:
                print("✅ Multi-direction detection working")
                
                # Show confidence distribution
                sorted_dirs = sorted(directions.items(), key=lambda x: x[1], reverse=True)
                for direction, conf in sorted_dirs:
                    if direction != "unknown":
                        print(f"  - {direction}: {conf:.3f}")
            else:
                print("⚠️  Single direction detected - may need tuning")
                
        except Exception as e:
            print(f"❌ Error testing {test_name}: {e}")
        
        # Save test image for visual inspection
        output_path = f"test_output_{test_name}.png"
        cv2.imwrite(output_path, test_image)
        print(f"📁 Test image saved: {output_path}")

def test_nlp_parsing():
    """Test enhanced NLP parsing"""
    print("\n\n🔍 TESTING ENHANCED NLP PARSING")
    print("=" * 60)
    
    # Import the main class
    try:
        from real_time_predict_nlp_hybrid import RealTimeTrafficSignDetectorNLPHybrid
        
        # Test NLP parsing method directly
        test_texts = [
            "CHỢ THÁI 3.3 Km | DẦN 1.5 Km",
            "8km đến TÂN CƯƠNG | 91.5km đến BẮC KAN", 
            "D. Tôn Đức Thắng 450m | D. Lê Lợi 680m",
            "76km đến HÀ NỘI | 1km đến HƯỚNG"
        ]
        
        detector = RealTimeTrafficSignDetectorNLPHybrid()
        
        for text in test_texts:
            print(f"\n📝 Testing text: '{text}'")
            locations = detector._parse_locations_from_nlp(text)
            
            print(f"Parsed locations: {len(locations)}")
            for i, loc in enumerate(locations):
                print(f"  Location {i+1}: {loc['name']} - {loc['distance']}km")
                
    except Exception as e:
        print(f"❌ Error testing NLP parsing: {e}")

def test_fusion_logic():
    """Test improved fusion logic"""
    print("\n\n🔍 TESTING IMPROVED FUSION LOGIC")
    print("=" * 60)
    
    try:
        from real_time_predict_nlp_hybrid import RealTimeTrafficSignDetectorNLPHybrid
        
        detector = RealTimeTrafficSignDetectorNLPHybrid()
        
        # Test cases
        test_cases = [
            {
                'cv_directions': {'rẽ trái': 0.65, 'đi thẳng': 0.45},
                'cv_confidence': 0.8,
                'nlp_text': 'CHỢ THÁI 3.3 Km | DẦN 1.5 Km',
                'expected': 'Multi-direction with locations'
            },
            {
                'cv_directions': {'rẽ phải': 0.55, 'rẽ trái': 0.35},
                'cv_confidence': 0.6,
                'nlp_text': '8km đến TÂN CƯƠNG | 91.5km đến BẮC KAN',
                'expected': 'Junction mapping'
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n🧪 Test Case {i+1}: {test_case['expected']}")
            print("-" * 30)
            
            result = detector._fuse_cv_nlp_results_multi(
                test_case['cv_directions'],
                test_case['cv_confidence'], 
                test_case['nlp_text']
            )
            
            print(f"Input CV: {test_case['cv_directions']}")
            print(f"Input NLP: '{test_case['nlp_text']}'")
            print(f"Result: '{result}'")
            
            # Analyze result
            if '|' in result:
                print("✅ Multi-direction result detected")
            else:
                print("ℹ️  Single direction result")
                
    except Exception as e:
        print(f"❌ Error testing fusion logic: {e}")

if __name__ == "__main__":
    print("🚦 JUNCTION DETECTION IMPROVEMENT TESTS")
    print("=" * 60)
    print("Kiểm tra cải thiện cho biển báo hình chữ T và multi-direction")
    print()
    
    # Run all tests
    test_junction_detection()
    test_nlp_parsing()
    test_fusion_logic()
    
    print("\n\n✅ ALL TESTS COMPLETED")
    print("Kiểm tra output để đánh giá cải thiện")
