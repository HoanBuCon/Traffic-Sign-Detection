import os
import sys
import cv2
import time
from ultralytics import YOLO
import numpy as np
import yaml
from config import Config
from utils import ImageEnhancer
import unicodedata
from collections import deque, Counter
from src.sort import Sort
import torch
from PIL import Image, ImageFont, ImageDraw
import torchvision.transforms as T
from transformers import AutoModel, AutoTokenizer
import threading
import queue
import hashlib
import math
from scipy.spatial.distance import euclidean
from scipy import ndimage

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_for_vintern(image, input_size=448, max_num=4):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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

# Load descriptions từ data.yaml
def load_descriptions_from_yaml(yaml_path='data.yaml'):
    """Load descriptions từ file data.yaml"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
            descriptions = data_yaml.get('descriptions', [])
            if not descriptions:
                print(f"[WARNING] No descriptions found in {yaml_path}")
                return []
            else:
                print(f"[INFO] Loaded {len(descriptions)} descriptions from {yaml_path}")
            return descriptions
    except Exception as e:
        print(f"[ERROR] Failed to load descriptions from {yaml_path}: {e}")
        return []

# Load descriptions từ data.yaml
descriptions_vi = load_descriptions_from_yaml()
descriptions_vi_no_diacritics = [remove_vietnamese_diacritics(desc) for desc in descriptions_vi]

def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    inter_area = max(0, inter_rect_x2 - inter_rect_x1 + 1) * max(0, inter_rect_y2 - inter_rect_y1 + 1)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / float(b1_area + b2_area - inter_area)
    return iou

class ArrowDetector:
    """Computer Vision module để detect hướng mũi tên"""
    
    def __init__(self):
        self.arrow_templates = self._create_arrow_templates()
        self.junction_templates = self._create_junction_templates()
        print("[INFO] ArrowDetector initialized with CV templates and junction detection")
    
    def _create_junction_templates(self):
        """Tạo templates cho các kiểu junction (chữ T, chữ Y, ngã ba, ngã tư)"""
        templates = {}
        
        # Template cho junction chữ T (T-junction)
        for size in [30, 40, 50, 60]:
            # T-junction hướng lên (⊥) - đường chính dọc, nhánh ngang trên
            t_up = np.zeros((size, size), dtype=np.uint8)
            center = size // 2
            # Ngang trên (nhánh)
            cv2.line(t_up, (center-12, center-8), (center+12, center-8), 255, 3)
            # Dọc xuống (đường chính)
            cv2.line(t_up, (center, center-8), (center, center+12), 255, 3)
            templates[f"t_junction_up_{size}"] = t_up
            
            # T-junction hướng xuống (⊤) - đường chính dọc, nhánh ngang dưới
            t_down = np.zeros((size, size), dtype=np.uint8)
            # Ngang dưới (nhánh)
            cv2.line(t_down, (center-12, center+8), (center+12, center+8), 255, 3)
            # Dọc lên (đường chính)
            cv2.line(t_down, (center, center-12), (center, center+8), 255, 3)
            templates[f"t_junction_down_{size}"] = t_down
            
            # T-junction hướng trái (⊣) - đường chính ngang, nhánh dọc trái
            t_left = np.zeros((size, size), dtype=np.uint8)
            # Dọc trái (nhánh)
            cv2.line(t_left, (center-8, center-12), (center-8, center+12), 255, 3)
            # Ngang phải (đường chính)
            cv2.line(t_left, (center-8, center), (center+12, center), 255, 3)
            templates[f"t_junction_left_{size}"] = t_left
            
            # T-junction hướng phải (⊢) - đường chính ngang, nhánh dọc phải
            t_right = np.zeros((size, size), dtype=np.uint8)
            # Dọc phải (nhánh)
            cv2.line(t_right, (center+8, center-12), (center+8, center+12), 255, 3)
            # Ngang trái (đường chính)
            cv2.line(t_right, (center-12, center), (center+8, center), 255, 3)
            templates[f"t_junction_right_{size}"] = t_right
        
        return templates
    
    def _create_arrow_templates(self):
        """Tạo templates mũi tên cho các hướng với rotation để handle góc xiên"""
        templates = {}
        
        # Template cho mũi tên trái (pointing left) - nhiều kích thước + rotation
        for size in [20, 30, 40, 50]:
            for angle in [0, -15, 15]:  # Thêm template nghiêng ±15 độ
                left_arrow = np.zeros((size, size), dtype=np.uint8)
                # Vẽ mũi tên trái: < 
                pts = np.array([[size*0.7, size*0.3], [size*0.3, size*0.5], [size*0.7, size*0.7]], dtype=np.int32)
                cv2.fillPoly(left_arrow, [pts], 255)
                
                # Rotate template nếu có angle
                if angle != 0:
                    M = cv2.getRotationMatrix2D((size//2, size//2), angle, 1.0)
                    left_arrow = cv2.warpAffine(left_arrow, M, (size, size))
                
                templates[f'left_{size}_{angle}'] = left_arrow
        
        # Template cho mũi tên phải (pointing right) - nhiều kích thước + rotation  
        for size in [20, 30, 40, 50]:
            for angle in [0, -15, 15]:  # Thêm template nghiêng ±15 độ
                right_arrow = np.zeros((size, size), dtype=np.uint8)
                # Vẽ mũi tên phải: >
                pts = np.array([[size*0.3, size*0.3], [size*0.7, size*0.5], [size*0.3, size*0.7]], dtype=np.int32)
                cv2.fillPoly(right_arrow, [pts], 255)
                
                # Rotate template nếu có angle
                if angle != 0:
                    M = cv2.getRotationMatrix2D((size//2, size//2), angle, 1.0)
                    right_arrow = cv2.warpAffine(right_arrow, M, (size, size))
                
                templates[f'right_{size}_{angle}'] = right_arrow
        
        # Template cho mũi tên thẳng (pointing up) - nhiều kích thước + rotation
        for size in [20, 30, 40, 50]:
            for angle in [0, -10, 10]:  # Ít rotation hơn cho straight
                up_arrow = np.zeros((size, size), dtype=np.uint8)
                # Vẽ mũi tên lên: ^
                pts = np.array([[size*0.3, size*0.7], [size*0.5, size*0.3], [size*0.7, size*0.7]], dtype=np.int32)
                cv2.fillPoly(up_arrow, [pts], 255)
                
                # Rotate template nếu có angle
                if angle != 0:
                    M = cv2.getRotationMatrix2D((size//2, size//2), angle, 1.0)
                    up_arrow = cv2.warpAffine(up_arrow, M, (size, size))
                
                templates[f'up_{size}_{angle}'] = up_arrow
            
        return templates
    
    def _detect_junction_type(self, gray: np.ndarray) -> dict:
        """Detect loại junction (chữ T, Y, ngã ba, ngã tư) trong ảnh"""
        try:
            # Apply edge detection với nhiều threshold
            edges = cv2.Canny(gray, 50, 150)
            
            # Morphological operations để làm sạch
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            junction_scores = {}
            best_match_score = 0.0
            best_junction_type = None
            
            # Template matching với junction templates
            for template_name, template in self.junction_templates.items():
                result = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                junction_type = template_name.split('_')[0] + '_' + template_name.split('_')[1]  # t_junction
                direction = template_name.split('_')[2]  # up, down, left, right
                
                if junction_type not in junction_scores:
                    junction_scores[junction_type] = {}
                junction_scores[junction_type][direction] = max_val
                
                if max_val > best_match_score:
                    best_match_score = max_val
                    best_junction_type = template_name
            
            # Phân tích line intersections để confirm junction
            line_analysis = self._analyze_line_intersections(edges)
            
            # Threshold để xác định có phải junction không
            is_junction = (best_match_score > 0.4 or 
                          line_analysis['intersection_count'] >= 3)
            
            return {
                'is_junction': is_junction,
                'junction_type': best_junction_type,
                'confidence': best_match_score,
                'junction_scores': junction_scores,
                'line_analysis': line_analysis
            }
            
        except Exception as e:
            print(f"[ERROR] Junction detection: {e}")
            return {'is_junction': False, 'junction_type': None, 'confidence': 0.0}
    
    def _analyze_line_intersections(self, edges: np.ndarray) -> dict:
        """Phân tích các giao điểm đường để xác định junction"""
        try:
            # Detect lines using HoughLinesP
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                                   minLineLength=20, maxLineGap=10)
            
            if lines is None:
                return {'intersection_count': 0, 'line_count': 0}
            
            # Tính giao điểm giữa các lines
            intersections = []
            line_count = len(lines)
            
            for i in range(len(lines)):
                for j in range(i+1, len(lines)):
                    intersection = self._find_line_intersection(lines[i][0], lines[j][0])
                    if intersection:
                        intersections.append(intersection)
            
            # Group intersections gần nhau
            grouped_intersections = self._group_nearby_points(intersections, threshold=10)
            
            return {
                'intersection_count': len(grouped_intersections),
                'line_count': line_count,
                'intersections': grouped_intersections
            }
            
        except Exception as e:
            print(f"[ERROR] Line intersection analysis: {e}")
            return {'intersection_count': 0, 'line_count': 0}
    
    def _find_line_intersection(self, line1, line2):
        """Tìm giao điểm của 2 đường thẳng"""
        try:
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-6:  # Parallel lines
                return None
            
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
            
            if 0 <= t <= 1 and 0 <= u <= 1:  # Intersection within line segments
                x = x1 + t*(x2-x1)
                y = y1 + t*(y2-y1)
                return (int(x), int(y))
            
            return None
        except:
            return None
    
    def _group_nearby_points(self, points, threshold=10):
        """Group các điểm gần nhau thành clusters"""
        if not points:
            return []
        
        grouped = []
        used = set()
        
        for i, point in enumerate(points):
            if i in used:
                continue
            
            group = [point]
            used.add(i)
            
            for j, other_point in enumerate(points):
                if j in used:
                    continue
                
                distance = np.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2)
                if distance <= threshold:
                    group.append(other_point)
                    used.add(j)
            
            # Lấy center của group
            center_x = int(np.mean([p[0] for p in group]))
            center_y = int(np.mean([p[1] for p in group]))
            grouped.append((center_x, center_y))
        
        return grouped
    
    def _region_based_analysis(self, gray: np.ndarray, junction_info: dict) -> tuple:
        """Phân tích theo vùng để xác định hướng cho junction"""
        try:
            height, width = gray.shape
            directions = {}
            
            # Chia ảnh thành 4 vùng: trên, dưới, trái, phải
            regions = {
                'top': gray[:height//2, :],
                'bottom': gray[height//2:, :],
                'left': gray[:, :width//2],
                'right': gray[:, width//2:]
            }
            
            # Phân tích density và edge count cho mỗi vùng
            region_scores = {}
            for region_name, region in regions.items():
                edges = cv2.Canny(region, 50, 150)
                edge_density = np.sum(edges) / (region.shape[0] * region.shape[1])
                
                # Detect arrow patterns trong region
                arrow_score = self._detect_arrow_in_region(region)
                
                combined_score = edge_density * 0.6 + arrow_score * 0.4
                region_scores[region_name] = combined_score
            
            # Map regions to directions
            region_to_direction = {
                'top': 'đi thẳng',      # Mũi tên hướng lên
                'bottom': 'đi thẳng',   # Mũi tên hướng xuống  
                'left': 'rẽ trái',      # Mũi tên hướng trái
                'right': 'rẽ phải'      # Mũi tên hướng phải
            }
            
            # Lọc directions có score cao
            threshold = np.mean(list(region_scores.values())) + np.std(list(region_scores.values())) * 0.5
            
            for region_name, score in region_scores.items():
                if score > threshold:
                    direction = region_to_direction[region_name]
                    if direction not in directions:
                        directions[direction] = 0.0
                    directions[direction] += score
            
            # Normalize scores
            total_score = sum(directions.values())
            if total_score > 0:
                for direction in directions:
                    directions[direction] /= total_score
            
            overall_confidence = min(total_score * 0.8, 1.0)  # Scale down confidence
            
            return directions, overall_confidence
            
        except Exception as e:
            print(f"[ERROR] Region-based analysis: {e}")
            return {}, 0.0
    
    def _detect_arrow_in_region(self, region: np.ndarray) -> float:
        """Detect arrow patterns trong một region cụ thể"""
        try:
            # Template matching với smaller templates
            max_score = 0.0
            
            for template_name, template in self.arrow_templates.items():
                if template.shape[0] > region.shape[0] or template.shape[1] > region.shape[1]:
                    continue
                
                result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
                _, current_max, _, _ = cv2.minMaxLoc(result)
                max_score = max(max_score, current_max)
            
            return max_score
            
        except Exception as e:
            return 0.0
    
    def _combine_junction_results(self, tm_directions, contour_directions, edge_direction, 
                                 region_directions, tm_conf, contour_conf, edge_conf, 
                                 region_conf, junction_info) -> tuple:
        """Combine kết quả cho junction với logic đặc biệt"""
        try:
            all_directions = {}
            
            # Weight adjustment cho junction
            junction_confidence = junction_info.get('confidence', 0.0)
            
            # Tăng weight cho region analysis khi detect junction
            region_weight = 0.4 if junction_confidence > 0.5 else 0.2
            tm_weight = 0.3
            contour_weight = 0.2  
            edge_weight = 0.1
            
            # Merge template matching results
            for direction, confidence in tm_directions.items():
                if direction not in all_directions:
                    all_directions[direction] = 0.0
                all_directions[direction] += confidence * tm_weight * tm_conf
            
            # Merge contour analysis results
            for direction, confidence in contour_directions.items():
                if direction not in all_directions:
                    all_directions[direction] = 0.0
                all_directions[direction] += confidence * contour_weight * contour_conf
            
            # Add edge analysis result
            if edge_direction != "unknown" and edge_conf > 0.2:
                if edge_direction not in all_directions:
                    all_directions[edge_direction] = 0.0
                all_directions[edge_direction] += edge_conf * edge_weight
            
            # Add region analysis results (quan trọng cho junction)
            for direction, confidence in region_directions.items():
                if direction not in all_directions:
                    all_directions[direction] = 0.0
                all_directions[direction] += confidence * region_weight * region_conf
            
            # Normalize và filter
            total_score = sum(all_directions.values())
            if total_score > 0:
                normalized_directions = {}
                
                for direction, score in all_directions.items():
                    normalized_score = score / total_score
                    # Lower threshold cho junction (cho phép multi-direction)
                    if normalized_score > 0.1:  
                        normalized_directions[direction] = normalized_score
                
                # Re-normalize
                final_total = sum(normalized_directions.values())
                if final_total > 0:
                    for direction in normalized_directions:
                        normalized_directions[direction] /= final_total
                
                # Boost confidence cho junction detection
                final_confidence = min(total_score * (1.0 + junction_confidence * 0.3), 1.0)
                return normalized_directions, final_confidence
            
            return {"unknown": 1.0}, 0.0
            
        except Exception as e:
            print(f"[ERROR] Junction results combination: {e}")
            return {"unknown": 1.0}, 0.0
    
    def detect_arrow_direction(self, image: np.ndarray) -> tuple:
        """
        Detect nhiều hướng mũi tên trong ảnh sử dụng nhiều phương pháp CV
        Returns: (directions_dict, confidence) - dict chứa tất cả hướng detected
        """
        try:
            # Preprocessing ảnh
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Enhance contrast với CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Kiểm tra junction trước (biển báo hình chữ T)
            junction_result = self._detect_junction_type(enhanced)
            
            # Phương pháp 1: Template Matching với multi-direction support
            tm_directions, tm_confidence = self._template_matching_multi(enhanced)
            
            # Phương pháp 2: Contour Analysis với multi-direction support
            contour_directions, contour_confidence = self._contour_analysis_multi(enhanced)
            
            # Phương pháp 3: Edge Direction Analysis
            edge_direction, edge_confidence = self._edge_direction_analysis(enhanced)
            
            # Phương pháp 4: Region-based Analysis cho junction
            if junction_result['is_junction']:
                region_directions, region_confidence = self._region_based_analysis(enhanced, junction_result)
                # Combine with junction-aware logic
                final_directions, final_confidence = self._combine_junction_results(
                    tm_directions, contour_directions, edge_direction, region_directions,
                    tm_confidence, contour_confidence, edge_confidence, region_confidence,
                    junction_result
                )
            else:
                # Phương pháp 3: Edge Direction Analysis (single direction)
                edge_direction, edge_confidence = self._edge_direction_analysis(enhanced)
                
                # Combine kết quả từ các phương pháp
                final_directions, final_confidence = self._combine_multi_results(
                    tm_directions, contour_directions, edge_direction,
                    tm_confidence, contour_confidence, edge_confidence
                )
            
            return final_directions, final_confidence
            
        except Exception as e:
            print(f"[ERROR] Arrow detection: {e}")
            return {"unknown": 1.0}, 0.0
            
            # Phương pháp 3: Edge Direction Analysis
            edge_direction, edge_confidence = self._edge_direction_analysis(enhanced)
            
            # Combine multi-direction results
            final_directions, final_confidence = self._combine_multi_results(tm_directions, contour_directions, edge_direction, tm_confidence, contour_confidence, edge_confidence)
            
            return final_directions, final_confidence
            
        except Exception as e:
            print(f"[ERROR] ArrowDetector error: {e}")
            return {"unknown": 1.0}, 0.0
    
    def _template_matching(self, gray: np.ndarray) -> tuple:
        """Template matching với multi-scale và mirror detection cải thiện"""
        best_match_val = 0.0
        best_direction = "unknown"
        
        # Apply Gaussian blur và Canny để tăng cường edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Tăng blur để giảm noise
        edges = cv2.Canny(blurred, 30, 100)  # Giảm threshold để detect edges nhẹ hơn
        
        # Morphological operations để clean up edges
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        # Mirror detection check - so sánh intensity left vs right với window nhỏ hơn
        height, width = edges.shape
        center_y = height // 2
        roi_height = height // 3  # Chỉ xét vùng giữa
        
        left_roi = edges[center_y-roi_height//2:center_y+roi_height//2, :width//2]
        right_roi = edges[center_y-roi_height//2:center_y+roi_height//2, width//2:]
        
        left_intensity = np.sum(left_roi)
        right_intensity = np.sum(right_roi)
        intensity_ratio = left_intensity / (right_intensity + 1e-6)
        
        print(f"[CV DEBUG] ROI Intensity L/R ratio: {intensity_ratio:.2f}")
        
        # Template matching với normalization tốt hơn
        direction_scores = {"rẽ trái": 0.0, "rẽ phải": 0.0, "đi thẳng": 0.0}
        
        for template_name, template in self.arrow_templates.items():
            # Match với ảnh edge
            result = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # Weighted score based on template size và angle
            direction = template_name.split('_')[0]
            size = int(template_name.split('_')[1])
            angle = int(template_name.split('_')[2])
            
            # Weight: larger templates = higher weight, 0-angle = higher weight
            size_weight = (size - 15) / 35  # 0.1 to 1.0
            angle_weight = 1.0 - abs(angle) / 15  # 1.0 for 0°, lower for tilted
            weighted_score = max_val * size_weight * angle_weight
            
            if direction == "left":
                direction_scores["rẽ trái"] += weighted_score
            elif direction == "right":
                direction_scores["rẽ phải"] += weighted_score
            elif direction == "up":
                direction_scores["đi thẳng"] += weighted_score
        
        # Tìm best direction từ accumulated scores
        best_direction = max(direction_scores, key=direction_scores.get)
        best_match_val = direction_scores[best_direction] / len([t for t in self.arrow_templates.keys() if t.startswith(best_direction.replace("rẽ ", "").replace("đi thẳng", "up"))])
        
        # Enhanced mirror correction với threshold chặt hơn
        if best_direction == "rẽ trái" and intensity_ratio < 0.6:
            print(f"[CV DEBUG] Strong mirror correction: {best_direction} -> rẽ phải (L/R={intensity_ratio:.2f})")
            best_direction = "rẽ phải"
            best_match_val *= 0.7
        elif best_direction == "rẽ phải" and intensity_ratio > 1.6:
            print(f"[CV DEBUG] Strong mirror correction: {best_direction} -> rẽ trái (L/R={intensity_ratio:.2f})")
            best_direction = "rẽ trái"
            best_match_val *= 0.7
        
        return best_direction, best_match_val
    
    def _template_matching_multi(self, gray: np.ndarray) -> tuple:
        """Template matching với multi-direction detection được cải thiện"""
        # Apply Gaussian blur và Canny để tăng cường edges
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Giảm blur để giữ chi tiết
        edges = cv2.Canny(blurred, 50, 150)  # Tăng threshold để detect edges rõ hơn
        
        # Morphological operations được điều chỉnh
        kernel = np.ones((2,2), np.uint8)  # Kernel nhỏ hơn
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Additional preprocessing: điều chỉnh contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        enhanced_gray = clahe.apply(gray)
        
        # Template matching với normalization tốt hơn - track ALL directions
        direction_scores = {"rẽ trái": [], "rẽ phải": [], "đi thẳng": []}
        
        for template_name, template in self.arrow_templates.items():
            # Match với cả edge image và enhanced gray
            result_edge = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED)
            result_gray = cv2.matchTemplate(enhanced_gray, template, cv2.TM_CCOEFF_NORMED)
            
            # Combine cả 2 kết quả
            combined_result = 0.6 * result_edge + 0.4 * result_gray
            _, max_val, _, max_loc = cv2.minMaxLoc(combined_result)
            
            direction = template_name.split('_')[0]
            size = int(template_name.split('_')[1])
            angle = int(template_name.split('_')[2])
            
            # Improved weight calculation
            size_weight = 0.5 + (size - 20) / 60  # 0.5 to 1.0
            angle_weight = 1.0 - abs(angle) / 20   # Penalty cho angle cao
            
            # Quality boost cho high-confidence matches
            quality_boost = 1.0
            if max_val > 0.6:
                quality_boost = 1.2
            elif max_val > 0.4:
                quality_boost = 1.1
            
            weighted_score = max_val * size_weight * angle_weight * quality_boost
            
            # Map to Vietnamese và store individual scores
            vn_direction = "unknown"
            if direction == "left":
                vn_direction = "rẽ trái"
            elif direction == "right":
                vn_direction = "rẽ phải"
            elif direction == "up":
                vn_direction = "đi thẳng"
            
            if vn_direction != "unknown":
                direction_scores[vn_direction].append(weighted_score)
        
        # Aggregate scores với statistical analysis
        final_direction_scores = {}
        for direction, scores in direction_scores.items():
            if scores:
                # Use both max và average để balance outliers
                max_score = max(scores)
                avg_score = sum(scores) / len(scores)
                # Weighted combination: 70% max, 30% average
                final_score = 0.7 * max_score + 0.3 * avg_score
                final_direction_scores[direction] = final_score
        
        # Filter directions có confidence đủ cao với adaptive threshold
        detected_directions = {}
        total_score = sum(final_direction_scores.values())
        
        if total_score > 0:
            max_score = max(final_direction_scores.values())
            
            for direction, score in final_direction_scores.items():
                normalized_score = score / total_score
                
                # Adaptive threshold dựa trên score distribution
                if max_score > 0.6:  # Có 1 direction rất dominant
                    threshold = 0.2
                else:  # Scores tương đương nhau
                    threshold = 0.15
                
                if normalized_score > threshold and score > 0.1:  # Absolute minimum
                    detected_directions[direction] = normalized_score
        
        # Fallback nếu không detect được gì
        if not detected_directions and final_direction_scores:
            best_dir = max(final_direction_scores, key=final_direction_scores.get)
            best_score = final_direction_scores[best_dir] / max(total_score, 1e-6)
            if best_score > 0.05:  # Very low threshold cho fallback
                detected_directions[best_dir] = best_score
        
        overall_confidence = min(sum(detected_directions.values()) * 1.1, 1.0)  # Slight boost
        return detected_directions, overall_confidence
    
    def _contour_analysis_multi(self, gray: np.ndarray) -> tuple:
        """Phân tích contour để tìm nhiều hình dạng mũi tên"""
        try:
            # Threshold để tạo binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            direction_detections = {}  # direction -> list of confidences
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # Skip quá nhỏ
                    continue
                
                # Approximate contour shape
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 3:  # Có ít nhất 3 điểm (triangle-like)
                    # Tìm tip (đỉnh nhọn) của mũi tên
                    direction, confidence = self._find_arrow_tip(approx)
                    if direction != "unknown" and confidence > 0.3:
                        if direction not in direction_detections:
                            direction_detections[direction] = []
                        direction_detections[direction].append(confidence)
            
            # Aggregate multi-detections cho mỗi direction
            final_directions = {}
            for direction, confidences in direction_detections.items():
                # Lấy max confidence nhưng có factor cho số lượng detections
                max_conf = max(confidences)
                count_factor = min(len(confidences) / 3.0, 1.0)  # Bonus cho multiple detections
                final_conf = max_conf * (0.7 + 0.3 * count_factor)
                final_directions[direction] = final_conf
            
            overall_confidence = min(sum(final_directions.values()), 1.0) if final_directions else 0.0
            return final_directions, overall_confidence
            
        except Exception as e:
            print(f"[ERROR] Multi contour analysis: {e}")
            return {}, 0.0
    
    def _combine_multi_results(self, tm_directions, contour_directions, edge_direction, tm_conf, contour_conf, edge_conf) -> tuple:
        """Combine kết quả từ nhiều phương pháp với multi-direction support được cải thiện"""
        all_directions = {}
        
        # Cải thiện weight distribution dựa trên reliability
        tm_weight = 0.45 if tm_conf > 0.4 else 0.35
        contour_weight = 0.35 if contour_conf > 0.5 else 0.25  
        edge_weight = 0.20 if edge_conf > 0.3 else 0.15
        
        # Merge template matching results với weight cải thiện
        for direction, confidence in tm_directions.items():
            if direction not in all_directions:
                all_directions[direction] = 0.0
            # Áp dụng penalty cho confidence quá thấp
            adjusted_conf = confidence if confidence >= 0.25 else confidence * 0.5
            all_directions[direction] += adjusted_conf * tm_weight * tm_conf
        
        # Merge contour analysis results với validation tốt hơn
        for direction, confidence in contour_directions.items():
            if direction not in all_directions:
                all_directions[direction] = 0.0
            # Bonus cho contour với confidence cao
            adjusted_conf = confidence * 1.1 if confidence >= 0.7 else confidence
            all_directions[direction] += adjusted_conf * contour_weight * contour_conf
        
        # Add edge analysis result với conditional weighting
        if edge_direction != "unknown" and edge_conf > 0.15:
            if edge_direction not in all_directions:
                all_directions[edge_direction] = 0.0
            # Edge analysis weight dựa trên edge confidence
            dynamic_edge_weight = min(edge_weight * (edge_conf / 0.5), 0.3)
            all_directions[edge_direction] += edge_conf * dynamic_edge_weight
        
        # Normalize và filter với threshold adaptive
        total_score = sum(all_directions.values())
        if total_score > 0:
            normalized_directions = {}
            max_score = max(all_directions.values())
            
            for direction, score in all_directions.items():
                normalized_score = score / total_score
                # Adaptive threshold: cao hơn nếu có 1 direction dominant
                threshold = 0.15 if max_score / total_score > 0.6 else 0.12
                
                if normalized_score > threshold:
                    # Boost confidence cho direction có score cao rõ rệt
                    if normalized_score > 0.4:
                        normalized_score = min(normalized_score * 1.1, 1.0)
                    normalized_directions[direction] = normalized_score
            
            # Re-normalize after boosting
            final_total = sum(normalized_directions.values())
            if final_total > 0:
                for direction in normalized_directions:
                    normalized_directions[direction] /= final_total
            
            final_confidence = min(total_score * 1.2, 1.0)  # Slight boost cho overall confidence
            return normalized_directions, final_confidence
        
        return {"unknown": 1.0}, 0.0
    
    def _contour_analysis(self, gray: np.ndarray) -> tuple:
        """Phân tích contour để tìm hình dạng mũi tên"""
        try:
            # Threshold để tạo binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_confidence = 0.0
            best_direction = "unknown"
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # Skip quá nhỏ
                    continue
                
                # Approximate contour shape
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 3:  # Có ít nhất 3 điểm (triangle-like)
                    # Tìm tip (đỉnh nhọn) của mũi tên
                    direction, confidence = self._find_arrow_tip(approx)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_direction = direction
            
            return best_direction, best_confidence
            
        except Exception as e:
            print(f"[ERROR] Contour analysis: {e}")
            return "unknown", 0.0
    
    def _find_arrow_tip(self, approx_points) -> tuple:
        """Tìm hướng tip của mũi tên từ các điểm contour với angle normalization cải thiện"""
        try:
            points = approx_points.reshape(-1, 2)
            if len(points) < 3:
                return "unknown", 0.0
            
            # Tính center của shape
            center = np.mean(points, axis=0)
            
            # Tìm điểm xa nhất từ center (có thể là tip)
            distances = [euclidean(point, center) for point in points]
            tip_idx = np.argmax(distances)
            tip_point = points[tip_idx]
            
            # Xác định hướng từ center đến tip
            direction_vector = tip_point - center
            angle_rad = math.atan2(direction_vector[1], direction_vector[0])
            
            # Normalize angle to [0, 360) degrees để tránh nhầm lẫn
            angle_deg = (math.degrees(angle_rad) + 360) % 360
            
            # Classify angle thành direction với boundary rõ ràng hơn
            if 315 <= angle_deg or angle_deg < 45:  # Right (0° ±45°)
                return "rẽ phải", 0.75
            elif 135 <= angle_deg < 225:  # Left (180° ±45°)  
                return "rẽ trái", 0.75
            elif 45 <= angle_deg < 135:  # Down/Southeast (90° ±45°) -> đi thẳng
                return "đi thẳng", 0.65
            elif 225 <= angle_deg < 315:  # Up/Northwest (270° ±45°) -> đi thẳng
                return "đi thẳng", 0.70
            
            return "unknown", 0.0
            
        except Exception as e:
            print(f"[ERROR] Find arrow tip: {e}")
            return "unknown", 0.0
    
    def _edge_direction_analysis(self, gray: np.ndarray) -> tuple:
        """Phân tích edge direction sử dụng Sobel gradients với algorithm cải thiện"""
        try:
            # Apply enhancement trước khi tính gradients
            enhanced = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Tính gradients với kernel size lớn hơn để stable hơn
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
            
            # Tính magnitude và direction
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            direction = np.arctan2(sobely, sobelx)
            
            # Chỉ xét pixels có magnitude cao (edges mạnh) với adaptive threshold
            magnitude_threshold = np.percentile(magnitude, 85)  # Tăng từ 80% lên 85%
            strong_edges_mask = magnitude > magnitude_threshold
            
            if np.sum(strong_edges_mask) == 0:
                return "unknown", 0.0
            
            strong_directions_rad = direction[strong_edges_mask]
            strong_magnitudes = magnitude[strong_edges_mask]
            
            # Convert to degrees và normalize to [0, 360)
            strong_directions_deg = (np.degrees(strong_directions_rad) + 360) % 360
            
            # Weighted histogram based on magnitude
            hist, bins = np.histogram(strong_directions_deg, bins=72, range=(0, 360), 
                                      weights=strong_magnitudes)  # Weight by magnitude
            
            # Smooth histogram để giảm noise
            from scipy import ndimage
            hist_smooth = ndimage.gaussian_filter1d(hist, sigma=1.0)
            
            # Tìm top 3 dominant directions để cross-validation
            top_bins = np.argsort(hist_smooth)[-3:]
            dominant_angles = [(bins[i] + bins[i + 1]) / 2 for i in top_bins]
            dominant_counts = [hist_smooth[i] for i in top_bins]
            
            # Lấy direction có count cao nhất
            primary_angle = dominant_angles[-1]
            primary_count = dominant_counts[-1]
            
            # Validate với secondary peaks
            secondary_count = dominant_counts[-2] if len(dominant_counts) > 1 else 0
            peak_ratio = primary_count / (secondary_count + 1e-6)
            
            # Convert angle thành traffic direction với improved boundaries
            total_weight = np.sum(hist_smooth)
            base_confidence = primary_count / total_weight
            
            # Direction classification với tighter boundaries
            detected_direction = "unknown"
            confidence_modifier = 1.0
            
            if 345 <= primary_angle or primary_angle < 15:  # Right (0° ±15°) 
                detected_direction = "rẽ phải"
                confidence_modifier = 1.1  # Boost for clear horizontal
            elif 165 <= primary_angle < 195:  # Left (180° ±15°)
                detected_direction = "rẽ trái" 
                confidence_modifier = 1.1
            elif 75 <= primary_angle < 105:  # Down (90° ±15°)
                detected_direction = "đi thẳng"
                confidence_modifier = 0.9  # Slight penalty for down-pointing
            elif 255 <= primary_angle < 285:  # Up (270° ±15°)
                detected_direction = "đi thẳng"
                confidence_modifier = 1.0
            else:
                # Diagonal directions - lower confidence
                if 15 <= primary_angle < 75 or 285 <= primary_angle < 345:
                    detected_direction = "rẽ phải"
                    confidence_modifier = 0.7  # Penalty for diagonal
                elif 105 <= primary_angle < 165 or 195 <= primary_angle < 255:
                    detected_direction = "rẽ trái"
                    confidence_modifier = 0.7
            
            # Final confidence calculation
            final_confidence = base_confidence * confidence_modifier
            
            # Boost confidence nếu có clear dominant peak
            if peak_ratio > 2.0:  # Very dominant peak
                final_confidence *= 1.2
            elif peak_ratio > 1.5:  # Moderately dominant
                final_confidence *= 1.1
            
            final_confidence = min(final_confidence, 1.0)  # Cap at 1.0
            
            return detected_direction, final_confidence
            
        except Exception as e:
            print(f"[ERROR] Edge direction analysis: {e}")
            return "unknown", 0.0
    
    def _combine_results(self, results) -> tuple:
        """Combine kết quả từ nhiều phương pháp với weighted voting cải thiện"""
        direction_scores = {}
        total_confidence = 0.0
        
        for direction, confidence, weight in results:
            if direction != "unknown" and confidence > 0.1:  # Chỉ xét confidence > threshold
                if direction not in direction_scores:
                    direction_scores[direction] = 0.0
                weighted_score = confidence * weight
                direction_scores[direction] += weighted_score
                total_confidence += weighted_score
        
        if not direction_scores:
            return "unknown", 0.0
        
        # Chọn direction có score cao nhất
        best_direction = max(direction_scores, key=direction_scores.get)
        best_score = direction_scores[best_direction]
        
        # Normalize score
        if total_confidence > 0:
            normalized_score = best_score / total_confidence
        else:
            normalized_score = 0.0
        
        # Kiểm tra consensus - nếu 2 methods đồng ý thì tăng confidence
        consensus_count = sum(1 for direction, conf, weight in results if direction == best_direction and conf > 0.2)
        if consensus_count >= 2:
            normalized_score *= 1.2  # Bonus cho consensus
            print(f"[CV DEBUG] Consensus bonus: {consensus_count} methods agree on {best_direction}")
        
        return best_direction, min(normalized_score, 1.0)  # Cap tại 1.0

class RealTimeTrafficSignDetectorNLPHybrid:
    def __init__(self, model_path=None):
        print("[DEBUG] Init class RealTimeTrafficSignDetectorNLPHybrid")
        if model_path is None:
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
        with open('data.yaml', 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
            self.class_names = data_yaml.get('names', {})
        self.tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
        print("[DEBUG] Before init OCR queue/cache/label_buffers")
        self.ocr_queue = queue.Queue()
        self.ocr_cache = {}
        self.ocr_queue_sent = set()
        self.ocr_bbox_cache = {}  # Cache để lưu bbox cuối cùng của mỗi object
        self.ocr_image_hash_cache = {}  # Cache để lưu hash của ảnh đã OCR
        self.ocr_timestamp_cache = {}  # Cache timestamp để hết hạn
        self.label_buffers = {}
        self.buffer_size = 10
        self.ocr_cache_timeout = 30.0  # Cache timeout 30 giây
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[DEBUG] Before init ArrowDetector")
        self.arrow_detector = ArrowDetector()
        print("[DEBUG] Before load Vintern NLP model")
        self._load_nlp_model()
        print("[DEBUG] Before start OCR thread")
        threading.Thread(target=self.ocr_worker, daemon=True).start()
        print("[DEBUG] After start OCR thread")

    def _load_nlp_model(self):
        print("[INFO] Đang tải mô hình Vintern NLP...")
        try:
            self.vintern_model = AutoModel.from_pretrained(
                "5CD-AI/Vintern-1B-v3_5",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=False,
            ).eval().to(self.device)
            self.vintern_tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-1B-v3_5", trust_remote_code=True, use_fast=False)
            print(f"[INFO] Đã tải xong Vintern NLP và chạy trên {self.device}.")
        except Exception as e:
            print(f"[ERROR] Không thể tải Vintern NLP: {e}")
            self.vintern_model = None
            self.vintern_tokenizer = None

    def _compute_image_hash(self, image: np.ndarray) -> str:
        """Tính hash của ảnh để xác định nội dung duy nhất"""
        try:
            # Resize ảnh về kích thước cố định để tránh ảnh hưởng của scale
            resized = cv2.resize(image, (64, 64))
            # Chuyển sang grayscale để giảm ảnh hưởng của màu sắc
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # Tính hash MD5
            img_hash = hashlib.md5(gray.tobytes()).hexdigest()
            return img_hash
        except Exception as e:
            print(f"[ERROR] Không thể tính hash ảnh: {e}")
            return ""

    def _is_cache_valid(self, object_id: int, current_bbox: list, image_hash: str) -> bool:
        """Kiểm tra cache có còn hợp lệ không với tăng cường sensitivity"""
        current_time = time.time()
        
        # Kiểm tra timeout - giảm timeout để refresh thường xuyên hơn
        if object_id in self.ocr_timestamp_cache:
            if current_time - self.ocr_timestamp_cache[object_id] > 15.0:  # Giảm từ 30s xuống 15s
                return False
        
        # Kiểm tra bbox thay đổi - tăng sensitivity
        if object_id in self.ocr_bbox_cache:
            old_bbox = self.ocr_bbox_cache[object_id]
            iou_bbox = bbox_iou(current_bbox, old_bbox)
            if iou_bbox < 0.85:  # Tăng từ 0.7 lên 0.85 để sensitive hơn với bbox changes
                return False
        
        # Kiểm tra nội dung ảnh thay đổi (quan trọng nhất)
        if object_id in self.ocr_image_hash_cache:
            old_hash = self.ocr_image_hash_cache[object_id]
            if old_hash != image_hash:
                return False
        
        return True

    def _clear_object_cache(self, object_id: int):
        """Xóa toàn bộ cache của một object"""
        caches_to_clear = [
            self.ocr_cache,
            self.ocr_bbox_cache, 
            self.ocr_image_hash_cache,
            self.ocr_timestamp_cache
        ]
        
        for cache_dict in caches_to_clear:
            if object_id in cache_dict:
                del cache_dict[object_id]
        
        self.ocr_queue_sent.discard((object_id, ))

    def _update_object_cache(self, object_id: int, bbox: list, image_hash: str, ocr_result: str = None):
        """Cập nhật cache cho object"""
        current_time = time.time()
        self.ocr_bbox_cache[object_id] = bbox
        self.ocr_image_hash_cache[object_id] = image_hash
        self.ocr_timestamp_cache[object_id] = current_time
        
        if ocr_result is not None:
            self.ocr_cache[object_id] = ocr_result

    def _cleanup_expired_cache(self):
        """Dọn dẹp cache hết hạn"""
        current_time = time.time()
        expired_objects = []
        
        for object_id, timestamp in self.ocr_timestamp_cache.items():
            if current_time - timestamp > self.ocr_cache_timeout:
                expired_objects.append(object_id)
        
        for object_id in expired_objects:
            self._clear_object_cache(object_id)

    def ocr_worker(self):
        while True:
            object_id, sign_crop = self.ocr_queue.get()
            try:
                # Tính hash của ảnh để đảm bảo tính nhất quán
                image_hash = self._compute_image_hash(sign_crop)
                
                # Kiểm tra xem object_id có còn hợp lệ không
                if object_id in self.ocr_image_hash_cache:
                    cached_hash = self.ocr_image_hash_cache[object_id]
                    if cached_hash != image_hash:
                        continue  # Skip OCR nếu hash không khớp
                
                ocr_text = self._get_text_from_sign(sign_crop)
                if ocr_text and len(ocr_text.strip()) > 0:
                    # Cập nhật cache với kết quả OCR
                    self.ocr_cache[object_id] = ocr_text
            except Exception as e:
                print(f"[ERROR] OCR thread: {e}")
            finally:
                self.ocr_queue.task_done()

    def _get_text_from_sign(self, sign_image: np.ndarray) -> str:
        """Hybrid method: Computer Vision + NLP"""
        if self.vintern_model is None or self.vintern_tokenizer is None:
            return ""
        try:
            # BƯỚC 1: Computer Vision Arrow Detection (Multi-Direction)
            cv_directions, cv_confidence = self.arrow_detector.detect_arrow_direction(sign_image)
            
            # BƯỚC 2: NLP Text Reading
            pil_image = Image.fromarray(cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB))
            pixel_values = load_image_for_vintern(pil_image, input_size=448, max_num=4).to(torch.float16).to(self.device)
            generation_config = dict(max_new_tokens=200, do_sample=False, num_beams=3, repetition_penalty=1.3, pad_token_id=self.vintern_tokenizer.eos_token_id)
            
            # Prompt cực kỳ đơn giản
            question = """<image>
Đọc text"""
            
            response, history = self.vintern_model.chat(self.vintern_tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            
            # Xử lý NLP response
            response = response.strip()
            response = response.replace('**', '').replace('`', '').replace('*', '')
            
            # Tách các dòng và lọc thông tin hữu ích
            all_lines = []
            for line in response.split('\n'):
                line = line.strip()
                if line and len(line) >= 3 and len(line) <= 120:
                    # Loại bỏ các dòng mô tả không cần thiết
                    if not any(skip in line.lower() for skip in [
                        'biển báo', 'hình ảnh', 'theo thứ tự', 'liệt kê', 'đọc được',
                        'nhìn thấy', 'có thể thấy', 'xuất hiện', 'tôi thấy', 'dựa vào'
                    ]):
                        # Làm sạch dòng
                        clean_line = line.lstrip('- ').lstrip('• ').lstrip('1234567890. ').strip()
                        if clean_line and len(clean_line) >= 2:
                            all_lines.append(clean_line)
            
            nlp_text = ""
            if all_lines:
                nlp_text = ' | '.join(all_lines[:4])  # Lấy tối đa 4 dòng đầu tiên
            else:
                # Fallback: lấy response gốc và làm sạch
                fallback = response.replace('\n', ' ').strip()
                if len(fallback) > 200:
                    fallback = fallback[:200] + "..."
                nlp_text = fallback
            
            # BƯỚC 3: Fusion CV + NLP Results (Multi-Direction)
            final_result = self._fuse_cv_nlp_results_multi(cv_directions, cv_confidence, nlp_text)
            print(f"[FINAL RESULT] {final_result}")
            
            return final_result
                
        except Exception as e:
            print(f"[ERROR] Lỗi khi xử lý Hybrid CV+NLP: {e}")
            return ""

    def _fuse_cv_nlp_results_multi(self, cv_directions: dict, cv_confidence: float, nlp_text: str) -> str:
        """
        Smart fusion của CV (multi-directions) và NLP results với cải thiện cho junction detection
        """
        # Parse multiple locations from NLP text
        locations = self._parse_locations_from_nlp(nlp_text)
        
        # Validate locations - đảm bảo có ít nhất 1 location hợp lệ
        valid_locations = [loc for loc in locations if loc['name'] and len(loc['name']) > 2 and loc['name'] != 'm']
        
        # Cải thiện logic fusion với junction awareness
        if len(valid_locations) >= 1 and cv_confidence >= 0.4:  # Giảm threshold để accept junction results
            # Có location hợp lệ và CV confidence đủ
            if len(cv_directions) > 1 and "unknown" not in cv_directions:
                # Multi-direction case (có thể là junction) với intelligent mapping
                mapped_result = self._map_directions_to_locations_intelligent(cv_directions, valid_locations)
                if mapped_result and len(mapped_result) > 10:  # Validate kết quả không quá ngắn
                    return mapped_result
            
            # Single direction case hoặc fallback cho junction
            best_direction_item = max(cv_directions.items(), key=lambda x: x[1])
            best_direction, best_confidence = best_direction_item
            
            if best_confidence >= 0.25 and best_direction != "unknown":  # Lower threshold cho junction
                if len(valid_locations) == 1:
                    loc = valid_locations[0]
                    result = f"{best_direction} {loc['distance']}km đến {loc['name']}"
                    return result
                elif len(valid_locations) > 1:
                    # Prioritize mapping với consistent logic
                    if len(cv_directions) >= 2:
                        # Multi-direction detected, map theo priority
                        mapped_result = self._map_directions_with_priority(cv_directions, valid_locations)
                        if mapped_result:
                            return mapped_result
                    
                    # Fallback: ghép với location đầu tiên
                    loc = valid_locations[0]
                    result = f"{best_direction} {loc['distance']}km đến {loc['name']}"
                    return result
        
        # Enhanced fallback strategies
        
        # Strategy 1: Kiểm tra NLP text có direction keywords không
        if self._has_direction_keywords(nlp_text):
            # Enhance NLP với CV information nếu có
            if cv_directions and cv_confidence >= 0.3:
                best_cv_direction = max(cv_directions.items(), key=lambda x: x[1])[0]
                enhanced_nlp = self._enhance_nlp_with_cv_smart(nlp_text, best_cv_direction, cv_confidence)
                return enhanced_nlp
            return nlp_text
        
        # Strategy 2: CV-only với confidence validation
        if cv_directions and cv_confidence >= 0.25:  # Lower threshold
            # Nếu có multi-directions với confidence tương đối
            if len(cv_directions) >= 2:
                sorted_dirs = sorted(cv_directions.items(), key=lambda x: x[1], reverse=True)
                # Chỉ lấy top 2 directions với confidence gap không quá lớn
                if len(sorted_dirs) >= 2 and sorted_dirs[1][1] / sorted_dirs[0][1] > 0.6:
                    dir_list = [f"{dir_name}" for dir_name, conf in sorted_dirs[:2] if conf >= 0.2]
                    if len(dir_list) >= 2:
                        return " | ".join(dir_list)
            
            # Single best direction
            best_direction = max(cv_directions.items(), key=lambda x: x[1])[0]
            if best_direction != "unknown":
                return best_direction
        
        # Strategy 3: Sử dụng NLP nếu có nội dung hữu ích
        if nlp_text and len(nlp_text.strip()) > 5:
            # Enhanced filtering cho unwanted responses
            unwanted_phrases = [
                "too blurry", "cannot read", "không đọc được", 
                "image is", "text in the image", "unclear", "not clear enough",
                "no text", "no information", "no relevant information"
            ]
            if not any(phrase in nlp_text.lower() for phrase in unwanted_phrases):
                return nlp_text
        
        # Last resort: Intelligent unknown handling
        return "Biển báo chỉ dẫn - Chưa xác định được hướng rõ ràng"
    
    def _parse_locations_from_nlp(self, nlp_text: str) -> list:
        """Parse các địa điểm và khoảng cách từ NLP text với multiple patterns được cải thiện"""
        if not nlp_text:
            return []
        
        locations = []
        # Split by | để tách các dòng
        parts = nlp_text.split('|')
        
        for part in parts:
            part = part.strip()
            
            # Pattern 1: "TÊN ĐỊA ĐIỂM + SỐ + Km" (như "CHỢ THÁI 3.3 Km")
            import re
            match1 = re.search(r'([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ\s]+?)\s+(\d+(?:[,\.]\d+)?)\s*[Kk]m?', part, re.IGNORECASE)
            
            # Pattern 2: "TÊN + SỐ,SỐ Km" (như "CHỢ THÁI 3,3 Km") 
            match2 = re.search(r'([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ\s]+?)\s+(\d+[,\.]\d+)\s*[Kk]m', part, re.IGNORECASE)
            
            # Pattern 3: "SỐ,SỐ Km + TÊN" (như "3,3 Km CHỢ THÁI") - cải thiện
            match3 = re.search(r'(\d+[,\.]\d+)\s*[Kk]m?\s+([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ\s]+)', part, re.IGNORECASE)
            
            location_found = False
            
            if match2:  # Ưu tiên pattern 2 (CHỢ THÁI 3,3 Km)
                location_name = match2.group(1).strip()
                distance = match2.group(2).replace(',', '.')
                locations.append({
                    'name': location_name,
                    'distance': distance,
                    'original': part
                })
                location_found = True
            elif match1:  # Pattern 1 (CHỢ THÁI 3 Km)
                location_name = match1.group(1).strip()
                distance = match1.group(2).replace(',', '.')
                locations.append({
                    'name': location_name,
                    'distance': distance,
                    'original': part
                })
                location_found = True
            elif match3:  # Pattern 3 (3,3 Km CHỢ THÁI)
                distance = match3.group(1).replace(',', '.')
                location_name = match3.group(2).strip()
                locations.append({
                    'name': location_name,
                    'distance': distance,
                    'original': part
                })
                location_found = True
            
            # Nếu không match được pattern nào, thử extract đơn giản
            if not location_found and len(part) > 3:
                # Fallback: tìm số và text riêng biệt
                distance_match = re.search(r'(\d+[,\.]\d+)', part)
                if distance_match:
                    distance = distance_match.group(1).replace(',', '.')
                    # Loại bỏ số và "Km" để lấy tên địa điểm
                    location_name = re.sub(r'\d+[,\.]\d+\s*[Kk]m?', '', part).strip()
                    if location_name and len(location_name) > 2:
                        locations.append({
                            'name': location_name,
                            'distance': distance,
                            'original': part
                        })
        
        return locations
    
    def _map_directions_to_locations_improved(self, cv_directions: dict, locations: list) -> str:
        """Map CV directions với locations based on improved logic với validation"""
        if len(locations) < 1 or len(cv_directions) < 1:
            return ""
        
        # Sort CV directions by confidence
        sorted_directions = sorted(cv_directions.items(), key=lambda x: x[1], reverse=True)
        
        # Intelligent mapping strategy
        result_parts = []
        used_directions = []
        
        # Nếu có nhiều locations và directions
        if len(locations) >= 2 and len(sorted_directions) >= 2:
            # Map top 2 directions với 2 locations đầu tiên
            for i, location in enumerate(locations[:2]):
                if i < len(sorted_directions):
                    direction_name, confidence = sorted_directions[i]
                    if confidence >= 0.25:  # Threshold tối thiểu
                        result_parts.append(f"{direction_name} {location['distance']}km đến {location['name']}")
                        used_directions.append(direction_name)
        else:
            # Single location hoặc ít directions
            best_direction = sorted_directions[0][0] if sorted_directions[0][1] >= 0.3 else None
            if best_direction and locations:
                loc = locations[0]
                result_parts.append(f"{best_direction} {loc['distance']}km đến {loc['name']}")
        
        return " | ".join(result_parts) if result_parts else ""
    
    def _map_directions_to_locations_intelligent(self, cv_directions: dict, locations: list) -> str:
        """Intelligent mapping cho junction với logic cải thiện"""
        if len(locations) < 1 or len(cv_directions) < 1:
            return ""
        
        # Sort CV directions by confidence
        sorted_directions = sorted(cv_directions.items(), key=lambda x: x[1], reverse=True)
        
        # Filter directions với confidence threshold
        valid_directions = [(dir_name, conf) for dir_name, conf in sorted_directions if conf >= 0.2]
        
        result_parts = []
        
        # Strategy: Map directions theo spatial logic và confidence
        if len(locations) >= 2 and len(valid_directions) >= 2:
            # Multi-direction, multi-location case
            
            # Phân tích pattern của locations để smart mapping
            location_patterns = self._analyze_location_patterns(locations)
            
            # Map theo priority: confidence cao + pattern phù hợp
            for i, (direction, confidence) in enumerate(valid_directions[:len(locations)]):
                if i < len(locations):
                    loc = locations[i]
                    
                    # Thêm logic để adjust direction dựa trên location pattern
                    adjusted_direction = self._adjust_direction_by_pattern(direction, loc, location_patterns)
                    
                    result_parts.append(f"{adjusted_direction} {loc['distance']}km đến {loc['name']}")
                    
                    if len(result_parts) >= 3:  # Limit to 3 directions max
                        break
        
        elif len(locations) >= 1:
            # Single hoặc limited locations
            best_direction, best_confidence = valid_directions[0] if valid_directions else ("đi thẳng", 0.5)
            
            for i, loc in enumerate(locations[:2]):  # Limit to 2 locations
                if i == 0:
                    result_parts.append(f"{best_direction} {loc['distance']}km đến {loc['name']}")
                else:
                    # Use second best direction nếu có
                    second_direction = valid_directions[1][0] if len(valid_directions) > 1 else best_direction
                    result_parts.append(f"{second_direction} {loc['distance']}km đến {loc['name']}")
        
        return " | ".join(result_parts) if result_parts else ""
    
    def _analyze_location_patterns(self, locations: list) -> dict:
        """Phân tích pattern của locations để intelligent mapping"""
        patterns = {
            'has_main_road': False,
            'has_short_distances': False,
            'has_long_distances': False,
            'distance_variance': 0.0
        }
        
        if not locations:
            return patterns
        
        # Analyze distances
        distances = []
        for loc in locations:
            try:
                dist = float(loc['distance'].replace(',', '.'))
                distances.append(dist)
            except:
                distances.append(1.0)  # Default
        
        if distances:
            patterns['distance_variance'] = np.var(distances) if len(distances) > 1 else 0.0
            patterns['has_short_distances'] = any(d <= 2.0 for d in distances)
            patterns['has_long_distances'] = any(d >= 50.0 for d in distances)
            
            # Detect main road pattern (long distance suggests main route)
            if patterns['has_long_distances'] and patterns['has_short_distances']:
                patterns['has_main_road'] = True
        
        return patterns
    
    def _adjust_direction_by_pattern(self, direction: str, location: dict, patterns: dict) -> str:
        """Adjust direction dựa trên pattern analysis"""
        try:
            distance = float(location['distance'].replace(',', '.'))
            
            # Logic: Long distance thường là đi thẳng (main road)
            if distance >= 50.0 and patterns['has_main_road']:
                if direction in ['rẽ trái', 'rẽ phải']:
                    # Có thể adjust to đi thẳng nếu confidence gap không lớn
                    return direction  # Keep original for now, có thể cải thiện sau
            
            # Short distance thường là turn (rẽ)
            elif distance <= 5.0:
                if direction == 'đi thẳng':
                    # Có thể adjust nhưng cần thận trọng
                    return direction  # Keep original
            
            return direction
            
        except:
            return direction
    
    def _map_directions_with_priority(self, cv_directions: dict, locations: list) -> str:
        """Map directions với priority-based logic"""
        if not cv_directions or not locations:
            return ""
        
        # Sort directions by confidence
        sorted_dirs = sorted(cv_directions.items(), key=lambda x: x[1], reverse=True)
        
        # Priority mapping
        result_parts = []
        used_directions = set()
        
        for i, loc in enumerate(locations[:3]):  # Max 3 locations
            if i < len(sorted_dirs):
                direction, confidence = sorted_dirs[i]
                if confidence >= 0.15 and direction not in used_directions:
                    result_parts.append(f"{direction} {loc['distance']}km đến {loc['name']}")
                    used_directions.add(direction)
            elif result_parts:  # Reuse best direction for remaining locations
                best_direction = sorted_dirs[0][0]
                result_parts.append(f"{best_direction} {loc['distance']}km đến {loc['name']}")
        
        return " | ".join(result_parts)
    
    def _enhance_nlp_with_cv_smart(self, nlp_text: str, cv_direction: str, cv_confidence: float) -> str:
        """Smart enhancement of NLP với CV direction"""
        if not nlp_text or cv_confidence < 0.2:
            return nlp_text
        
        # Kiểm tra nếu NLP đã có direction
        nlp_lower = nlp_text.lower()
        has_direction = any(keyword in nlp_lower for keyword in 
                           ["rẽ trái", "rẽ phải", "đi thẳng", "quay", "turn", "straight"])
        
        if not has_direction and cv_confidence >= 0.4:
            # Thêm CV direction vào đầu với confidence indicator
            confidence_indicator = "🔍" if cv_confidence >= 0.7 else ""
            return f"{confidence_indicator}{cv_direction} - {nlp_text}"
        
        return nlp_text

    def _has_direction_keywords(self, text: str) -> bool:
        """Kiểm tra xem text có chứa direction keywords không"""
        if not text:
            return False
        
        text_lower = text.lower()
        direction_keywords = [
            "rẽ trái", "rẽ phải", "đi thẳng", "quay trái", "quay phải",
            "turn left", "turn right", "go straight", "straight ahead",
            "belok kiri", "belok kanan", "lurus"
        ]
        
        return any(keyword in text_lower for keyword in direction_keywords)

    def _map_directions_to_locations(self, cv_directions: dict, locations: list) -> str:
        """Map CV directions với locations based on position analysis"""
        if len(locations) < 2 or len(cv_directions) < 2:
            return ""
        
        # Sort CV directions by confidence
        sorted_directions = sorted(cv_directions.items(), key=lambda x: x[1], reverse=True)
        
        # Map directions to locations (simple heuristic)
        result_parts = []
        direction_list = [item[0] for item in sorted_directions]
        
        for i, location in enumerate(locations[:len(direction_list)]):
            direction = direction_list[i] if i < len(direction_list) else direction_list[0]
            result_parts.append(f"{direction} {location['distance']}km đến {location['name']}")
        
        return " | ".join(result_parts)

    def _extract_direction_from_nlp(self, nlp_text: str) -> str:
        """Extract hướng di chuyển từ NLP text - updated để support multi-direction"""
        if not nlp_text:
            return "unknown"
        
        nlp_lower = nlp_text.lower()
        
        # Đếm số directions detected
        directions_found = []
        
        if any(keyword in nlp_lower for keyword in ["rẽ trái", "quay trái", "belok kiri", "turn left"]):
            directions_found.append("rẽ trái")
        if any(keyword in nlp_lower for keyword in ["rẽ phải", "quay phải", "belok kanan", "turn right"]):
            directions_found.append("rẽ phải")
        if any(keyword in nlp_lower for keyword in ["đi thẳng", "lurus", "straight", "go straight"]):
            directions_found.append("đi thẳng")
        
        # Nếu có nhiều directions, return first found (legacy compatibility)
        if directions_found:
            return directions_found[0]
        
        return "unknown"

    def _are_directions_consistent(self, cv_direction: str, nlp_direction: str) -> bool:
        """Kiểm tra 2 directions có consistent không"""
        if cv_direction == "unknown" or nlp_direction == "unknown":
            return False
        
        # Exact match
        if cv_direction == nlp_direction:
            return True
        
        # Cả 2 đều là "đi thẳng" variants
        straight_variants = ["đi thẳng", "straight", "lurus"]
        if (any(v in cv_direction.lower() for v in straight_variants) and 
            any(v in nlp_direction.lower() for v in straight_variants)):
            return True
        
        return False

    def _enhance_nlp_with_cv(self, nlp_text: str, cv_direction: str) -> str:
        """
        Enhance NLP text với thông tin direction từ CV
        """
        # Nếu NLP text đã chứa direction rõ ràng, giữ nguyên
        direction_keywords = ["rẽ trái", "rẽ phải", "đi thẳng", "quay trái", "quay phải"]
        if any(keyword in nlp_text.lower() for keyword in direction_keywords):
            return nlp_text
        
        # Nếu không có direction rõ ràng, thêm CV direction vào đầu
        if nlp_text and len(nlp_text.strip()) > 0:
            return f"{cv_direction} - {nlp_text}"
        else:
            return cv_direction

    def _check_cv_nlp_consistency(self, cv_direction: str, nlp_text: str) -> bool:
        """
        Kiểm tra consistency giữa CV direction và NLP text
        """
        nlp_lower = nlp_text.lower()
        
        # Mapping CV direction sang NLP keywords
        direction_mapping = {
            "rẽ trái": ["trái", "left", "←"],
            "rẽ phải": ["phải", "right", "→"], 
            "đi thẳng": ["thẳng", "straight", "↑", "lên", "ahead"]
        }
        
        if cv_direction in direction_mapping:
            keywords = direction_mapping[cv_direction]
            return any(keyword in nlp_lower for keyword in keywords)
        
        return False  # Unknown direction không consistent

    def smooth_label(self, class_idx, object_id, confidence):
        if object_id not in self.label_buffers:
            self.label_buffers[object_id] = deque(maxlen=self.buffer_size)
        if confidence >= self.config.CONFIDENCE_THRESHOLD - 0.1:
            self.label_buffers[object_id].append(class_idx)
        
        # Làm sạch cache cho các object đã biến mất
        current_track_ids = set(trk.id for trk in self.tracker.trackers)
        for obj_id in list(self.label_buffers.keys()):
            if obj_id not in current_track_ids:
                del self.label_buffers[obj_id]
                # Xóa toàn bộ cache liên quan đến object này
                self._clear_object_cache(obj_id)
        
        # Dọn dẹp cache hết hạn
        self._cleanup_expired_cache()
        
        most_common = Counter(self.label_buffers.get(object_id, [])).most_common(1)
        return most_common[0][0] if most_common else class_idx

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
        dets = []
        det_class_map = {}
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                bbox = box.xyxy[0].tolist()
                score = float(box.conf)
                class_idx = int(box.cls)
                dets.append(bbox + [score])
                det_class_map[len(dets) - 1] = {'class_idx': class_idx, 'confidence': score}
        if len(dets) == 0:
            trackers = np.empty((0, 5))
        else:
            trackers = self.tracker.update(np.array(dets))
        detections = []
        for trk in trackers:
            x1, y1, x2, y2, object_id = trk.astype(int)
            best_iou = -1
            matched_det_idx = -1
            for i, det_bbox_score in enumerate(dets):
                det_bbox = det_bbox_score[:4]
                current_iou = bbox_iou([x1,y1,x2,y2], det_bbox)
                if current_iou > best_iou:
                    best_iou = current_iou
                    matched_det_idx = i
            class_idx = 0
            confidence = 0.0
            if matched_det_idx != -1 and best_iou > 0.1:
                class_idx = det_class_map[matched_det_idx]['class_idx']
                confidence = det_class_map[matched_det_idx]['confidence']
            class_idx_smooth = self.smooth_label(class_idx, object_id, confidence)
            class_label = self.class_names[class_idx_smooth] if isinstance(self.class_names, list) and class_idx_smooth < len(self.class_names) else str(class_idx_smooth)
            class_label_vi = descriptions_vi_no_diacritics[class_idx_smooth] if class_idx_smooth < len(descriptions_vi_no_diacritics) else class_label
            detection = {
                'object_id': object_id,
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': class_idx_smooth,
                'class_label': class_label,
                'class_label_vi': class_label_vi,
                'ocr_text': None
            }
            if class_label == "sus":
                padding = 5
                crop_x1 = max(0, x1 - padding)
                crop_y1 = max(0, y1 - padding)
                crop_x2 = min(frame.shape[1], x2 + padding)
                crop_y2 = min(frame.shape[0], y2 + padding)
                
                # Kiểm tra crop hợp lệ
                if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
                    detection['ocr_text'] = "Invalid crop"
                    detections.append(detection)
                    continue
                
                current_bbox = [crop_x1, crop_y1, crop_x2, crop_y2]
                sign_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Tính hash của ảnh crop để xác định nội dung
                image_hash = self._compute_image_hash(sign_crop)
                if not image_hash:
                    detection['ocr_text'] = "Hash error"
                    detections.append(detection)
                    continue
                
                # Kiểm tra cache có còn hợp lệ không
                cache_valid = self._is_cache_valid(object_id, current_bbox, image_hash)
                
                if cache_valid and object_id in self.ocr_cache:
                    # Sử dụng cache hợp lệ
                    detection['ocr_text'] = self.ocr_cache[object_id]
                else:
                    # Cache không hợp lệ hoặc không tồn tại
                    if not cache_valid:
                        self._clear_object_cache(object_id)
                    
                    # Cập nhật cache với thông tin mới
                    self._update_object_cache(object_id, current_bbox, image_hash)
                    
                    # Gửi tới OCR queue nếu chưa gửi
                    if (object_id, ) not in self.ocr_queue_sent:
                        self.ocr_queue.put((object_id, sign_crop.copy()))
                        self.ocr_queue_sent.add((object_id, ))
                    
                    detection['ocr_text'] = "⏳ Hybrid Processing..."
            detections.append(detection)
        return detections

    def draw_and_show(self, frame: np.ndarray, detections):
        # Vẽ bounding box bằng OpenCV trước
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Chuyển frame sang PIL để vẽ text Unicode
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font_path = os.path.join(os.path.dirname(__file__), "arial.ttf")
        if not os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/arial.ttf"
        try:
            font = ImageFont.truetype(font_path, 20)
        except:
            font = ImageFont.load_default()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            # Lấy mã biển báo từ class_names thay vì dùng class_id_code hardcode
            class_code = det['class_label'] if 'class_label' in det else f"Class_{det['class_id']}"
            description_no_diacritics = det['class_label_vi'] if 'class_label_vi' in det else class_code
            confidence = det['confidence']
            
            # Format: Mã biển | Description_tieng_viet_khong_dai | Độ tin cậy
            label = f"{class_code} | {description_no_diacritics} | {confidence:.1%}"
            draw.text((x1, y1 - 25), label, font=font, fill=(0,255,0))
            
            # Chỉ hiển thị OCR khi có kết quả thành công (không phải "..." hay "⏳ Processing...")
            if (det.get('ocr_text') and 
                det['ocr_text'] != "..." and 
                not det['ocr_text'].startswith("⏳") and
                not det['ocr_text'].startswith("Invalid") and
                not det['ocr_text'].startswith("Hash error") and
                len(det['ocr_text'].strip()) > 0):
                ocr_label = f"HYBRID: {det['ocr_text']}"
                draw.text((x1, y2 + 5), ocr_label, font=font, fill=(255,165,0))  # Orange cho hybrid
                print(f"{class_code} | {description_no_diacritics} | HYBRID: {det['ocr_text']}")
        frame_show = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow('Traffic Sign Detection - Hybrid CV+NLP', frame_show)

    def run_webcam(self, cam_id=0, save_video=True):
        try:
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
                video_filename = os.path.join(output_dir, f"result_hybrid_webcam_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
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
                self.draw_and_show(frame_draw, detections)
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
        except Exception as e:
            import traceback
            print("[ERROR] Lỗi khi chạy webcam:", e)
            traceback.print_exc()

    def run_video(self, video_path, save_video=True):
        """Chạy detection trên video file"""
        try:
            if not os.path.exists(video_path):
                print(f"[ERROR] Không tìm thấy video: {video_path}")
                return
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ERROR] Không mở được video: {video_path}")
                return
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"[INFO] Video: {video_path}")
            print(f"[INFO] Độ phân giải: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")
            
            out = None
            if save_video:
                output_dir = 'real_time_output'
                os.makedirs(output_dir, exist_ok=True)
                video_filename = os.path.join(output_dir, f"result_hybrid_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
                print(f"[INFO] Kết quả sẽ được lưu tại: {video_filename}")
            
            frame_count = 0
            print("Nhấn 'q' để thoát, 'space' để tạm dừng/tiếp tục.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\n[INFO] Đã xử lý xong video!")
                    break
                
                frame_count += 1
                print(f"\r[PROCESSING] Frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)", end="")
                
                detections = self.predict_frame(frame)
                frame_draw = frame.copy()
                self.draw_and_show(frame_draw, detections)
                
                if out is not None:
                    out.write(frame_draw)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] Dừng bởi người dùng.")
                    break
                elif key == ord(' '):  # Space để pause/resume
                    print("\n[PAUSE] Nhấn space để tiếp tục...")
                    while True:
                        if cv2.waitKey(0) & 0xFF == ord(' '):
                            break
            
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            
            if out is not None:
                print(f"\n[INFO] Video kết quả đã lưu tại: {video_filename}")
                
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Lỗi khi xử lý video: {e}")
            traceback.print_exc()

def main():
    print("=" * 60)
    print("🚗 TRAFFIC SIGN DETECTION - HYBRID CV+NLP SYSTEM 🚗")
    print("=" * 60)
    print("Chọn chế độ chạy:")
    print("1. Webcam (Camera real-time)")
    print("2. Video test (test_video.mov từ thư mục input)")
    print("=" * 60)
    
    while True:
        try:
            choice = input("Nhập lựa chọn (1 hoặc 2): ").strip()
            
            if choice == "1":
                print("\n[DEBUG] Khởi động chương trình Hybrid CV+NLP - Chế độ Webcam")
                detector = RealTimeTrafficSignDetectorNLPHybrid()
                print("[DEBUG] Đã tạo detector Hybrid")
                detector.run_webcam(save_video=True)
                break
                
            elif choice == "2":
                # Đường dẫn video test
                video_path = os.path.join("input", "test_video.mov")
                
                # Kiểm tra file tồn tại
                if not os.path.exists(video_path):
                    print(f"\n[ERROR] Không tìm thấy video: {video_path}")
                    print("Vui lòng đảm bảo file test_video.mov có trong thư mục input/")
                    continue
                
                print(f"\n[DEBUG] Khởi động chương trình Hybrid CV+NLP - Chế độ Video Test")
                print(f"[DEBUG] Video: {video_path}")
                detector = RealTimeTrafficSignDetectorNLPHybrid()
                print("[DEBUG] Đã tạo detector Hybrid")
                detector.run_video(video_path, save_video=True)
                break
                
            else:
                print("❌ Lựa chọn không hợp lệ! Vui lòng nhập 1 hoặc 2.")
                continue
                
        except KeyboardInterrupt:
            print("\n\n[INFO] Chương trình bị ngắt bởi người dùng.")
            break
        except Exception as e:
            print(f"\n[ERROR] Lỗi: {e}")
            continue
    
    print("[DEBUG] Kết thúc chương trình Hybrid")

if __name__ == "__main__":
    main()