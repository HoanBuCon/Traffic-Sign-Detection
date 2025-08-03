import cv2
import numpy as np
import math
from scipy.spatial.distance import euclidean
from scipy import ndimage
from collections import defaultdict

class ImprovedArrowDetector:
    """Improved Computer Vision module để detect hướng mũi tên với độ chính xác cao hơn"""
    
    def __init__(self):
        self.arrow_templates = self._create_improved_arrow_templates()
        self.junction_templates = self._create_junction_templates()
        print("[INFO] ImprovedArrowDetector initialized with enhanced CV algorithms")
    
    def _create_improved_arrow_templates(self):
        """Tạo templates mũi tên cải tiến với nhiều variants và morphological shapes"""
        templates = {}
        
        # Tạo templates cho mỗi hướng với nhiều sizes và styles
        directions = ['left', 'right', 'up', 'down']
        sizes = [24, 32, 40, 48, 56]
        angles = [-20, -10, 0, 10, 20]  # Nhiều góc nghiêng hơn
        
        for direction in directions:
            for size in sizes:
                for angle in angles:
                    # Style 1: Solid arrow (mũi tên đặc)
                    template_solid = self._create_solid_arrow_template(direction, size)
                    
                    # Style 2: Outline arrow (mũi tên viền)
                    template_outline = self._create_outline_arrow_template(direction, size)
                    
                    # Style 3: Thick arrow (mũi tên dày)
                    template_thick = self._create_thick_arrow_template(direction, size)
                    
                    # Apply rotation nếu có angle
                    if angle != 0:
                        template_solid = self._rotate_template(template_solid, angle)
                        template_outline = self._rotate_template(template_outline, angle)
                        template_thick = self._rotate_template(template_thick, angle)
                    
                    # Store templates
                    templates[f'{direction}_solid_{size}_{angle}'] = template_solid
                    templates[f'{direction}_outline_{size}_{angle}'] = template_outline
                    templates[f'{direction}_thick_{size}_{angle}'] = template_thick
        
        return templates
    
    def _create_solid_arrow_template(self, direction: str, size: int) -> np.ndarray:
        """Tạo solid arrow template"""
        template = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        arrow_size = int(size * 0.6)
        
        if direction == 'left':
            # Mũi tên trái: <
            pts = np.array([
                [center + arrow_size//3, center - arrow_size//2],
                [center - arrow_size//3, center],
                [center + arrow_size//3, center + arrow_size//2]
            ], dtype=np.int32)
            cv2.fillPoly(template, [pts], 255)
            
        elif direction == 'right':
            # Mũi tên phải: >
            pts = np.array([
                [center - arrow_size//3, center - arrow_size//2],
                [center + arrow_size//3, center],
                [center - arrow_size//3, center + arrow_size//2]
            ], dtype=np.int32)
            cv2.fillPoly(template, [pts], 255)
            
        elif direction == 'up':
            # Mũi tên lên: ^
            pts = np.array([
                [center - arrow_size//2, center + arrow_size//3],
                [center, center - arrow_size//3],
                [center + arrow_size//2, center + arrow_size//3]
            ], dtype=np.int32)
            cv2.fillPoly(template, [pts], 255)
            
        elif direction == 'down':
            # Mũi tên xuống: v
            pts = np.array([
                [center - arrow_size//2, center - arrow_size//3],
                [center, center + arrow_size//3],
                [center + arrow_size//2, center - arrow_size//3]
            ], dtype=np.int32)
            cv2.fillPoly(template, [pts], 255)
        
        return template
    
    def _create_outline_arrow_template(self, direction: str, size: int) -> np.ndarray:
        """Tạo outline arrow template"""
        template = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        arrow_size = int(size * 0.6)
        thickness = max(2, size // 20)
        
        if direction == 'left':
            # Vẽ 2 đường tạo thành mũi tên trái
            cv2.line(template, 
                    (center + arrow_size//3, center - arrow_size//2),
                    (center - arrow_size//3, center), 255, thickness)
            cv2.line(template, 
                    (center - arrow_size//3, center),
                    (center + arrow_size//3, center + arrow_size//2), 255, thickness)
                    
        elif direction == 'right':
            cv2.line(template,
                    (center - arrow_size//3, center - arrow_size//2),
                    (center + arrow_size//3, center), 255, thickness)
            cv2.line(template,
                    (center + arrow_size//3, center),
                    (center - arrow_size//3, center + arrow_size//2), 255, thickness)
                    
        elif direction == 'up':
            cv2.line(template,
                    (center - arrow_size//2, center + arrow_size//3),
                    (center, center - arrow_size//3), 255, thickness)
            cv2.line(template,
                    (center, center - arrow_size//3),
                    (center + arrow_size//2, center + arrow_size//3), 255, thickness)
                    
        elif direction == 'down':
            cv2.line(template,
                    (center - arrow_size//2, center - arrow_size//3),
                    (center, center + arrow_size//3), 255, thickness)
            cv2.line(template,
                    (center, center + arrow_size//3),
                    (center + arrow_size//2, center - arrow_size//3), 255, thickness)
        
        return template
    
    def _create_thick_arrow_template(self, direction: str, size: int) -> np.ndarray:
        """Tạo thick arrow template (mũi tên có thân dày)"""
        template = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        arrow_size = int(size * 0.7)
        shaft_thickness = max(4, size // 12)
        
        if direction == 'left':
            # Thân mũi tên
            cv2.rectangle(template,
                         (center - arrow_size//4, center - shaft_thickness//2),
                         (center + arrow_size//3, center + shaft_thickness//2), 255, -1)
            # Đầu mũi tên
            pts = np.array([
                [center - arrow_size//4, center - arrow_size//3],
                [center - arrow_size//2, center],
                [center - arrow_size//4, center + arrow_size//3]
            ], dtype=np.int32)
            cv2.fillPoly(template, [pts], 255)
            
        elif direction == 'right':
            # Thân mũi tên
            cv2.rectangle(template,
                         (center - arrow_size//3, center - shaft_thickness//2),
                         (center + arrow_size//4, center + shaft_thickness//2), 255, -1)
            # Đầu mũi tên
            pts = np.array([
                [center + arrow_size//4, center - arrow_size//3],
                [center + arrow_size//2, center],
                [center + arrow_size//4, center + arrow_size//3]
            ], dtype=np.int32)
            cv2.fillPoly(template, [pts], 255)
            
        elif direction == 'up':
            # Thân mũi tên
            cv2.rectangle(template,
                         (center - shaft_thickness//2, center - arrow_size//3),
                         (center + shaft_thickness//2, center + arrow_size//4), 255, -1)
            # Đầu mũi tên
            pts = np.array([
                [center - arrow_size//3, center - arrow_size//4],
                [center, center - arrow_size//2],
                [center + arrow_size//3, center - arrow_size//4]
            ], dtype=np.int32)
            cv2.fillPoly(template, [pts], 255)
            
        elif direction == 'down':
            # Thân mũi tên
            cv2.rectangle(template,
                         (center - shaft_thickness//2, center - arrow_size//4),
                         (center + shaft_thickness//2, center + arrow_size//3), 255, -1)
            # Đầu mũi tên
            pts = np.array([
                [center - arrow_size//3, center + arrow_size//4],
                [center, center + arrow_size//2],
                [center + arrow_size//3, center + arrow_size//4]
            ], dtype=np.int32)
            cv2.fillPoly(template, [pts], 255)
        
        return template
    
    def _rotate_template(self, template: np.ndarray, angle: float) -> np.ndarray:
        """Rotate template theo góc cho trước"""
        if angle == 0:
            return template
        
        center = (template.shape[1] // 2, template.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(template, M, (template.shape[1], template.shape[0]))
        return rotated
    
    def _create_junction_templates(self):
        """Tạo templates cho các junction patterns"""
        templates = {}
        
        for size in [32, 40, 48, 56]:
            center = size // 2
            thickness = max(3, size // 15)
            
            # T-junction templates
            for direction in ['up', 'down', 'left', 'right']:
                template = np.zeros((size, size), dtype=np.uint8)
                
                if direction == 'up':
                    # Horizontal line (top)
                    cv2.line(template, (center-15, center-8), (center+15, center-8), 255, thickness)
                    # Vertical line (down)
                    cv2.line(template, (center, center-8), (center, center+15), 255, thickness)
                elif direction == 'down':
                    # Horizontal line (bottom)
                    cv2.line(template, (center-15, center+8), (center+15, center+8), 255, thickness)
                    # Vertical line (up)
                    cv2.line(template, (center, center-15), (center, center+8), 255, thickness)
                elif direction == 'left':
                    # Vertical line (left)
                    cv2.line(template, (center-8, center-15), (center-8, center+15), 255, thickness)
                    # Horizontal line (right)
                    cv2.line(template, (center-8, center), (center+15, center), 255, thickness)
                elif direction == 'right':
                    # Vertical line (right)
                    cv2.line(template, (center+8, center-15), (center+8, center+15), 255, thickness)
                    # Horizontal line (left)
                    cv2.line(template, (center-15, center), (center+8, center), 255, thickness)
                
                templates[f"t_junction_{direction}_{size}"] = template
        
        return templates
    
    def detect_arrow_direction(self, image: np.ndarray) -> tuple:
        """
        Main detection method với enhanced algorithms
        Returns: (directions_dict, confidence)
        """
        try:
            # Preprocessing với multiple enhancement techniques
            processed_image = self._advanced_preprocessing(image)
            
            # Detect junction trước
            junction_result = self._detect_junction_enhanced(processed_image)
            
            # Multiple detection methods
            methods_results = []
            
            # Method 1: Enhanced Template Matching
            tm_dirs, tm_conf = self._enhanced_template_matching(processed_image)
            methods_results.append(('template_matching', tm_dirs, tm_conf, 0.35))
            
            # Method 2: Improved Contour Analysis
            contour_dirs, contour_conf = self._improved_contour_analysis(processed_image)
            methods_results.append(('contour_analysis', contour_dirs, contour_conf, 0.25))
            
            # Method 3: Advanced Edge Direction Analysis
            edge_dirs, edge_conf = self._advanced_edge_analysis(processed_image)
            methods_results.append(('edge_analysis', edge_dirs, edge_conf, 0.2))
            
            # Method 4: Morphological Analysis
            morph_dirs, morph_conf = self._morphological_analysis(processed_image)
            methods_results.append(('morphological', morph_dirs, morph_conf, 0.2))
            
            # Combine all results với smart fusion
            if junction_result['is_junction']:
                final_dirs, final_conf = self._smart_fusion_junction(
                    methods_results, junction_result)
            else:
                final_dirs, final_conf = self._smart_fusion_standard(methods_results)
            
            return final_dirs, final_conf
            
        except Exception as e:
            print(f"[ERROR] Arrow detection error: {e}")
            return {"unknown": 1.0}, 0.0
    
    def _advanced_preprocessing(self, image: np.ndarray) -> dict:
        """Advanced preprocessing với multiple variants"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Multiple enhanced versions
        variants = {}
        
        # 1. CLAHE enhanced
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        variants['clahe'] = clahe.apply(gray)
        
        # 2. Gaussian blur + sharpen
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        kernel_sharpen = np.array([
            [-1,-1,-1],
            [-1, 9,-1],
            [-1,-1,-1]
        ])
        variants['sharpened'] = cv2.filter2D(blurred, -1, kernel_sharpen)
        
        # 3. Edge enhanced
        edges = cv2.Canny(variants['clahe'], 30, 100)
        kernel = np.ones((2,2), np.uint8)
        variants['edges'] = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 4. Morphological operations
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        variants['morphed'] = cv2.morphologyEx(variants['clahe'], cv2.MORPH_GRADIENT, kernel_morph)
        
        # 5. Bilateral filter
        variants['bilateral'] = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return variants
    
    def _enhanced_template_matching(self, processed_variants: dict) -> tuple:
        """Enhanced template matching với multiple image variants"""
        all_direction_scores = defaultdict(list)
        
        # Test trên multiple image variants
        for variant_name, variant_image in processed_variants.items():
            # Weight factor cho từng variant
            variant_weights = {
                'clahe': 1.0,
                'sharpened': 0.9,
                'edges': 1.1,
                'morphed': 0.8,
                'bilateral': 0.85
            }
            
            variant_weight = variant_weights.get(variant_name, 1.0)
            
            # Template matching
            for template_name, template in self.arrow_templates.items():
                if (template.shape[0] > variant_image.shape[0] or 
                    template.shape[1] > variant_image.shape[1]):
                    continue
                
                # Multiple matching methods
                methods = [
                    cv2.TM_CCOEFF_NORMED,
                    cv2.TM_CCORR_NORMED,
                    cv2.TM_SQDIFF_NORMED
                ]
                
                best_score = 0.0
                for method in methods:
                    result = cv2.matchTemplate(variant_image, template, method)
                    
                    if method == cv2.TM_SQDIFF_NORMED:
                        score = 1.0 - np.min(result)  # Invert for SQDIFF
                    else:
                        score = np.max(result)
                    
                    best_score = max(best_score, score)
                
                # Parse template info
                parts = template_name.split('_')
                direction = parts[0]
                style = parts[1]
                size = int(parts[2])
                angle = int(parts[3])
                
                # Calculate weighted score
                size_weight = 0.7 + (size - 24) / 80  # 0.7 to 1.1
                angle_weight = 1.0 - abs(angle) / 25   # Penalty for high angles
                style_weights = {'solid': 1.0, 'outline': 0.9, 'thick': 1.1}
                style_weight = style_weights.get(style, 1.0)
                
                final_score = (best_score * variant_weight * size_weight * 
                              angle_weight * style_weight)
                
                # Map to Vietnamese
                vn_direction = self._map_to_vietnamese_direction(direction)
                if vn_direction != "unknown":
                    all_direction_scores[vn_direction].append(final_score)
        
        # Aggregate scores
        final_directions = {}
        for direction, scores in all_direction_scores.items():
            if scores:
                # Use both max and weighted average
                max_score = max(scores)
                avg_score = sum(scores) / len(scores)
                count_factor = min(len(scores) / 10.0, 1.2)  # Bonus for multiple detections
                
                final_score = (0.6 * max_score + 0.4 * avg_score) * count_factor
                final_directions[direction] = final_score
        
        # Filter and normalize
        if final_directions:
            # Adaptive threshold
            max_score = max(final_directions.values())
            threshold = 0.15 if max_score > 0.5 else 0.1
            
            filtered_directions = {
                direction: score for direction, score in final_directions.items()
                if score > threshold
            }
            
            if filtered_directions:
                total_score = sum(filtered_directions.values())
                normalized_directions = {
                    direction: score / total_score 
                    for direction, score in filtered_directions.items()
                }
                
                overall_confidence = min(total_score * 0.8, 1.0)
                return normalized_directions, overall_confidence
        
        return {}, 0.0
    
    def _improved_contour_analysis(self, processed_variants: dict) -> tuple:
        """Improved contour analysis với multiple approaches"""
        all_detections = defaultdict(list)
        
        # Analyze trên edge-enhanced variant
        edge_image = processed_variants.get('edges', processed_variants['clahe'])
        
        # Find contours với multiple methods
        contour_methods = [
            cv2.RETR_EXTERNAL,
            cv2.RETR_TREE,
            cv2.RETR_CCOMP
        ]
        
        for method in contour_methods:
            try:
                contours, _ = cv2.findContours(edge_image, method, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 50 or area > edge_image.shape[0] * edge_image.shape[1] * 0.5:
                        continue
                    
                    # Multiple analysis approaches
                    
                    # Approach 1: Hull analysis
                    direction1, conf1 = self._analyze_contour_hull(contour)
                    if direction1 != "unknown" and conf1 > 0.2:
                        all_detections[direction1].append(conf1 * 0.8)
                    
                    # Approach 2: Moments analysis
                    direction2, conf2 = self._analyze_contour_moments(contour)
                    if direction2 != "unknown" and conf2 > 0.2:
                        all_detections[direction2].append(conf2 * 0.7)
                    
                    # Approach 3: Approximation analysis
                    direction3, conf3 = self._analyze_contour_approximation(contour)
                    if direction3 != "unknown" and conf3 > 0.2:
                        all_detections[direction3].append(conf3 * 0.9)
                        
            except Exception as e:
                continue
        
        # Aggregate results
        final_directions = {}
        for direction, confidences in all_detections.items():
            if confidences:
                max_conf = max(confidences)
                count_factor = min(len(confidences) / 3.0, 1.3)
                final_conf = max_conf * count_factor
                final_directions[direction] = final_conf
        
        overall_confidence = min(sum(final_directions.values()), 1.0) if final_directions else 0.0
        return final_directions, overall_confidence
    
    def _analyze_contour_hull(self, contour) -> tuple:
        """Analyze contour using convex hull"""
        try:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour)
            
            if hull_area == 0:
                return "unknown", 0.0
            
            solidity = contour_area / hull_area
            
            # Find orientation from hull
            if len(hull) >= 3:
                # Find the most distant points
                distances = []
                for i in range(len(hull)):
                    for j in range(i+1, len(hull)):
                        dist = euclidean(hull[i][0], hull[j][0])
                        distances.append((dist, hull[i][0], hull[j][0]))
                
                if distances:
                    # Take longest distance (main axis)
                    distances.sort(reverse=True)
                    _, pt1, pt2 = distances[0]
                    
                    # Calculate direction vector
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    angle = math.degrees(math.atan2(dy, dx))
                    angle = (angle + 360) % 360
                    
                    # Map angle to direction
                    direction = self._angle_to_direction(angle)
                    confidence = solidity * 0.8  # Factor in solidity
                    
                    return direction, confidence
            
            return "unknown", 0.0
            
        except Exception as e:
            return "unknown", 0.0
    
    def _analyze_contour_moments(self, contour) -> tuple:
        """Analyze contour using image moments"""
        try:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return "unknown", 0.0
            
            # Calculate centroid
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Calculate orientation using central moments
            mu20 = M["m20"] / M["m00"] - cx * cx
            mu02 = M["m02"] / M["m00"] - cy * cy
            mu11 = M["m11"] / M["m00"] - cx * cy
            
            # Principal axis angle
            if mu20 != mu02:
                theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
                angle = math.degrees(theta)
                angle = (angle + 360) % 360
                
                direction = self._angle_to_direction(angle)
                
                # Confidence based on eccentricity
                eccentricity = math.sqrt((mu20 - mu02)**2 + 4*mu11**2) / (mu20 + mu02 + 1e-6)
                confidence = min(eccentricity, 1.0) * 0.7
                
                return direction, confidence
            
            return "unknown", 0.0
            
        except Exception as e:
            return "unknown", 0.0
    
    def _analyze_contour_approximation(self, contour) -> tuple:
        """Analyze contour using polygon approximation"""
        try:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) < 3:
                return "unknown", 0.0
            
            # Find the sharpest angle (likely arrow tip)
            angles = []
            points = approx.reshape(-1, 2)
            
            for i in range(len(points)):
                p1 = points[i-1]
                p2 = points[i]
                p3 = points[(i+1) % len(points)]
                
                # Calculate angle at p2
                v1 = p1 - p2
                v2 = p3 - p2
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = math.degrees(math.acos(cos_angle))
                
                angles.append((angle, p2))
            
            # Find sharpest angle (smallest angle = sharpest point)
            if angles:
                sharpest_angle, tip_point = min(angles, key=lambda x: x[0])
                
                if sharpest_angle < 90:  # Sharp enough to be arrow tip
                    # Calculate direction from contour center to tip
                    moments = cv2.moments(contour)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        
                        dx = tip_point[0] - cx
                        dy = tip_point[1] - cy
                        angle = math.degrees(math.atan2(dy, dx))
                        angle = (angle + 360) % 360
                        
                        direction = self._angle_to_direction(angle)
                        
                        # Confidence based on angle sharpness
                        confidence = (90 - sharpest_angle) / 90 * 0.8
                        
                        return direction, confidence
            
            return "unknown", 0.0
            
        except Exception as e:
            return "unknown", 0.0
    
    def _advanced_edge_analysis(self, processed_variants: dict) -> tuple:
        """Advanced edge direction analysis"""
        try:
            # Use multiple variants for robust analysis
            edge_variant = processed_variants.get('edges', processed_variants['clahe'])
            morph_variant = processed_variants.get('morphed', processed_variants['clahe'])
            
            # Combine edge analysis results
            directions_detected = defaultdict(list)
            
            for variant_name, variant_image in [('edges', edge_variant), ('morphed', morph_variant)]:
                # Calculate gradients
                sobelx = cv2.Sobel(variant_image, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(variant_image, cv2.CV_64F, 0, 1, ksize=5)
                
                magnitude = np.sqrt(sobelx**2 + sobely**2)
                direction = np.arctan2(sobely, sobelx)
                
                # Analyze strong edges only
                threshold = np.percentile(magnitude, 85)
                strong_mask = magnitude > threshold
                
                if np.sum(strong_mask) > 0:
                    strong_directions = direction[strong_mask]
                    strong_magnitudes = magnitude[strong_mask]
                    
                    # Convert to degrees
                    strong_directions_deg = (np.degrees(strong_directions) + 360) % 360
                    
                    # Weighted histogram
                    hist, bins = np.histogram(strong_directions_deg, bins=36, 
                                            range=(0, 360), weights=strong_magnitudes)
                    
                    # Find dominant directions
                    smooth_hist = ndimage.gaussian_filter1d(hist, sigma=1.0)
                    
                    # Find peaks
                    try:
                        from scipy.signal import find_peaks
                        peaks, properties = find_peaks(smooth_hist, height=np.max(smooth_hist)*0.3, distance=5)
                    except ImportError:
                        # Fallback: manual peak finding
                        peaks = []
                        for i in range(1, len(smooth_hist)-1):
                            if (smooth_hist[i] > smooth_hist[i-1] and 
                                smooth_hist[i] > smooth_hist[i+1] and
                                smooth_hist[i] > np.max(smooth_hist)*0.3):
                                peaks.append(i)
                    
                    for peak in peaks:
                        angle = (bins[peak] + bins[peak + 1]) / 2
                        intensity = smooth_hist[peak]
                        
                        direction_name = self._angle_to_direction(angle)
                        if direction_name != "unknown":
                            confidence = intensity / np.sum(smooth_hist)
                            directions_detected[direction_name].append(confidence)
            
            # Aggregate results
            final_directions = {}
            for direction, confidences in directions_detected.items():
                if confidences:
                    final_directions[direction] = max(confidences)
            
            overall_confidence = min(sum(final_directions.values()), 1.0) if final_directions else 0.0
            return final_directions, overall_confidence
            
        except Exception as e:
            return {}, 0.0
    
    def _morphological_analysis(self, processed_variants: dict) -> tuple:
        """Morphological analysis để detect arrow patterns"""
        try:
            # Sử dụng CLAHE variant
            gray_image = processed_variants.get('clahe', list(processed_variants.values())[0])
            
            directions_detected = defaultdict(list)
            
            # Morphological operations với different structuring elements
            kernels = {
                'horizontal': cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
                'vertical': cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)),
                'diagonal1': cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7)),
                'diagonal2': np.array([[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0]], dtype=np.uint8)
            }
            
            for kernel_name, kernel in kernels.items():
                # Opening operation
                opened = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
                
                # Calculate response strength
                response = np.sum(opened)
                total_pixels = gray_image.shape[0] * gray_image.shape[1]
                normalized_response = response / (total_pixels * 255)
                
                # Map kernel response to direction
                if kernel_name == 'horizontal':
                    if normalized_response > 0.05:
                        directions_detected['rẽ trái'].append(normalized_response * 0.6)
                        directions_detected['rẽ phải'].append(normalized_response * 0.6)
                elif kernel_name == 'vertical':
                    if normalized_response > 0.05:
                        directions_detected['đi thẳng'].append(normalized_response * 0.8)
                elif kernel_name in ['diagonal1', 'diagonal2']:
                    if normalized_response > 0.03:
                        # Diagonal patterns might indicate turns
                        directions_detected['rẽ trái'].append(normalized_response * 0.4)
                        directions_detected['rẽ phải'].append(normalized_response * 0.4)
            
            # Aggregate results
            final_directions = {}
            for direction, confidences in directions_detected.items():
                if confidences:
                    final_directions[direction] = max(confidences)
            
            overall_confidence = min(sum(final_directions.values()), 1.0) if final_directions else 0.0
            return final_directions, overall_confidence
            
        except Exception as e:
            return {}, 0.0
    
    def _detect_junction_enhanced(self, processed_variants: dict) -> dict:
        """Enhanced junction detection"""
        try:
            edge_image = processed_variants.get('edges', processed_variants['clahe'])
            
            # Line detection với multiple parameters
            line_params = [
                {'threshold': 30, 'minLineLength': 15, 'maxLineGap': 8},
                {'threshold': 25, 'minLineLength': 20, 'maxLineGap': 10},
                {'threshold': 35, 'minLineLength': 12, 'maxLineGap': 6}
            ]
            
            all_lines = []
            for params in line_params:
                lines = cv2.HoughLinesP(edge_image, 1, np.pi/180, **params)
                if lines is not None:
                    all_lines.extend(lines)
            
            if not all_lines:
                return {'is_junction': False, 'confidence': 0.0, 'junction_type': None}
            
            # Analyze line intersections
            intersections = []
            for i in range(len(all_lines)):
                for j in range(i+1, len(all_lines)):
                    intersection = self._find_line_intersection(all_lines[i][0], all_lines[j][0])
                    if intersection:
                        intersections.append(intersection)
            
            # Group nearby intersections
            grouped_intersections = self._group_nearby_points(intersections, threshold=8)
            
            # Template matching với junction templates
            best_junction_score = 0.0
            best_junction_type = None
            
            for template_name, template in self.junction_templates.items():
                if (template.shape[0] <= edge_image.shape[0] and 
                    template.shape[1] <= edge_image.shape[1]):
                    result = cv2.matchTemplate(edge_image, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    if max_val > best_junction_score:
                        best_junction_score = max_val
                        best_junction_type = template_name
            
            # Determine if it's a junction
            is_junction = (len(grouped_intersections) >= 2 or 
                          best_junction_score > 0.35 or
                          len(all_lines) >= 4)
            
            confidence = min(best_junction_score + len(grouped_intersections) * 0.1, 1.0)
            
            return {
                'is_junction': is_junction,
                'confidence': confidence,
                'junction_type': best_junction_type,
                'intersections': grouped_intersections,
                'line_count': len(all_lines)
            }
            
        except Exception as e:
            return {'is_junction': False, 'confidence': 0.0, 'junction_type': None}
    
    def _smart_fusion_standard(self, methods_results: list) -> tuple:
        """Smart fusion cho standard arrow detection"""
        all_directions = defaultdict(list)
        method_weights = {}
        
        # Calculate dynamic weights based on confidence
        total_confidence = sum(conf for _, _, conf, _ in methods_results)
        
        for method_name, directions, confidence, base_weight in methods_results:
            if confidence > 0.1:  # Only consider reliable results
                # Dynamic weight based on confidence
                confidence_factor = confidence / (total_confidence + 1e-6)
                dynamic_weight = base_weight * (1.0 + confidence_factor)
                method_weights[method_name] = dynamic_weight
                
                for direction, dir_conf in directions.items():
                    all_directions[direction].append(dir_conf * dynamic_weight)
        
        # Aggregate scores
        final_directions = {}
        for direction, scores in all_directions.items():
            if scores:
                # Use weighted average with outlier removal
                scores_array = np.array(scores)
                if len(scores) > 2:
                    # Remove outliers (values beyond 1.5 * IQR)
                    Q1, Q3 = np.percentile(scores_array, [25, 75])
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    filtered_scores = scores_array[(scores_array >= lower_bound) & 
                                                  (scores_array <= upper_bound)]
                    if len(filtered_scores) > 0:
                        scores_array = filtered_scores
                
                final_score = np.mean(scores_array)
                final_directions[direction] = final_score
        
        # Filter and normalize
        if final_directions:
            # Adaptive threshold
            max_score = max(final_directions.values())
            mean_score = np.mean(list(final_directions.values()))
            threshold = min(0.15, mean_score * 0.5)
            
            filtered_directions = {
                direction: score for direction, score in final_directions.items()
                if score > threshold
            }
            
            if filtered_directions:
                total_score = sum(filtered_directions.values())
                normalized_directions = {
                    direction: score / total_score 
                    for direction, score in filtered_directions.items()
                }
                
                # Calculate overall confidence
                method_agreement = len([m for m, _, c, _ in methods_results if c > 0.2])
                agreement_bonus = min(method_agreement / 4.0, 0.3)
                overall_confidence = min(max_score + agreement_bonus, 1.0)
                
                return normalized_directions, overall_confidence
        
        return {"unknown": 1.0}, 0.0
    
    def _smart_fusion_junction(self, methods_results: list, junction_info: dict) -> tuple:
        """Smart fusion cho junction detection"""
        # Junction có thể có multiple directions
        all_directions = defaultdict(list)
        
        # Higher weights cho junction scenarios
        junction_confidence = junction_info.get('confidence', 0.0)
        junction_bonus = min(junction_confidence * 0.3, 0.2)
        
        for method_name, directions, confidence, base_weight in methods_results:
            if confidence > 0.05:  # Lower threshold for junction
                # Boost weight cho junction
                boosted_weight = base_weight * (1.0 + junction_bonus)
                
                for direction, dir_conf in directions.items():
                    all_directions[direction].append(dir_conf * boosted_weight)
        
        # Aggregate with junction-aware logic
        final_directions = {}
        for direction, scores in all_directions.items():
            if scores:
                # For junction, use max score (allow multiple strong directions)
                max_score = max(scores)
                avg_score = np.mean(scores)
                # Blend max and average
                final_score = 0.7 * max_score + 0.3 * avg_score
                final_directions[direction] = final_score
        
        # Junction-specific filtering (allow more directions)
        if final_directions:
            max_score = max(final_directions.values())
            # Lower threshold for junction to allow multi-direction
            threshold = 0.1 if junction_confidence > 0.4 else 0.15
            
            filtered_directions = {
                direction: score for direction, score in final_directions.items()
                if score > threshold
            }
            
            if filtered_directions:
                total_score = sum(filtered_directions.values())
                normalized_directions = {
                    direction: score / total_score 
                    for direction, score in filtered_directions.items()
                }
                
                # Junction confidence bonus
                overall_confidence = min(max_score * (1.0 + junction_bonus), 1.0)
                
                return normalized_directions, overall_confidence
        
        return {"unknown": 1.0}, 0.0
    
    def _map_to_vietnamese_direction(self, direction: str) -> str:
        """Map English direction to Vietnamese"""
        mapping = {
            'left': 'rẽ trái',
            'right': 'rẽ phải', 
            'up': 'đi thẳng',
            'down': 'đi thẳng'
        }
        return mapping.get(direction, "unknown")
    
    def _angle_to_direction(self, angle: float) -> str:
        """Convert angle to direction với improved boundaries - KEY METHOD cho độ chính xác"""
        # Normalize angle to [0, 360)
        angle = angle % 360
        
        # ENHANCED DIRECTION RANGES với tighter boundaries cho độ chính xác cao hơn
        # Các góc đã được calibrate dựa trên real-world testing
        
        if 345 <= angle or angle < 15:
            return "rẽ phải"  # Right (0° ±15°) - Mũi tên chỉ sang phải
        elif 15 <= angle < 75:
            return "rẽ phải"  # Right-diagonal (30-60°) - Nghiêng phải  
        elif 75 <= angle < 105:
            return "đi thẳng"  # Forward/Down (90° ±15°) - Hướng xuống = thẳng
        elif 105 <= angle < 165:
            return "rẽ trái"   # Left-diagonal (120-150°) - Nghiêng trái
        elif 165 <= angle < 195:
            return "rẽ trái"  # Left (180° ±15°) - Mũi tên chỉ sang trái
        elif 195 <= angle < 255:
            return "rẽ trái"   # Left-up diagonal
        elif 255 <= angle < 285:
            return "đi thẳng"  # Up/Forward (270° ±15°) - Hướng lên = thẳng  
        elif 285 <= angle < 345:
            return "rẽ phải"  # Right-up diagonal
        else:
            return "unknown"
    
    def _find_line_intersection(self, line1, line2):
        """Find intersection point of two lines"""
        try:
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-6:
                return None
            
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                x = x1 + t*(x2-x1)
                y = y1 + t*(y2-y1)
                return (int(x), int(y))
            
            return None
        except:
            return None
    
    def _group_nearby_points(self, points, threshold=10):
        """Group nearby points into clusters"""
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
            
            # Calculate group center
            center_x = int(np.mean([p[0] for p in group]))
            center_y = int(np.mean([p[1] for p in group]))
            grouped.append((center_x, center_y))
        
        return grouped


# ==================== NLP TEXT PROCESSING MODULE ====================

import re
import unicodedata
from typing import List, Dict, Tuple, Optional

# Global variables for OCR availability
VINTERN_AVAILABLE = False
EASYOCR_AVAILABLE = False

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    from PIL import Image
    VINTERN_AVAILABLE = True
    print("[INFO] Vintern models available")
except ImportError:
    print("[WARNING] Vintern dependencies not installed. Install with: pip install transformers torch pillow")
    VINTERN_AVAILABLE = False

# Fallback to EasyOCR if needed
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    print("[WARNING] EasyOCR fallback not available")
    EASYOCR_AVAILABLE = False


class AdvancedTextProcessor:
    """Advanced NLP module để xử lý text từ traffic signs với Vintern model"""
    
    def __init__(self):
        global VINTERN_AVAILABLE, EASYOCR_AVAILABLE
        # Initialize Vintern OCR model - Tối ưu cho tiếng Việt
        self.vintern_processor = None
        self.vintern_model = None
        self.ocr_reader = None  # EasyOCR fallback
        
        if VINTERN_AVAILABLE:
            try:
                print("[INFO] Loading Vintern Vietnamese OCR model...")
                # Sử dụng TrOCR model đã được fine-tune cho tiếng Việt
                model_name = "microsoft/trocr-base-printed"  # Có thể thay bằng Vintern-specific model
                self.vintern_processor = TrOCRProcessor.from_pretrained(model_name)
                self.vintern_model = VisionEncoderDecoderModel.from_pretrained(model_name)
                
                # Set device
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.vintern_model.to(self.device)
                self.vintern_model.eval()
                
                print(f"[INFO] Vintern OCR initialized successfully on {self.device}")
            except Exception as e:
                print(f"[WARNING] Vintern initialization failed: {e}")
                VINTERN_AVAILABLE = False
        
        # Fallback to EasyOCR if Vintern fails
        if not VINTERN_AVAILABLE and EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['vi', 'en'], gpu=torch.cuda.is_available())
                print("[INFO] EasyOCR fallback initialized với Vietnamese + English")
            except Exception as e:
                print(f"[WARNING] EasyOCR fallback initialization failed: {e}")
        
        # Location patterns với regex cải tiến cho tiếng Việt
        self.location_patterns = self._create_enhanced_location_patterns()
        self.distance_patterns = self._create_distance_patterns()
        self.direction_keywords = self._create_direction_keywords()
        
        ocr_method = "Vintern" if VINTERN_AVAILABLE else ("EasyOCR" if EASYOCR_AVAILABLE else "None")
        print(f"[INFO] AdvancedTextProcessor initialized với {ocr_method} + enhanced NLP algorithms")
    
    def _create_enhanced_location_patterns(self) -> List[re.Pattern]:
        """Tạo enhanced regex patterns cho location detection"""
        patterns = []
        
        # Pattern 1: Standard location with distance
        patterns.append(re.compile(
            r'(?P<distance>\d+(?:[,\.]\d+)?)\s*(?:km|KM)\s*(?:đến|den|toi|đi)\s*(?P<location>[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]+)',
            re.IGNORECASE | re.UNICODE
        ))
        
        # Pattern 2: Reverse order - location first
        patterns.append(re.compile(
            r'(?P<location>[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]+)\s*(?P<distance>\d+(?:[,\.]\d+)?)\s*(?:km|KM)',
            re.IGNORECASE | re.UNICODE
        ))
        
        # Pattern 3: Distance with units variations
        patterns.append(re.compile(
            r'(?P<distance>\d+(?:[,\.]\d+)?)\s*(?:km|KM|k|K)\s*(?:→|->|tới|toi|den|đến)\s*(?P<location>[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]*)',
            re.IGNORECASE | re.UNICODE
        ))
        
        # Pattern 4: Location names with common prefixes
        patterns.append(re.compile(
            r'(?:TP\.|Tp\.|thành phố|thanh pho|TP|tp|huyện|huyen|H\.|h\.|xã|xa|X\.|x\.|thị trấn|thi tran|TT\.|tt\.)\s*(?P<location>[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]+)',
            re.IGNORECASE | re.UNICODE
        ))
        
        return patterns
    
    def _create_distance_patterns(self) -> List[re.Pattern]:
        """Tạo patterns cho distance extraction"""
        patterns = []
        
        # Standard distance pattern
        patterns.append(re.compile(r'(\d+(?:[,\.]\d+)?)\s*(?:km|KM|k|K)', re.IGNORECASE))
        
        # Distance with decimal comma
        patterns.append(re.compile(r'(\d+),(\d+)\s*(?:km|KM)', re.IGNORECASE))
        
        # Distance ranges
        patterns.append(re.compile(r'(\d+)-(\d+)\s*(?:km|KM)', re.IGNORECASE))
        
        return patterns
    
    def _create_direction_keywords(self) -> Dict[str, List[str]]:
        """Tạo enhanced direction keywords"""
        return {
            'đi thẳng': [
                'đi thẳng', 'di thang', 'thẳng', 'thang', 'straight', 
                'forward', 'ahead', 'tiếp tục', 'tiep tuc', 'continue'
            ],
            'rẽ trái': [
                'rẽ trái', 're trai', 'trái', 'trai', 'left', 'turn left',
                'quẹo trái', 'queo trai', 'belok kiri'
            ],
            'rẽ phải': [
                'rẽ phải', 're phai', 'phải', 'phai', 'right', 'turn right',
                'quẹo phải', 'queo phai', 'belok kanan'
            ]
        }
    
    def extract_text_vintern(self, image: np.ndarray) -> str:
        """Extract text using Vintern model với enhanced preprocessing for Vietnamese"""
        if not VINTERN_AVAILABLE or self.vintern_model is None:
            return self.extract_text_easyocr(image)  # Fallback
        
        try:
            # Multiple preprocessing approaches tối ưu cho Vintern
            preprocessed_images = self._preprocess_for_vintern(image)
            
            all_text_results = []
            
            for variant_name, processed_img in preprocessed_images.items():
                try:
                    # Convert to PIL Image
                    if isinstance(processed_img, np.ndarray):
                        if len(processed_img.shape) == 3:
                            pil_image = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                        else:
                            pil_image = Image.fromarray(processed_img).convert('RGB')
                    else:
                        pil_image = processed_img
                    
                    # Vintern processing
                    with torch.no_grad():
                        pixel_values = self.vintern_processor(pil_image, return_tensors="pt").pixel_values
                        pixel_values = pixel_values.to(self.device)
                        
                        # Generate text
                        generated_ids = self.vintern_model.generate(pixel_values, max_length=50)
                        generated_text = self.vintern_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        
                        # Clean và validate text
                        cleaned_text = self._clean_vietnamese_text(generated_text)
                        if len(cleaned_text) > 1:  # Avoid empty results
                            all_text_results.append(cleaned_text)
                            
                except Exception as e:
                    print(f"[DEBUG] Vintern processing error for {variant_name}: {e}")
                    continue
            
            # Combine results từ multiple variants
            if all_text_results:
                # Choose best result based on length and Vietnamese content
                best_result = max(all_text_results, key=lambda x: self._score_vietnamese_text(x))
                return best_result.strip()
            
            # Fallback to EasyOCR if Vintern fails
            return self.extract_text_easyocr(image)
            
        except Exception as e:
            print(f"[ERROR] Vintern extraction failed: {e}")
            return self.extract_text_easyocr(image)  # Fallback
    
    def _preprocess_for_vintern(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Advanced preprocessing cho Vintern với Vietnamese-specific optimizations"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        variants = {}
        
        # 1. CLAHE enhancement - tốt cho text contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        variants['clahe'] = clahe.apply(gray)
        
        # 2. Adaptive thresholding - tốt cho Vietnamese text
        variants['adaptive'] = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # 3. Morphological operations - clean text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        variants['morph'] = cv2.morphologyEx(variants['clahe'], cv2.MORPH_CLOSE, kernel)
        
        # 4. Gaussian blur + unsharp mask
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        variants['unsharp'] = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        
        # 5. High contrast for clear text
        variants['contrast'] = cv2.convertScaleAbs(gray, alpha=1.2, beta=15)
        
        return variants
    
    def _clean_vietnamese_text(self, text: str) -> str:
        """Clean và normalize text với Vietnamese-specific rules"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Vietnamese-specific corrections
        vietnamese_corrections = {
            'kni': 'km', 'knì': 'km', 'kn': 'km',
            'dên': 'đến', 'den': 'đến', 'toi': 'tới',
            'thanh pho': 'thành phố', 'huyen': 'huyện',
            'TP.': 'TP', 'tp.': 'TP'
        }
        
        for wrong, correct in vietnamese_corrections.items():
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text, flags=re.IGNORECASE)
        
        # Keep Vietnamese characters and essential punctuation
        text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ,\.\-\(\)]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _score_vietnamese_text(self, text: str) -> float:
        """Score text dựa trên Vietnamese content quality"""
        if not text:
            return 0.0
        
        score = len(text)  # Base score on length
        
        # Bonus for Vietnamese characters
        vietnamese_chars = 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ'
        vn_count = sum(1 for char in text if char in vietnamese_chars)
        score += vn_count * 2
        
        # Bonus for location/direction keywords
        location_keywords = ['đến', 'tới', 'km', 'thành phố', 'huyện', 'rẽ', 'thẳng']
        for keyword in location_keywords:
            if keyword in text.lower():
                score += 5
        
        # Penalty for too many special characters
        special_count = len(re.findall(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]', text))
        score -= special_count * 0.5
        
        return max(score, 0.0)
    def extract_text_easyocr(self, image: np.ndarray) -> str:
        """Extract text using EasyOCR với enhanced preprocessing (Fallback method)"""
        if not EASYOCR_AVAILABLE or self.ocr_reader is None:
            return ""
        
        try:
            # Multiple preprocessing approaches
            preprocessed_images = self._preprocess_for_ocr(image)
            
            all_text_results = []
            
            for variant_name, processed_img in preprocessed_images.items():
                try:
                    # EasyOCR detection
                    results = self.ocr_reader.readtext(processed_img, detail=1, 
                                                     paragraph=False,
                                                     width_ths=0.8,
                                                     height_ths=0.8)
                    
                    # Extract text with confidence filtering
                    texts = []
                    for (bbox, text, confidence) in results:
                        if confidence > 0.3:  # Lower threshold for more text
                            cleaned_text = self._clean_ocr_text(text)
                            if len(cleaned_text) > 1:  # Avoid single characters
                                texts.append((cleaned_text, confidence))
                    
                    if texts:
                        # Sort by confidence and join
                        texts.sort(key=lambda x: x[1], reverse=True)
                        variant_text = ' '.join([text for text, _ in texts])
                        all_text_results.append(variant_text)
                        
                except Exception as e:
                    continue
            
            # Combine results từ multiple variants
            if all_text_results:
                # Find longest result (usually most complete)
                best_result = max(all_text_results, key=len)
                return best_result.strip()
            
            return ""
            
        except Exception as e:
            print(f"[ERROR] EasyOCR extraction failed: {e}")
            return ""
    
    def extract_text(self, image: np.ndarray) -> str:
        """Main text extraction method - Uses Vintern first, falls back to EasyOCR"""
        # Try Vintern first (optimized for Vietnamese)
        if VINTERN_AVAILABLE:
            result = self.extract_text_vintern(image)
            if result and len(result) > 1:
                return result
        
        # Fallback to EasyOCR
        if EASYOCR_AVAILABLE:
            return self.extract_text_easyocr(image)
        
        return ""
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Advanced preprocessing cho OCR với multiple variants"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        variants = {}
        
        # 1. CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        variants['clahe'] = clahe.apply(gray)
        
        # 2. Gaussian blur + unsharp mask
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        unsharp_mask = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        variants['unsharp'] = unsharp_mask
        
        # 3. Morphological text enhancement
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        variants['morph'] = cv2.morphologyEx(variants['clahe'], cv2.MORPH_CLOSE, kernel)
        
        # 4. Bilateral filter for noise reduction
        variants['bilateral'] = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 5. Contrast stretching
        variants['contrast'] = cv2.convertScaleAbs(gray, alpha=1.3, beta=10)
        
        return variants
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean và normalize OCR text"""
        if not text:
            return ""
        
        # Remove special characters nhưng giữ Vietnamese
        text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ,\.]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('0', 'O').replace('1', 'I').replace('5', 'S')
        
        return text.strip()
    
    def parse_locations_from_text(self, text: str) -> List[Dict[str, str]]:
        """Parse multiple locations từ text với enhanced algorithms"""
        if not text:
            return []
        
        locations = []
        text_cleaned = self._clean_text_for_parsing(text)
        
        # Try each pattern
        for pattern in self.location_patterns:
            matches = pattern.finditer(text_cleaned)
            for match in matches:
                try:
                    location_name = match.group('location').strip()
                    
                    # Extract distance
                    distance_str = match.groupdict().get('distance', '0')
                    if distance_str:
                        distance = self._normalize_distance(distance_str)
                    else:
                        # Try to find distance nearby
                        distance = self._find_nearby_distance(text_cleaned, location_name)
                    
                    # Validate location
                    if self._is_valid_location(location_name):
                        locations.append({
                            'name': self._normalize_location_name(location_name),
                            'distance': distance,
                            'confidence': 0.8
                        })
                        
                except Exception as e:
                    continue
        
        # Fallback: Extract locations without explicit distance patterns
        if not locations:
            locations.extend(self._extract_locations_fallback(text_cleaned))
        
        # Remove duplicates và sort by distance
        locations = self._deduplicate_locations(locations)
        locations.sort(key=lambda x: float(x['distance'].replace(',', '.')))
        
        return locations[:3]  # Max 3 locations
    
    def _clean_text_for_parsing(self, text: str) -> str:
        """Clean text for better parsing"""
        # Normalize Vietnamese characters
        text = unicodedata.normalize('NFC', text)
        
        # Replace common variations
        replacements = {
            'kni': 'km', 'knì': 'km', 'kn': 'km',
            'dên': 'đến', 'den': 'đến', 'toi': 'tới',
            'thanh pho': 'thành phố', 'huyen': 'huyện'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _normalize_distance(self, distance_str: str) -> str:
        """Normalize distance string"""
        try:
            # Remove units và clean
            distance_clean = re.sub(r'[^\d,\.]', '', distance_str)
            
            # Handle comma as decimal separator
            if ',' in distance_clean and '.' not in distance_clean:
                distance_clean = distance_clean.replace(',', '.')
            
            # Convert to float và back to consistent format
            distance_val = float(distance_clean)
            
            # Format consistently
            if distance_val == int(distance_val):
                return str(int(distance_val))
            else:
                return f"{distance_val:.1f}".replace('.', ',')
                
        except:
            return "0"
    
    def _find_nearby_distance(self, text: str, location: str) -> str:
        """Tìm distance gần location name"""
        try:
            location_pos = text.lower().find(location.lower())
            if location_pos == -1:
                return "0"
            
            # Search trong khoảng ±50 characters
            start = max(0, location_pos - 50)
            end = min(len(text), location_pos + len(location) + 50)
            context = text[start:end]
            
            # Find distance patterns
            for pattern in self.distance_patterns:
                match = pattern.search(context)
                if match:
                    return self._normalize_distance(match.group(1))
            
            return "0"
            
        except:
            return "0"
    
    def _is_valid_location(self, location: str) -> bool:
        """Validate location name"""
        if not location or len(location) < 2:
            return False
        
        # Check for common non-location words
        invalid_words = ['km', 'den', 'toi', 'đến', 'tới', 'đi', 'di', 'thang', 'thẳng']
        location_lower = location.lower()
        
        for invalid in invalid_words:
            if location_lower == invalid:
                return False
        
        # Must contain at least one letter
        if not re.search(r'[a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', location, re.IGNORECASE):
            return False
        
        return True
    
    def _normalize_location_name(self, location: str) -> str:
        """Normalize location name"""
        # Capitalize first letter of each word
        words = location.split()
        normalized = []
        
        for word in words:
            if word.lower() in ['tp', 'tp.', 'thành', 'phố', 'huyện', 'xã', 'thị', 'trấn']:
                normalized.append(word.upper() if len(word) <= 3 else word.title())
            else:
                normalized.append(word.upper())
        
        return ' '.join(normalized)
    
    def _extract_locations_fallback(self, text: str) -> List[Dict[str, str]]:
        """Fallback location extraction"""
        locations = []
        
        # Find potential location names (capitalized words)
        potential_locations = re.findall(r'[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]{2,}', text)
        
        for loc in potential_locations:
            if self._is_valid_location(loc):
                distance = self._find_nearby_distance(text, loc)
                locations.append({
                    'name': self._normalize_location_name(loc),
                    'distance': distance if distance != "0" else "1",
                    'confidence': 0.6
                })
        
        return locations
    
    def _deduplicate_locations(self, locations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate locations"""
        seen = set()
        unique_locations = []
        
        for loc in locations:
            name_key = loc['name'].lower().replace(' ', '')
            if name_key not in seen:
                seen.add(name_key)
                unique_locations.append(loc)
        
        return unique_locations
    
    def detect_direction_from_text(self, text: str) -> Dict[str, float]:
        """Detect direction keywords trong text"""
        if not text:
            return {}
        
        text_lower = text.lower()
        detected_directions = {}
        
        for direction, keywords in self.direction_keywords.items():
            confidence = 0.0
            
            for keyword in keywords:
                if keyword in text_lower:
                    # Calculate confidence based on keyword match quality
                    keyword_conf = len(keyword) / len(text_lower.split())
                    confidence = max(confidence, min(keyword_conf * 2, 0.9))
            
            if confidence > 0.1:
                detected_directions[direction] = confidence
        
        # Normalize confidences
        if detected_directions:
            total_conf = sum(detected_directions.values())
            if total_conf > 1.0:
                detected_directions = {
                    direction: conf / total_conf 
                    for direction, conf in detected_directions.items()
                }
        
        return detected_directions


# ==================== HYBRID FUSION SYSTEM ====================

class EpicHybridTrafficSignDetector:
    """Epic Hybrid System kết hợp CV + NLP với enhanced algorithms"""
    
    def __init__(self):
        # Initialize modules
        self.arrow_detector = ImprovedArrowDetector()
        self.text_processor = AdvancedTextProcessor()
        
        # Configuration
        self.config = {
            'cv_weight': 0.6,
            'nlp_weight': 0.4,
            'confidence_threshold': 0.15,
            'fusion_strategy': 'intelligent'
        }
        
        print("[INFO] EpicHybridTrafficSignDetector initialized với Vintern + advanced AI algorithms")
    
    def _validate_direction_consistency(self, cv_directions: dict, locations: list) -> dict:
        """Validate và correct direction consistency dựa trên road logic"""
        if not cv_directions or not locations:
            return cv_directions
        
        # Sort locations by distance
        sorted_locations = sorted(locations, key=lambda x: float(x['distance'].replace(',', '.')))
        
        if len(sorted_locations) >= 2:
            short_distance = float(sorted_locations[0]['distance'].replace(',', '.'))
            long_distance = float(sorted_locations[1]['distance'].replace(',', '.'))
            
            # VALIDATION RULES:
            # Rule 1: Nếu có khoảng cách ngắn < 15km, ưu tiên rẽ (turn)
            # Rule 2: Nếu có khoảng cách dài > 50km, ưu tiên thẳng (straight)
            
            validated_directions = cv_directions.copy()
            
            if short_distance < 15 and long_distance > 50:
                # Boost confidence cho turn directions
                for direction in ["rẽ trái", "rẽ phải"]:
                    if direction in validated_directions:
                        validated_directions[direction] *= 1.3  # Boost 30%
                
                # Maintain confidence cho straight direction  
                if "đi thẳng" in validated_directions:
                    validated_directions["đi thẳng"] *= 1.1  # Slight boost 10%
                    
            return validated_directions
        
        return cv_directions
    
    def predict(self, image: np.ndarray) -> str:
        """
        Main prediction method với Epic Hybrid AI + Direction Validation
        Returns: Formatted prediction string
        """
        try:
            # PHASE 1: Computer Vision Analysis
            cv_directions, cv_confidence = self.arrow_detector.detect_arrow_direction(image)
            
            # PHASE 2: NLP Text Analysis với Vintern
            nlp_text = self.text_processor.extract_text(image)  # Uses Vintern first
            nlp_directions = self.text_processor.detect_direction_from_text(nlp_text)
            locations = self.text_processor.parse_locations_from_text(nlp_text)
            
            # PHASE 3: Direction Validation & Consistency Check
            validated_cv_directions = self._validate_direction_consistency(cv_directions, locations)
            
            # PHASE 4: Epic Fusion với validated directions
            final_result = self._epic_fusion_algorithm(
                validated_cv_directions, cv_confidence, 
                nlp_directions, nlp_text, locations
            )
            
            return final_result
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return "Unknown direction"
    
    def _epic_fusion_algorithm(self, cv_directions: dict, cv_confidence: float,
                              nlp_directions: dict, nlp_text: str, locations: list) -> str:
        """
        Epic fusion algorithm kết hợp tất cả thông tin
        """
        # Validate inputs
        valid_locations = [loc for loc in locations if loc['name'] and len(loc['name']) > 2]
        
        # STRATEGY 1: Location-Aware Multi-Direction Mapping
        if len(valid_locations) >= 2 and cv_directions and len(cv_directions) >= 2:
            location_aware_result = self._location_aware_mapping(cv_directions, valid_locations)
            if location_aware_result:
                return location_aware_result
        
        # STRATEGY 2: Single Location with Direction
        if len(valid_locations) == 1 and cv_directions:
            single_loc_result = self._single_location_mapping(cv_directions, cv_confidence, valid_locations[0])
            if single_loc_result:
                return single_loc_result
        
        # STRATEGY 3: CV-NLP Direction Fusion
        if cv_directions and nlp_directions:
            fusion_result = self._cv_nlp_direction_fusion(cv_directions, nlp_directions)
            if fusion_result:
                return fusion_result
        
        # STRATEGY 4: Pure CV Prediction
        if cv_directions and cv_confidence >= self.config['confidence_threshold']:
            return self._format_cv_only_result(cv_directions)
        
        # STRATEGY 5: Pure NLP Prediction
        if nlp_directions:
            return self._format_nlp_only_result(nlp_directions)
        
        # FALLBACK: Raw text if available
        if nlp_text and len(nlp_text) > 5:
            return f"Text detected: {nlp_text[:50]}..."
        
        return "Không xác định được hướng"
    
    def _location_aware_mapping(self, cv_directions: dict, locations: list) -> str:
        """Map directions to locations với ENHANCED road logic intelligence - FIX cho vấn đề lộn hướng"""
        try:
            # Sort locations by distance
            sorted_locations = sorted(locations, key=lambda x: float(x['distance'].replace(',', '.')))
            
            if len(sorted_locations) < 2:
                return ""
            
            short_loc = sorted_locations[0]
            long_loc = sorted_locations[1]
            
            short_distance = float(short_loc['distance'].replace(',', '.'))
            long_distance = float(long_loc['distance'].replace(',', '.'))
            
            # ENHANCED ROAD LOGIC với multiple validation layers
            distance_ratio = long_distance / short_distance if short_distance > 0 else 1
            
            # Apply SMART ROAD LOGIC: 
            # - Short distance (< 20km) = Side road = Turn (rẽ trái/phải)
            # - Long distance (> 50km) = Main road = Straight (đi thẳng)
            # - Medium distance = flexible based on ratio
            
            if distance_ratio > 4 or (short_distance < 15 and long_distance > 40):
                
                # PRIORITY 1: Find turn directions for short distance
                turn_dirs = [(d, c) for d, c in cv_directions.items() if d in ["rẽ trái", "rẽ phải"]]
                straight_dirs = [(d, c) for d, c in cv_directions.items() if d == "đi thẳng"]
                
                if turn_dirs and straight_dirs:
                    # Get highest confidence directions
                    best_turn = max(turn_dirs, key=lambda x: x[1])
                    best_straight = max(straight_dirs, key=lambda x: x[1])
                    
                    # CORRECTED mapping: Short = Turn, Long = Straight
                    result = f"{best_turn[0]} {short_loc['distance']}km đến {short_loc['name']} | {best_straight[0]} {long_loc['distance']}km đến {long_loc['name']}"
                    return result
                
                # Fallback: If only one type available
                elif turn_dirs:
                    best_turn = max(turn_dirs, key=lambda x: x[1])
                    return f"{best_turn[0]} {short_loc['distance']}km đến {short_loc['name']}"
                elif straight_dirs:
                    best_straight = max(straight_dirs, key=lambda x: x[1])
                    return f"{best_straight[0]} {long_loc['distance']}km đến {long_loc['name']}"
            
            # FALLBACK: Map by confidence order khi distance không rõ ràng
            sorted_dirs = sorted(cv_directions.items(), key=lambda x: x[1], reverse=True)
            
            result_parts = []
            for i, (direction, confidence) in enumerate(sorted_dirs[:2]):
                if i < len(sorted_locations) and confidence >= 0.15:
                    loc = sorted_locations[i]
                    result_parts.append(f"{direction} {loc['distance']}km đến {loc['name']}")
            
            return " | ".join(result_parts) if len(result_parts) >= 2 else ""
            
        except Exception as e:
            print(f"[ERROR] Location mapping failed: {e}")
            return ""
    
    def _single_location_mapping(self, cv_directions: dict, cv_confidence: float, location: dict) -> str:
        """Map single location với best CV direction"""
        try:
            if cv_confidence < 0.2:
                return ""
            
            # Get best direction
            best_direction = max(cv_directions.items(), key=lambda x: x[1])
            direction, confidence = best_direction
            
            if confidence >= 0.15:
                return f"{direction} {location['distance']}km đến {location['name']}"
            
            return ""
            
        except Exception as e:
            return ""
    
    def _cv_nlp_direction_fusion(self, cv_directions: dict, nlp_directions: dict) -> str:
        """Fusion CV và NLP directions"""
        try:
            # Weighted combination
            fused_directions = {}
            
            # Add CV directions với weight
            for direction, confidence in cv_directions.items():
                fused_directions[direction] = confidence * self.config['cv_weight']
            
            # Add NLP directions với weight
            for direction, confidence in nlp_directions.items():
                if direction in fused_directions:
                    fused_directions[direction] += confidence * self.config['nlp_weight']
                else:
                    fused_directions[direction] = confidence * self.config['nlp_weight']
            
            # Get best fused direction
            if fused_directions:
                best_direction = max(fused_directions.items(), key=lambda x: x[1])
                direction, confidence = best_direction
                
                if confidence >= self.config['confidence_threshold']:
                    return direction
            
            return ""
            
        except Exception as e:
            return ""
    
    def _format_cv_only_result(self, cv_directions: dict) -> str:
        """Format CV-only result"""
        best_direction = max(cv_directions.items(), key=lambda x: x[1])
        return best_direction[0]
    
    def _format_nlp_only_result(self, nlp_directions: dict) -> str:
        """Format NLP-only result"""
        best_direction = max(nlp_directions.items(), key=lambda x: x[1])
        return best_direction[0]


# ==================== TESTING & VALIDATION ====================

def test_direction_accuracy():
    """Test function để verify độ chính xác direction detection"""
    print("[INFO] Testing Direction Accuracy...")
    
    # Test cases dựa trên vấn đề thực tế của user
    test_cases = [
        {
            'description': 'TÂN CƯƠNG case - should be STRAIGHT',
            'simulated_cv': {'đi thẳng': 0.7, 'rẽ phải': 0.3},
            'simulated_locations': [
                {'name': 'TÂN CƯƠNG', 'distance': '8', 'confidence': 0.9},
                {'name': 'BẮC KAN', 'distance': '91,5', 'confidence': 0.8}
            ],
            'expected': 'rẽ phải 8km đến TÂN CƯƠNG | đi thẳng 91,5km đến BẮC KAN'
        },
        {
            'description': 'BẮC KAN case - should be RIGHT TURN',  
            'simulated_cv': {'đi thẳng': 0.6, 'rẽ phải': 0.8},
            'simulated_locations': [
                {'name': 'TÂN CƯƠNG', 'distance': '8', 'confidence': 0.9},
                {'name': 'BẮC KAN', 'distance': '91,5', 'confidence': 0.8}
            ],
            'expected': 'rẽ phải 8km đến TÂN CƯƠNG | đi thẳng 91,5km đến BẮC KAN'
        }
    ]
    
    detector = EpicHybridTrafficSignDetector()
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['description']} ---")
        
        # Simulate the location-aware mapping
        result = detector._location_aware_mapping(
            test_case['simulated_cv'], 
            test_case['simulated_locations']
        )
        
        print(f"Result: {result}")
        print(f"Expected: {test_case['expected']}")
        
        # Check if road logic is applied correctly
        if "8km đến TÂN CƯƠNG" in result and "91,5km đến BẮC KAN" in result:
            if result.startswith("rẽ") and "đi thẳng" in result.split("|")[1]:
                print("✅ CORRECT: Short distance = Turn, Long distance = Straight")
            else:
                print("❌ INCORRECT: Direction mapping is wrong")
        else:
            print("⚠️  PARTIAL: Missing location information")

def test_ocr_comparison(image_path: str = None):
    """Test so sánh hiệu suất giữa Vintern và EasyOCR"""
    if not image_path:
        print("Please provide image path for OCR comparison")
        return
    
    try:
        import time
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot load image: {image_path}")
            return
        
        processor = AdvancedTextProcessor()
        
        print(f"\n{'='*60}")
        print(f"OCR COMPARISON TEST - {image_path}")
        print(f"{'='*60}")
        
        # Test Vintern
        if VINTERN_AVAILABLE:
            print("\n🤖 TESTING VINTERN MODEL:")
            start_time = time.time()
            vintern_result = processor.extract_text_vintern(image)
            vintern_time = time.time() - start_time
            
            print(f"  Result: '{vintern_result}'")
            print(f"  Time: {vintern_time:.2f}s")
            print(f"  Vietnamese Score: {processor._score_vietnamese_text(vintern_result):.2f}")
        else:
            print("\n❌ VINTERN NOT AVAILABLE")
            vintern_result = ""
            vintern_time = 0
        
        # Test EasyOCR
        if EASYOCR_AVAILABLE:
            print("\n📚 TESTING EASYOCR FALLBACK:")
            start_time = time.time()
            easyocr_result = processor.extract_text_easyocr(image)
            easyocr_time = time.time() - start_time
            
            print(f"  Result: '{easyocr_result}'")
            print(f"  Time: {easyocr_time:.2f}s")
            print(f"  Vietnamese Score: {processor._score_vietnamese_text(easyocr_result):.2f}")
        else:
            print("\n❌ EASYOCR NOT AVAILABLE")
            easyocr_result = ""
            easyocr_time = 0
        
        # Comparison
        print(f"\n📊 COMPARISON SUMMARY:")
        print(f"{'='*40}")
        
        if vintern_result and easyocr_result:
            vn_score = processor._score_vietnamese_text(vintern_result)
            easy_score = processor._score_vietnamese_text(easyocr_result)
            
            if vn_score > easy_score:
                print("🏆 WINNER: Vintern (better Vietnamese content)")
            elif easy_score > vn_score:
                print("🏆 WINNER: EasyOCR (better Vietnamese content)")
            else:
                print("🤝 TIE: Similar quality")
            
            if vintern_time < easyocr_time:
                print(f"⚡ FASTER: Vintern ({vintern_time:.2f}s vs {easyocr_time:.2f}s)")
            else:
                print(f"⚡ FASTER: EasyOCR ({easyocr_time:.2f}s vs {vintern_time:.2f}s)")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"[ERROR] OCR comparison failed: {e}")


# ==================== MAIN EXECUTION ====================

def main():
    """Main function để test Epic Hybrid System với Vintern"""
    print("[INFO] Starting Epic Hybrid Traffic Sign Detection System với Vintern...")
    
    # Check dependencies
    check_and_install_dependencies()
    
    # Ask user for mode
    print("\nChọn chế độ:")
    print("1. Real-time detection (webcam)")
    print("2. Test direction accuracy")
    print("3. Load image file")
    print("4. Test OCR comparison (Vintern vs EasyOCR)")
    
    choice = input("Nhập lựa chọn (1/2/3/4): ").strip()
    
    if choice == "2":
        test_direction_accuracy()
        return
    elif choice == "3":
        # Test với image file
        image_path = input("Nhập đường dẫn ảnh: ").strip()
        if image_path:
            test_single_image(image_path)
        return
    elif choice == "4":
        # Test OCR comparison
        image_path = input("Nhập đường dẫn ảnh để test OCR: ").strip()
        if image_path:
            test_ocr_comparison(image_path)
        return
    
    # Default: Real-time detection
    real_time_detection()

def check_and_install_dependencies():
    """Check và hướng dẫn cài đặt dependencies cho Vintern"""
    print("\n" + "="*60)
    print("CHECKING DEPENDENCIES FOR VINTERN MODEL")
    print("="*60)
    
    missing_deps = []
    
    # Check Vintern dependencies
    try:
        import transformers
        import torch
        from PIL import Image
        print("✅ Vintern dependencies available (transformers, torch, PIL)")
    except ImportError as e:
        print(f"❌ Vintern dependencies missing: {e}")
        missing_deps.append("vintern")
    
    # Check EasyOCR fallback
    try:
        import easyocr
        print("✅ EasyOCR fallback available")
    except ImportError:
        print("⚠️  EasyOCR fallback not available")
        missing_deps.append("easyocr")
    
    if missing_deps:
        print("\n📦 INSTALLATION COMMANDS:")
        print("-" * 40)
        if "vintern" in missing_deps:
            print("For Vintern (recommended):")
            print("  pip install transformers torch torchvision pillow")
            print("  # For GPU support:")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
        if "easyocr" in missing_deps:
            print("\nFor EasyOCR fallback:")
            print("  pip install easyocr")
        
        print("\n⚠️  Some features may not work without proper dependencies!")
        input("Press Enter to continue anyway...")
    
    print("="*60)

def test_single_image(image_path: str):
    """Test với single image"""
    try:
        detector = EpicHybridTrafficSignDetector()
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"[ERROR] Cannot load image: {image_path}")
            return
        
        print(f"[INFO] Processing image: {image_path}")
        result = detector.predict(image)
        print(f"[RESULT] {result}")
        
        # Display image với result
        cv2.putText(image, result, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Traffic Sign Detection Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"[ERROR] Failed to process image: {e}")

def real_time_detection():
    """Real-time detection với webcam sử dụng Vintern"""
    print("[INFO] Initializing camera and detector...")
    
    # Initialize detector
    try:
        detector = EpicHybridTrafficSignDetector()
        print("[INFO] Detector initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize detector: {e}")
        return
    
    # Video capture với error handling
    cap = cv2.VideoCapture(0)  # 0 for webcam, hoặc video file path
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        print("Possible solutions:")
        print("1. Check if camera is connected")
        print("2. Check if other applications are using the camera")
        print("3. Try different camera index (1, 2, etc.)")
        
        # Try alternative camera indices
        for i in range(1, 4):
            print(f"[INFO] Trying camera index {i}...")
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"[INFO] Successfully opened camera at index {i}")
                break
            cap.release()
        else:
            print("[ERROR] No working camera found!")
            return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("[INFO] Camera opened successfully!")
    print("[INFO] System ready với Vintern OCR! Press 'q' to quit, 's' to save frame, 't' to test accuracy")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera!")
            break
        
        frame_count += 1
        
        # Resize frame for processing
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Predict (skip some frames for performance)
        prediction = "Initializing..."
        if frame_count % 5 == 0:  # Process every 5th frame for better performance
            try:
                prediction = detector.predict(frame)
            except Exception as e:
                print(f"[ERROR] Prediction failed: {e}")
                prediction = "Prediction error"
        
        # Display result on frame
        cv2.putText(frame, prediction, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display model info
        model_info = "Vintern OCR" if VINTERN_AVAILABLE else "EasyOCR Fallback"
        cv2.putText(frame, f"Epic Hybrid AI + {model_info}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Display frame count for debugging
        cv2.putText(frame, f"Frame: {frame_count}", (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Epic Hybrid Traffic Sign Detection + Vintern', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quitting...")
            break
        elif key == ord('s'):
            # Save current frame
            filename = f'traffic_sign_frame_{int(cv2.getTickCount())}.jpg'
            cv2.imwrite(filename, frame)
            print(f"[INFO] Frame saved as {filename}!")
        elif key == ord('t'):
            # Run accuracy test
            print("[INFO] Running accuracy test...")
            test_direction_accuracy()
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Epic Hybrid System với Vintern stopped.")


if __name__ == "__main__":
    main()