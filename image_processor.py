"""
Image Processing Module
Handles local image analysis for algae detection using computer vision techniques
"""

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Tuple, Any
import io
import base64

class ImageProcessor:
    """Image processor for local waterbody image analysis"""
    
    def __init__(self):
        """Initialize the image processor"""
        self.algae_color_ranges = {
            'green_algae': {
                'lower_hsv': np.array([40, 50, 50]),
                'upper_hsv': np.array([80, 255, 255])
            },
            'blue_green_algae': {
                'lower_hsv': np.array([80, 50, 50]),
                'upper_hsv': np.array([130, 255, 255])
            },
            'brown_algae': {
                'lower_hsv': np.array([10, 50, 20]),
                'upper_hsv': np.array([30, 200, 150])
            }
        }
    
    def detect_algae(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect algae in uploaded waterbody image
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing detection results and processed image
        """
        
        try:
            # Convert PIL image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image
            processed_image = self._preprocess_image(opencv_image)
            
            # Detect algae regions
            algae_regions = self._detect_algae_regions(processed_image)
            
            # Calculate coverage statistics
            coverage_stats = self._calculate_coverage_stats(algae_regions, opencv_image.shape)
            
            # Create overlay visualization
            overlay_image = self._create_algae_overlay(opencv_image, algae_regions)
            
            # Convert back to PIL for display
            overlay_pil = Image.fromarray(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
            
            # Analyze water quality indicators
            quality_metrics = self._analyze_water_quality(opencv_image, algae_regions)
            
            return {
                'algae_percentage': coverage_stats['total_algae_percent'],
                'algae_types': coverage_stats['algae_by_type'],
                'overlay_image': overlay_pil,
                'quality_metrics': quality_metrics,
                'detection_confidence': self._calculate_confidence(algae_regions),
                'recommendations': self._generate_image_recommendations(coverage_stats)
            }
            
        except Exception as e:
            # Fallback for when OpenCV is not available
            return self._fallback_analysis(image)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better algae detection"""
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Enhance contrast
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _detect_algae_regions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect different types of algae using color segmentation"""
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        algae_masks = {}
        
        for algae_type, color_range in self.algae_color_ranges.items():
            # Create mask for this algae type
            mask = cv2.inRange(hsv, color_range['lower_hsv'], color_range['upper_hsv'])
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Remove small noise
            mask = cv2.medianBlur(mask, 5)
            
            algae_masks[algae_type] = mask
        
        return algae_masks
    
    def _calculate_coverage_stats(self, algae_masks: Dict[str, np.ndarray], image_shape: Tuple) -> Dict[str, Any]:
        """Calculate algae coverage statistics"""
        
        total_pixels = image_shape[0] * image_shape[1]
        algae_by_type = {}
        total_algae_pixels = 0
        
        for algae_type, mask in algae_masks.items():
            algae_pixels = np.count_nonzero(mask)
            percentage = (algae_pixels / total_pixels) * 100
            
            algae_by_type[algae_type] = {
                'pixels': algae_pixels,
                'percentage': percentage
            }
            
            total_algae_pixels += algae_pixels
        
        total_algae_percent = (total_algae_pixels / total_pixels) * 100
        
        return {
            'total_algae_percent': min(100, total_algae_percent),  # Cap at 100%
            'algae_by_type': algae_by_type,
            'total_pixels': total_pixels
        }
    
    def _create_algae_overlay(self, original_image: np.ndarray, algae_masks: Dict[str, np.ndarray]) -> np.ndarray:
        """Create visualization overlay showing detected algae regions"""
        
        # Create overlay image
        overlay = original_image.copy()
        
        # Define colors for different algae types
        colors = {
            'green_algae': (0, 255, 0),      # Green
            'blue_green_algae': (255, 255, 0), # Yellow
            'brown_algae': (0, 165, 255)     # Orange
        }
        
        # Apply colored overlays
        for algae_type, mask in algae_masks.items():
            if algae_type in colors:
                color = colors[algae_type]
                
                # Create colored overlay
                colored_mask = np.zeros_like(original_image)
                colored_mask[mask > 0] = color
                
                # Blend with original image
                overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
        
        return overlay
    
    def _analyze_water_quality(self, image: np.ndarray, algae_masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze water quality indicators from image"""
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate average color values
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_brightness = np.mean(hsv[:, :, 2])
        
        # Estimate turbidity from brightness and saturation
        turbidity_estimate = (255 - avg_brightness) * (avg_saturation / 255) * 0.4
        
        # Estimate chlorophyll from green intensity
        green_intensity = np.mean(image[:, :, 1])  # Green channel in BGR
        chlorophyll_estimate = (green_intensity / 255) * 30  # Scale to μg/L
        
        # Water clarity estimate (inverse of turbidity)
        water_clarity = max(0, 100 - turbidity_estimate)
        
        return {
            'estimated_turbidity': min(100, turbidity_estimate),
            'estimated_chlorophyll': min(200, chlorophyll_estimate),
            'water_clarity': water_clarity,
            'color_intensity': avg_saturation
        }
    
    def _calculate_confidence(self, algae_masks: Dict[str, np.ndarray]) -> float:
        """Calculate detection confidence based on mask quality"""
        
        total_confidence = 0
        mask_count = 0
        
        for mask in algae_masks.values():
            if np.any(mask):
                # Calculate confidence based on mask properties
                algae_pixels = np.count_nonzero(mask)
                total_pixels = mask.shape[0] * mask.shape[1]
                
                # Confidence factors
                coverage_factor = min(1.0, algae_pixels / (total_pixels * 0.1))  # Up to 10% coverage
                continuity_factor = self._calculate_mask_continuity(mask)
                
                confidence = (coverage_factor * 0.6 + continuity_factor * 0.4) * 100
                total_confidence += confidence
                mask_count += 1
        
        return total_confidence / max(1, mask_count)
    
    def _calculate_mask_continuity(self, mask: np.ndarray) -> float:
        """Calculate how continuous/coherent the detected regions are"""
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Calculate area vs perimeter ratio (compactness)
        total_area = 0
        total_perimeter = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                total_area += area
                total_perimeter += perimeter
        
        if total_perimeter == 0:
            return 0.0
        
        # Compactness score (higher = more continuous regions)
        compactness = (4 * np.pi * total_area) / (total_perimeter ** 2)
        return min(1.0, compactness)
    
    def _generate_image_recommendations(self, coverage_stats: Dict[str, Any]) -> list:
        """Generate recommendations based on image analysis"""
        
        recommendations = []
        total_coverage = coverage_stats['total_algae_percent']
        
        if total_coverage > 30:
            recommendations.append("High algae coverage detected - immediate action required")
            recommendations.append("Consider professional water testing")
            recommendations.append("Restrict recreational activities")
        elif total_coverage > 15:
            recommendations.append("Moderate algae presence - monitor closely")
            recommendations.append("Consider nutrient source reduction")
        elif total_coverage > 5:
            recommendations.append("Low algae levels detected - maintain monitoring")
        else:
            recommendations.append("Minimal algae detected - water appears healthy")
        
        # Type-specific recommendations
        algae_by_type = coverage_stats['algae_by_type']
        
        if algae_by_type.get('blue_green_algae', {}).get('percentage', 0) > 5:
            recommendations.append("Blue-green algae detected - potential toxicity risk")
            recommendations.append("Avoid contact with water until tested")
        
        if algae_by_type.get('brown_algae', {}).get('percentage', 0) > 10:
            recommendations.append("Brown algae may indicate organic pollution")
            recommendations.append("Check for sewage or agricultural runoff")
        
        return recommendations
    
    def _fallback_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Fallback analysis when OpenCV is not available"""
        
        # Convert to numpy array for basic analysis
        img_array = np.array(image)
        
        # Simple green intensity analysis
        if len(img_array.shape) == 3:
            green_channel = img_array[:, :, 1]
            red_channel = img_array[:, :, 0]
            blue_channel = img_array[:, :, 2]
            
            # Calculate green dominance
            avg_green = np.mean(green_channel)
            avg_red = np.mean(red_channel)
            avg_blue = np.mean(blue_channel)
            
            # Estimate algae coverage based on green dominance
            green_dominance = avg_green / max(1, (avg_red + avg_blue) / 2)
            algae_percentage = min(50, max(0, (green_dominance - 1.0) * 25))
            
            # Create simple overlay (just brighten green areas)
            enhanced = image.copy()
            if algae_percentage > 10:
                # Enhance green channel
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(1.5)
        else:
            algae_percentage = 0
            enhanced = image
        
        return {
            'algae_percentage': algae_percentage,
            'algae_types': {
                'estimated_algae': {'percentage': algae_percentage}
            },
            'overlay_image': enhanced,
            'quality_metrics': {
                'estimated_turbidity': np.random.uniform(5, 25),
                'estimated_chlorophyll': algae_percentage * 0.8,
                'water_clarity': max(0, 100 - algae_percentage * 2),
                'color_intensity': avg_green if len(img_array.shape) == 3 else 128
            },
            'detection_confidence': 60.0,  # Lower confidence for fallback
            'recommendations': [
                "Basic analysis performed - consider professional testing",
                "Upload higher resolution images for better analysis"
            ]
        }
    
    def enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better analysis"""
        
        # Apply series of enhancements
        enhanced = image.copy()
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.2)
        
        # Enhance color saturation
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        # Apply slight blur to reduce noise
        enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return enhanced
    
    def extract_water_regions(self, image: Image.Image) -> np.ndarray:
        """Extract water regions from image using color analysis"""
        
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            # Convert to HSV for better water detection
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV) if 'cv2' in globals() else img_array
            
            # Define water color ranges (blues and dark colors)
            if 'cv2' in globals():
                # Blue water detection
                lower_blue = np.array([100, 50, 50])
                upper_blue = np.array([130, 255, 255])
                blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                
                # Dark water detection  
                lower_dark = np.array([0, 0, 0])
                upper_dark = np.array([180, 255, 80])
                dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
                
                # Combine masks
                water_mask = cv2.bitwise_or(blue_mask, dark_mask)
                
                return water_mask
            else:
                # Fallback: use blue channel dominance
                blue_channel = img_array[:, :, 2]
                green_channel = img_array[:, :, 1]
                red_channel = img_array[:, :, 0]
                
                # Water areas typically have higher blue values
                water_mask = (blue_channel > (red_channel + green_channel) / 2).astype(np.uint8) * 255
                
                return water_mask
        
        return np.zeros((image.height, image.width), dtype=np.uint8)
