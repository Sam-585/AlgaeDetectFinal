"""
Spectral Indices Calculator
Calculates various spectral indices for algae and water quality assessment
"""

import numpy as np
from typing import Dict, Union, Any
import math

class SpectralIndicesCalculator:
    """Calculator for various spectral indices used in water quality assessment"""
    
    def __init__(self):
        """Initialize the calculator"""
        pass
    
    def calculate_ndvi(self, imagery_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate Normalized Difference Vegetation Index (NDVI)
        NDVI = (NIR - Red) / (NIR + Red)
        
        Args:
            imagery_data: Dictionary containing band data and metadata
            
        Returns:
            Dict containing NDVI statistics
        """
        
        bands = imagery_data['bands']
        satellite = imagery_data['metadata']['satellite']
        
        # Select appropriate bands based on satellite
        if satellite == "Sentinel-2":
            red = bands.get('B4', 0)
            nir = bands.get('B8', 0)
        elif satellite in ["Landsat 8/9"]:
            red = bands.get('SR_B4', 0)
            nir = bands.get('SR_B5', 0)
        else:  # MODIS
            red = bands.get('sur_refl_b01', 0)
            nir = bands.get('sur_refl_b02', 0)
        
        # Calculate NDVI
        if (nir + red) != 0:
            ndvi = (nir - red) / (nir + red)
        else:
            ndvi = 0
        
        # Clamp to valid range [-1, 1]
        ndvi = max(-1, min(1, ndvi))
        
        return {
            'mean': ndvi,
            'interpretation': self._interpret_ndvi(ndvi),
            'red_band': red,
            'nir_band': nir
        }
    
    def calculate_ndwi(self, imagery_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate Normalized Difference Water Index (NDWI)
        NDWI = (Green - NIR) / (Green + NIR)
        
        Args:
            imagery_data: Dictionary containing band data and metadata
            
        Returns:
            Dict containing NDWI statistics
        """
        
        bands = imagery_data['bands']
        satellite = imagery_data['metadata']['satellite']
        
        # Select appropriate bands
        if satellite == "Sentinel-2":
            green = bands.get('B3', 0)
            nir = bands.get('B8', 0)
        elif satellite in ["Landsat 8/9"]:
            green = bands.get('SR_B3', 0)
            nir = bands.get('SR_B5', 0)
        else:  # MODIS
            green = bands.get('sur_refl_b04', 0)
            nir = bands.get('sur_refl_b02', 0)
        
        # Calculate NDWI
        if (green + nir) != 0:
            ndwi = (green - nir) / (green + nir)
        else:
            ndwi = 0
        
        # Clamp to valid range
        ndwi = max(-1, min(1, ndwi))
        
        return {
            'mean': ndwi,
            'interpretation': self._interpret_ndwi(ndwi),
            'green_band': green,
            'nir_band': nir
        }
    
    def calculate_chlorophyll_a(self, imagery_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate Chlorophyll-a concentration using empirical algorithms
        Uses band ratio algorithms specific to each satellite
        
        Args:
            imagery_data: Dictionary containing band data and metadata
            
        Returns:
            Dict containing Chlorophyll-a estimates
        """
        
        bands = imagery_data['bands']
        satellite = imagery_data['metadata']['satellite']
        
        if satellite == "Sentinel-2":
            # Use B4/B5 ratio algorithm for Sentinel-2
            red = bands.get('B4', 0)
            red_edge = bands.get('B5', bands.get('B8', 0))  # Fallback to NIR if B5 not available
            
            if red_edge != 0:
                ratio = red / red_edge
                # Empirical algorithm: Chl-a = a * (Red/RedEdge)^b
                chl_a = 23.1 * (ratio ** 2.4)  # Simplified algorithm
            else:
                chl_a = 0
                
        elif satellite in ["Landsat 8/9"]:
            # Use Blue/Green ratio for Landsat
            blue = bands.get('SR_B2', 0)
            green = bands.get('SR_B3', 0)
            
            if green != 0:
                ratio = blue / green
                chl_a = 18.5 * (ratio ** 1.8)
            else:
                chl_a = 0
                
        else:  # MODIS
            # Use established MODIS Chl-a algorithm
            blue = bands.get('sur_refl_b03', 0)
            green = bands.get('sur_refl_b04', 0)
            
            if green != 0 and blue != 0:
                ratio = blue / green
                chl_a = 0.2424 * (ratio ** -1.981)  # MODIS OC3 algorithm
            else:
                chl_a = 0
        
        # Ensure reasonable range for inland waters (0-200 μg/L)
        chl_a = max(0, min(200, chl_a))
        
        return {
            'mean': chl_a,
            'unit': 'μg/L',
            'interpretation': self._interpret_chlorophyll(chl_a),
            'algorithm': f'{satellite}_ratio_method'
        }
    
    def calculate_turbidity(self, imagery_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate water turbidity using reflectance data
        
        Args:
            imagery_data: Dictionary containing band data and metadata
            
        Returns:
            Dict containing turbidity estimates
        """
        
        bands = imagery_data['bands']
        satellite = imagery_data['metadata']['satellite']
        
        if satellite == "Sentinel-2":
            # Use red band reflectance for turbidity
            red = bands.get('B4', 0)
            # Convert to reflectance (0-1) if needed
            if red > 1:
                red = red / 10000  # Scale factor for Sentinel-2
            
            # Empirical relationship: Turbidity = a * Red^b
            turbidity = 31.6 * (red ** 0.64)
            
        elif satellite in ["Landsat 8/9"]:
            # Use red band for Landsat
            red = bands.get('SR_B4', 0)
            if red > 1:
                red = red / 10000  # Scale factor
            
            turbidity = 28.4 * (red ** 0.72)
            
        else:  # MODIS
            # Use red band reflectance
            red = bands.get('sur_refl_b01', 0)
            turbidity = 25.2 * (red ** 0.68)
        
        # Ensure reasonable range (0-100 NTU)
        turbidity = max(0, min(100, turbidity))
        
        return {
            'mean': turbidity,
            'unit': 'NTU',
            'interpretation': self._interpret_turbidity(turbidity),
            'red_reflectance': red
        }
    
    def calculate_fai(self, imagery_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate Floating Algae Index (FAI)
        FAI = NIR - (Red + (SWIR - Red) * (λNIR - λRed) / (λSWIR - λRed))
        
        Args:
            imagery_data: Dictionary containing band data and metadata
            
        Returns:
            Dict containing FAI values
        """
        
        bands = imagery_data['bands']
        satellite = imagery_data['metadata']['satellite']
        
        if satellite == "Sentinel-2":
            red = bands.get('B4', 0)
            nir = bands.get('B8', 0)
            swir = bands.get('B11', 0)
            # Wavelengths in nm
            lambda_red = 665
            lambda_nir = 842
            lambda_swir = 1610
            
        elif satellite in ["Landsat 8/9"]:
            red = bands.get('SR_B4', 0)
            nir = bands.get('SR_B5', 0)
            swir = bands.get('SR_B6', 0)
            lambda_red = 655
            lambda_nir = 865
            lambda_swir = 1610
            
        else:  # MODIS doesn't have appropriate bands for FAI
            return {
                'mean': 0,
                'interpretation': 'Not available for MODIS',
                'note': 'MODIS lacks required SWIR bands for FAI calculation'
            }
        
        # Calculate FAI using the baseline subtraction method
        if (lambda_swir - lambda_red) != 0:
            baseline = red + (swir - red) * (lambda_nir - lambda_red) / (lambda_swir - lambda_red)
            fai = nir - baseline
        else:
            fai = 0
        
        # Normalize by scaling factor
        if satellite == "Sentinel-2":
            fai = fai / 10000
        elif satellite in ["Landsat 8/9"]:
            fai = fai / 10000
        
        return {
            'mean': fai,
            'interpretation': self._interpret_fai(fai),
            'red_band': red,
            'nir_band': nir,
            'swir_band': swir
        }
    
    def calculate_ndvi_from_rgb(self, image_array: np.ndarray) -> float:
        """
        Calculate NDVI from RGB image (approximation)
        Uses green band as proxy for NIR in absence of actual NIR data
        
        Args:
            image_array: RGB image as numpy array
            
        Returns:
            Approximate NDVI value
        """
        
        if len(image_array.shape) == 3:
            red = np.mean(image_array[:, :, 0])
            green = np.mean(image_array[:, :, 1])
            
            # Use green as NIR proxy (rough approximation)
            if (green + red) != 0:
                ndvi_approx = (green - red) / (green + red)
            else:
                ndvi_approx = 0
                
            return max(-1, min(1, ndvi_approx))
        
        return 0
    
    def calculate_chlorophyll_from_rgb(self, image_array: np.ndarray) -> float:
        """
        Estimate chlorophyll from RGB image
        
        Args:
            image_array: RGB image as numpy array
            
        Returns:
            Estimated chlorophyll-a concentration
        """
        
        if len(image_array.shape) == 3:
            red = np.mean(image_array[:, :, 0])
            green = np.mean(image_array[:, :, 1])
            blue = np.mean(image_array[:, :, 2])
            
            # Calculate green/red ratio as indicator
            if red != 0:
                green_red_ratio = green / red
                # Empirical relationship for RGB images
                chl_a = 15.0 * green_red_ratio ** 1.2
            else:
                chl_a = 0
                
            return max(0, min(100, chl_a))
        
        return 0
    
    def calculate_turbidity_from_rgb(self, image_array: np.ndarray) -> float:
        """
        Estimate turbidity from RGB image
        
        Args:
            image_array: RGB image as numpy array
            
        Returns:
            Estimated turbidity in NTU
        """
        
        if len(image_array.shape) == 3:
            # Use overall brightness as turbidity indicator
            brightness = np.mean(image_array)
            
            # Normalize to 0-1 range
            brightness_norm = brightness / 255.0
            
            # Empirical relationship
            turbidity = 50.0 * brightness_norm
            
            return max(0, min(100, turbidity))
        
        return 0
    
    def _interpret_ndvi(self, ndvi: float) -> str:
        """Interpret NDVI values"""
        if ndvi > 0.6:
            return "Dense vegetation/algae"
        elif ndvi > 0.3:
            return "Moderate vegetation/algae"
        elif ndvi > 0.1:
            return "Sparse vegetation"
        elif ndvi > -0.1:
            return "Non-vegetated/water"
        else:
            return "Water/bare soil"
    
    def _interpret_ndwi(self, ndwi: float) -> str:
        """Interpret NDWI values"""
        if ndwi > 0.3:
            return "Open water"
        elif ndwi > 0:
            return "Water/wet soil"
        elif ndwi > -0.3:
            return "Dry soil/vegetation"
        else:
            return "Built-up/bare soil"
    
    def _interpret_chlorophyll(self, chl_a: float) -> str:
        """Interpret Chlorophyll-a concentrations"""
        if chl_a > 30:
            return "Very high - severe algal bloom"
        elif chl_a > 15:
            return "High - algal bloom present"
        elif chl_a > 8:
            return "Moderate - elevated algae"
        elif chl_a > 3:
            return "Low-moderate - normal levels"
        else:
            return "Low - oligotrophic conditions"
    
    def _interpret_turbidity(self, turbidity: float) -> str:
        """Interpret turbidity values"""
        if turbidity > 40:
            return "Very high - heavily polluted"
        elif turbidity > 20:
            return "High - polluted"
        elif turbidity > 10:
            return "Moderate - slightly turbid"
        elif turbidity > 4:
            return "Low - clear water"
        else:
            return "Very low - very clear"
    
    def _interpret_fai(self, fai: float) -> str:
        """Interpret FAI values"""
        if fai > 0.015:
            return "Dense floating algae"
        elif fai > 0.005:
            return "Moderate floating algae"
        elif fai > 0:
            return "Light algae presence"
        else:
            return "No significant floating algae"
    
    def calculate_comprehensive_index(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive water quality index combining multiple indices
        
        Args:
            imagery_data: Dictionary containing band data and metadata
            
        Returns:
            Dict containing comprehensive analysis
        """
        
        # Calculate all indices
        ndvi = self.calculate_ndvi(imagery_data)
        ndwi = self.calculate_ndwi(imagery_data)
        chl_a = self.calculate_chlorophyll_a(imagery_data)
        turbidity = self.calculate_turbidity(imagery_data)
        fai = self.calculate_fai(imagery_data)
        
        # Calculate composite water quality score (0-100)
        # Higher chlorophyll-a and turbidity = lower score
        # Higher FAI = lower score
        # NDWI closer to water values = potential for better baseline
        
        chl_score = max(0, 100 - (chl_a['mean'] * 2))  # Chlorophyll penalty
        turbidity_score = max(0, 100 - turbidity['mean'])  # Turbidity penalty
        fai_score = max(0, 100 - (fai['mean'] * 1000))  # FAI penalty (scaled)
        
        # Combine scores with weights
        composite_score = (chl_score * 0.4 + turbidity_score * 0.3 + fai_score * 0.3)
        
        # Determine overall water quality category
        if composite_score > 80:
            quality_category = "Excellent"
        elif composite_score > 60:
            quality_category = "Good"
        elif composite_score > 40:
            quality_category = "Fair"
        elif composite_score > 20:
            quality_category = "Poor"
        else:
            quality_category = "Very Poor"
        
        return {
            'ndvi': ndvi,
            'ndwi': ndwi,
            'chlorophyll_a': chl_a,
            'turbidity': turbidity,
            'fai': fai,
            'composite_score': composite_score,
            'quality_category': quality_category,
            'recommendations': self._generate_recommendations(composite_score, chl_a['mean'], turbidity['mean'])
        }
    
    def _generate_recommendations(self, score: float, chl_a: float, turbidity: float) -> list:
        """Generate management recommendations based on water quality metrics"""
        
        recommendations = []
        
        if score < 40:
            recommendations.append("Immediate intervention required")
            recommendations.append("Restrict water use for drinking/recreation")
        
        if chl_a > 20:
            recommendations.append("Implement nutrient reduction strategies")
            recommendations.append("Consider algaecide treatment if appropriate")
        
        if turbidity > 25:
            recommendations.append("Investigate sediment sources")
            recommendations.append("Improve erosion control in watershed")
        
        if score > 70:
            recommendations.append("Maintain current management practices")
            recommendations.append("Continue regular monitoring")
        
        return recommendations
