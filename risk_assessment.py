"""
Risk Assessment Module
Evaluates algae bloom risks and environmental impacts
"""

import numpy as np
from typing import Dict, Any, List
import math
from datetime import datetime, timedelta

class RiskAssessment:
    """Risk assessment calculator for algae blooms"""
    
    def __init__(self):
        """Initialize risk assessment thresholds and parameters"""
        
        # Risk thresholds for different parameters
        self.chlorophyll_thresholds = {
            'low': 5.0,      # μg/L
            'medium': 15.0,   # μg/L
            'high': 30.0,     # μg/L
            'severe': 50.0    # μg/L
        }
        
        self.turbidity_thresholds = {
            'low': 10.0,      # NTU
            'medium': 25.0,   # NTU  
            'high': 50.0,     # NTU
            'severe': 100.0   # NTU
        }
        
        self.fai_thresholds = {
            'low': 0.001,
            'medium': 0.005,
            'high': 0.015,
            'severe': 0.030
        }
        
        # Environmental impact factors
        self.temperature_factor = 1.0  # Multiplier for temperature effects
        self.nutrient_factor = 1.0     # Multiplier for nutrient loading
        self.seasonal_factor = 1.0     # Seasonal variation factor
    
    def assess_algae_risk(self, spectral_indices: Dict[str, Any], imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive algae bloom risk assessment
        
        Args:
            spectral_indices: Dictionary of calculated spectral indices
            imagery_data: Satellite imagery data and metadata
            
        Returns:
            Dictionary containing comprehensive risk assessment
        """
        
        # Extract key parameters
        chl_a = self._extract_value(spectral_indices.get('Chlorophyll-a', {}))
        turbidity = self._extract_value(spectral_indices.get('Turbidity', {}))
        fai = self._extract_value(spectral_indices.get('FAI (Floating Algae Index)', {}))
        ndvi = self._extract_value(spectral_indices.get('NDVI', {}))
        ndwi = self._extract_value(spectral_indices.get('NDWI', {}))
        
        # Calculate individual risk scores
        chl_risk = self._calculate_chlorophyll_risk(chl_a)
        turbidity_risk = self._calculate_turbidity_risk(turbidity)
        fai_risk = self._calculate_fai_risk(fai)
        
        # Calculate vegetation risk (high NDVI over water can indicate algae)
        vegetation_risk = self._calculate_vegetation_risk(ndvi, ndwi)
        
        # Combine risks with weights
        combined_risk_score = (
            chl_risk * 0.35 +           # Chlorophyll-a is primary indicator
            fai_risk * 0.25 +           # FAI specifically targets floating algae
            turbidity_risk * 0.20 +     # Turbidity indicates water quality
            vegetation_risk * 0.20      # NDVI over water areas
        )
        
        # Apply environmental factors
        adjusted_risk_score = self._apply_environmental_factors(
            combined_risk_score, imagery_data
        )
        
        # Calculate algae coverage percentage
        algae_coverage = self._estimate_algae_coverage(chl_a, fai, ndvi)
        
        # Determine risk level
        risk_level = self._determine_risk_level(adjusted_risk_score)
        
        # Calculate growth rate if temporal data available
        growth_rate = self._estimate_growth_rate(spectral_indices, imagery_data)
        
        # Generate detailed assessment
        return {
            'risk_score': adjusted_risk_score,
            'risk_level': risk_level,
            'algae_coverage_percent': algae_coverage,
            'growth_rate': growth_rate,
            'individual_risks': {
                'chlorophyll': chl_risk,
                'turbidity': turbidity_risk,
                'floating_algae': fai_risk,
                'vegetation': vegetation_risk
            },
            'risk_factors': self._identify_risk_factors(chl_a, turbidity, fai, ndvi),
            'confidence_level': self._calculate_confidence(spectral_indices),
            'temporal_trend': self._analyze_temporal_trend(growth_rate),
            'environmental_conditions': self._assess_environmental_conditions(imagery_data)
        }
    
    def assess_algae_risk_from_image(self, image_results: Dict[str, Any], indices: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess risk from uploaded image analysis
        
        Args:
            image_results: Results from image processing
            indices: Calculated indices from image
            
        Returns:
            Risk assessment dictionary
        """
        
        # Extract parameters
        algae_coverage = image_results.get('algae_percentage', 0)
        detection_confidence = image_results.get('detection_confidence', 50)
        quality_metrics = image_results.get('quality_metrics', {})
        
        # Convert image metrics to standard parameters
        chl_a = quality_metrics.get('estimated_chlorophyll', indices.get('Chlorophyll-a', 0))
        turbidity = quality_metrics.get('estimated_turbidity', indices.get('Turbidity', 0))
        
        # Calculate risk based on coverage and quality metrics
        coverage_risk = min(1.0, algae_coverage / 50.0)  # Scale to 0-1
        quality_risk = (chl_a / 100.0 + turbidity / 100.0) / 2
        
        # Combine risks
        combined_risk = (coverage_risk * 0.6 + quality_risk * 0.4)
        
        # Adjust for detection confidence
        confidence_factor = detection_confidence / 100.0
        adjusted_risk = combined_risk * confidence_factor
        
        # Determine risk level
        risk_level = self._determine_risk_level(adjusted_risk)
        
        return {
            'risk_score': adjusted_risk,
            'risk_level': risk_level,
            'algae_coverage_percent': algae_coverage,
            'growth_rate': 0,  # Cannot determine from single image
            'individual_risks': {
                'visual_coverage': coverage_risk,
                'water_quality': quality_risk,
                'detection_confidence': confidence_factor
            },
            'confidence_level': detection_confidence,
            'image_based_assessment': True
        }
    
    def _extract_value(self, parameter: Any) -> float:
        """Extract numeric value from parameter dictionary or direct value"""
        
        if isinstance(parameter, dict):
            return parameter.get('mean', parameter.get('value', 0))
        elif isinstance(parameter, (int, float)):
            return float(parameter)
        else:
            return 0.0
    
    def _calculate_chlorophyll_risk(self, chl_a: float) -> float:
        """Calculate risk score based on chlorophyll-a concentration"""
        
        if chl_a >= self.chlorophyll_thresholds['severe']:
            return 1.0
        elif chl_a >= self.chlorophyll_thresholds['high']:
            return 0.8
        elif chl_a >= self.chlorophyll_thresholds['medium']:
            return 0.5
        elif chl_a >= self.chlorophyll_thresholds['low']:
            return 0.2
        else:
            return 0.0
    
    def _calculate_turbidity_risk(self, turbidity: float) -> float:
        """Calculate risk score based on turbidity"""
        
        if turbidity >= self.turbidity_thresholds['severe']:
            return 1.0
        elif turbidity >= self.turbidity_thresholds['high']:
            return 0.7
        elif turbidity >= self.turbidity_thresholds['medium']:
            return 0.4
        elif turbidity >= self.turbidity_thresholds['low']:
            return 0.1
        else:
            return 0.0
    
    def _calculate_fai_risk(self, fai: float) -> float:
        """Calculate risk score based on Floating Algae Index"""
        
        if fai >= self.fai_thresholds['severe']:
            return 1.0
        elif fai >= self.fai_thresholds['high']:
            return 0.8
        elif fai >= self.fai_thresholds['medium']:
            return 0.5
        elif fai >= self.fai_thresholds['low']:
            return 0.2
        else:
            return 0.0
    
    def _calculate_vegetation_risk(self, ndvi: float, ndwi: float) -> float:
        """Calculate risk based on vegetation indices over water"""
        
        # High NDVI over water areas (NDWI > 0) can indicate algae
        if ndwi > 0:  # Water present
            if ndvi > 0.4:
                return 0.8  # High vegetation over water = likely algae
            elif ndvi > 0.2:
                return 0.4  # Moderate vegetation
            else:
                return 0.1  # Low vegetation
        else:
            # Not over water, lower risk
            return max(0, ndvi * 0.2)
    
    def _apply_environmental_factors(self, base_risk: float, imagery_data: Dict[str, Any]) -> float:
        """Apply environmental factors to adjust risk score"""
        
        adjusted_risk = base_risk
        
        # Seasonal factor (summer months higher risk)
        current_month = datetime.now().month
        if current_month in [6, 7, 8, 9]:  # Summer/early fall
            adjusted_risk *= 1.2
        elif current_month in [4, 5, 10]:  # Spring/late fall
            adjusted_risk *= 1.0
        else:  # Winter
            adjusted_risk *= 0.8
        
        # Cloud cover factor (less reliable data with high clouds)
        cloud_cover = imagery_data.get('metadata', {}).get('cloud_cover', 20)
        if cloud_cover > 50:
            adjusted_risk *= 0.8  # Reduce confidence
        
        # Ensure risk stays within bounds
        return max(0.0, min(1.0, adjusted_risk))
    
    def _estimate_algae_coverage(self, chl_a: float, fai: float, ndvi: float) -> float:
        """Estimate algae coverage percentage from indices"""
        
        # Combine multiple indices to estimate coverage
        chl_coverage = min(50, chl_a * 1.5)  # Chlorophyll-a contribution
        fai_coverage = min(30, fai * 500)    # FAI contribution (scaled up)
        ndvi_coverage = min(20, max(0, (ndvi - 0.1) * 40))  # NDVI contribution
        
        # Weight the contributions
        total_coverage = (
            chl_coverage * 0.5 +
            fai_coverage * 0.3 +
            ndvi_coverage * 0.2
        )
        
        return min(100, max(0, total_coverage))
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine categorical risk level from numeric score"""
        
        if risk_score >= 0.8:
            return "High"
        elif risk_score >= 0.5:
            return "Medium"
        elif risk_score >= 0.2:
            return "Low"
        else:
            return "Minimal"
    
    def _estimate_growth_rate(self, spectral_indices: Dict[str, Any], imagery_data: Dict[str, Any]) -> float:
        """Estimate algae growth rate from available data"""
        
        # This would require temporal data for accurate calculation
        # For now, provide estimate based on current conditions
        
        chl_a = self._extract_value(spectral_indices.get('Chlorophyll-a', {}))
        turbidity = self._extract_value(spectral_indices.get('Turbidity', {}))
        
        # Higher nutrient conditions suggest faster potential growth
        if chl_a > 30:
            base_growth = 5.0  # %/week
        elif chl_a > 15:
            base_growth = 3.0
        elif chl_a > 5:
            base_growth = 1.5
        else:
            base_growth = 0.5
        
        # Adjust for environmental conditions
        seasonal_mult = 1.2 if datetime.now().month in [6, 7, 8] else 0.8
        
        estimated_growth = base_growth * seasonal_mult
        
        # Add some random variation
        variation = np.random.normal(0, 0.5)
        
        return max(-2.0, min(10.0, estimated_growth + variation))
    
    def _identify_risk_factors(self, chl_a: float, turbidity: float, fai: float, ndvi: float) -> List[str]:
        """Identify key risk factors contributing to algae risk"""
        
        factors = []
        
        if chl_a > self.chlorophyll_thresholds['medium']:
            factors.append("Elevated chlorophyll-a levels")
        
        if turbidity > self.turbidity_thresholds['medium']:
            factors.append("High water turbidity")
        
        if fai > self.fai_thresholds['medium']:
            factors.append("Significant floating algae presence")
        
        if ndvi > 0.3:
            factors.append("High vegetation index over water")
        
        # Seasonal factors
        current_month = datetime.now().month
        if current_month in [6, 7, 8, 9]:
            factors.append("Peak algae season (summer/early fall)")
        
        # Temperature factor (estimated)
        if current_month in [7, 8]:
            factors.append("High temperature conditions")
        
        return factors if factors else ["No significant risk factors identified"]
    
    def _calculate_confidence(self, spectral_indices: Dict[str, Any]) -> float:
        """Calculate confidence level of the assessment"""
        
        # Base confidence on data availability and quality
        available_indices = len([k for k, v in spectral_indices.items() if v])
        max_indices = 5  # NDVI, NDWI, Chl-a, Turbidity, FAI
        
        data_completeness = available_indices / max_indices
        
        # Check if data appears realistic
        chl_a = self._extract_value(spectral_indices.get('Chlorophyll-a', {}))
        turbidity = self._extract_value(spectral_indices.get('Turbidity', {}))
        
        data_quality = 1.0
        if chl_a > 200 or turbidity > 200:  # Unrealistic values
            data_quality = 0.5
        
        # Calculate overall confidence
        confidence = (data_completeness * 0.6 + data_quality * 0.4) * 100
        
        return max(20, min(100, confidence))  # Keep between 20-100%
    
    def _analyze_temporal_trend(self, growth_rate: float) -> str:
        """Analyze temporal trend from growth rate"""
        
        if growth_rate > 3:
            return "Rapidly increasing"
        elif growth_rate > 1:
            return "Moderately increasing"
        elif growth_rate > -1:
            return "Stable"
        else:
            return "Decreasing"
    
    def _assess_environmental_conditions(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess environmental conditions affecting algae growth"""
        
        metadata = imagery_data.get('metadata', {})
        
        # Assess season
        current_month = datetime.now().month
        if current_month in [6, 7, 8]:
            season_risk = "High (Summer)"
        elif current_month in [4, 5, 9, 10]:
            season_risk = "Medium (Spring/Fall)"
        else:
            season_risk = "Low (Winter)"
        
        # Assess data quality
        cloud_cover = metadata.get('cloud_cover', 20)
        if cloud_cover < 10:
            data_quality = "Excellent"
        elif cloud_cover < 30:
            data_quality = "Good"
        elif cloud_cover < 50:
            data_quality = "Fair"
        else:
            data_quality = "Poor"
        
        return {
            'seasonal_risk': season_risk,
            'data_quality': data_quality,
            'cloud_cover_percent': cloud_cover,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'satellite_platform': metadata.get('satellite', 'Unknown')
        }
    
    def predict_bloom_probability(self, current_conditions: Dict[str, Any], days_ahead: int = 14) -> Dict[str, Any]:
        """
        Predict probability of algae bloom occurrence
        
        Args:
            current_conditions: Current risk assessment results
            days_ahead: Number of days to predict ahead
            
        Returns:
            Dictionary containing bloom probability prediction
        """
        
        current_risk = current_conditions.get('risk_score', 0)
        growth_rate = current_conditions.get('growth_rate', 0)
        
        # Simple linear growth model (would be more sophisticated with ML)
        predicted_risk = current_risk + (growth_rate * days_ahead / 100)
        predicted_risk = max(0, min(1, predicted_risk))
        
        # Convert to probability categories
        if predicted_risk > 0.8:
            probability_category = "Very High"
            probability_percent = 85
        elif predicted_risk > 0.6:
            probability_category = "High"
            probability_percent = 70
        elif predicted_risk > 0.4:
            probability_category = "Medium"
            probability_percent = 50
        elif predicted_risk > 0.2:
            probability_category = "Low"
            probability_percent = 25
        else:
            probability_category = "Very Low"
            probability_percent = 10
        
        return {
            'prediction_days': days_ahead,
            'bloom_probability_percent': probability_percent,
            'probability_category': probability_category,
            'predicted_risk_score': predicted_risk,
            'confidence': "Medium",  # Simplified model has medium confidence
            'factors_considered': [
                "Current risk level",
                "Estimated growth rate",
                "Seasonal patterns",
                "Historical trends"
            ]
        }
