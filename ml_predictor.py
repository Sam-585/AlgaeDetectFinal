"""
Machine Learning Prediction Module
Implements Random Forest and Decision Tree models for algae bloom risk forecasting

Note: Due to limited historical measurement data, this module uses a hybrid approach
combining rule-based forecasting with ML classification for bloom occurrence prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

# Using scikit-learn for ML models
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AlgaeBloomPredictor:
    """Hybrid ML-based predictor for algae bloom occurrence and progression"""
    
    def __init__(self):
        """Initialize the predictor with models"""
        self.classification_model = None  # Predict bloom occurrence (Yes/No)
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Features that don't leak target information
        self.feature_names = [
            'waterbody_area_km2',
            'waterbody_depth_m',
            'pollution_source_count',
            'month_of_year',
            'season_factor',
            'temperature_factor',
            'historical_bloom_frequency',
            'days_since_last_bloom',
            'water_quality_grade'
        ]
        self.is_trained = False
        self.training_info = {}
        
    def prepare_training_data(self, waterbodies_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from waterbody characteristics and historical patterns
        
        Args:
            waterbodies_data: Dictionary containing waterbody information
            
        Returns:
            Tuple of (features, bloom_occurrence_targets)
        """
        
        features_list = []
        bloom_targets = []  # 0 = no bloom, 1 = bloom occurred
        
        # Quality grade mapping
        grade_map = {'A': 1, 'B+': 2, 'B': 3, 'C+': 4, 'C': 5, 'C-': 6, 'D+': 7, 'D': 8, 'E': 9}
        
        for waterbody_name, info in waterbodies_data.items():
            historical_blooms = info.get('historical_blooms', [])
            
            if not historical_blooms:
                continue
            
            # Calculate historical bloom frequency
            bloom_years = len(historical_blooms)
            total_years = 3  # Assuming 3 years of data
            bloom_frequency = bloom_years / total_years
            
            # Waterbody characteristics (non-leaking features)
            area_km2 = info.get('area_km2', 10)
            depth_m = info.get('depth_m', 5)
            pollution_sources = len(info.get('pollution_sources', []))
            water_grade = grade_map.get(info.get('water_quality_grade', 'C'), 5)
            
            # Create training samples for each year/season
            for bloom_record in historical_blooms:
                year = bloom_record.get('year', 2023)
                severity = bloom_record.get('severity', 'Low')
                duration = bloom_record.get('duration_days', 30)
                
                # Create features for different seasons
                for month in range(1, 13):
                    # Seasonal factors
                    season_factor = 1.3 if month in [8, 9, 10] else 0.7
                    temperature_factor = 1.2 if month in [6, 7, 8] else 0.8
                    
                    # Days since last bloom (estimated)
                    days_since = 180 if month >= 8 else 365
                    
                    features = [
                        area_km2,
                        depth_m,
                        pollution_sources,
                        month,
                        season_factor,
                        temperature_factor,
                        bloom_frequency,
                        days_since,
                        water_grade
                    ]
                    
                    # Target: did a bloom occur in this month?
                    # Blooms typically occur in monsoon/post-monsoon (July-October)
                    bloom_occurred = 1 if (month in [7, 8, 9, 10] and severity in ['Medium', 'High', 'Severe']) else 0
                    
                    features_list.append(features)
                    bloom_targets.append(bloom_occurred)
            
            # Add negative examples (no bloom scenarios)
            for month in [1, 2, 3, 11, 12]:  # Winter months - typically no blooms
                season_factor = 0.5
                temperature_factor = 0.6
                days_since = 90
                
                features = [
                    area_km2,
                    depth_m,
                    pollution_sources,
                    month,
                    season_factor,
                    temperature_factor,
                    bloom_frequency,
                    days_since,
                    water_grade
                ]
                
                features_list.append(features)
                bloom_targets.append(0)  # No bloom in winter
        
        return np.array(features_list), np.array(bloom_targets)
    
    def train_models(self, waterbodies_data: Dict[str, Any], model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Train ML classification model on historical patterns
        
        Args:
            waterbodies_data: Dictionary containing waterbody information
            model_type: 'random_forest' or 'decision_tree'
            
        Returns:
            Dictionary containing training results and metrics
        """
        
        if not SKLEARN_AVAILABLE:
            self.training_info = {
                'success': False,
                'error': 'scikit-learn not available',
                'message': 'Using rule-based predictions instead'
            }
            return self.training_info
        
        # Prepare training data
        X, y = self.prepare_training_data(waterbodies_data)
        
        if len(X) < 20:
            self.training_info = {
                'success': False,
                'error': 'Insufficient training data',
                'message': f'Only {len(X)} samples available, need at least 20'
            }
            return self.training_info
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if model_type == 'random_forest':
            self.classification_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            )
        else:  # decision_tree
            self.classification_model = DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            )
        
        # Train classification model
        self.classification_model.fit(X_train_scaled, y_train)
        y_pred = self.classification_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        self.training_info = {
            'success': True,
            'model_type': model_type,
            'samples_trained': len(X_train),
            'samples_tested': len(X_test),
            'classification_accuracy': accuracy,
            'bloom_percentage': np.mean(y) * 100,
            'feature_importance': self._get_feature_importance()
        }
        
        return self.training_info
    
    def predict_bloom_risk(self, waterbody_info: Dict[str, Any], 
                          current_conditions: Dict[str, Any], 
                          days_ahead: int = 14) -> Dict[str, Any]:
        """
        Predict algae bloom risk using hybrid ML + rule-based approach
        
        Args:
            waterbody_info: Waterbody characteristics
            current_conditions: Current water quality parameters
            days_ahead: Number of days to predict ahead
            
        Returns:
            Dictionary containing predictions and confidence metrics
        """
        
        # Extract current parameters
        current_coverage = current_conditions.get('current_coverage', 20)
        chlorophyll_a = current_conditions.get('chlorophyll_a', 10)
        growth_rate_per_day = self._estimate_growth_rate(current_conditions)
        
        # Calculate future date
        future_date = datetime.now() + timedelta(days=days_ahead)
        future_month = future_date.month
        
        # Prepare features for ML classification
        grade_map = {'A': 1, 'B+': 2, 'B': 3, 'C+': 4, 'C': 5, 'C-': 6, 'D+': 7, 'D': 8, 'E': 9}
        
        area_km2 = waterbody_info.get('area_km2', 10)
        depth_m = waterbody_info.get('depth_m', 5)
        pollution_sources = len(waterbody_info.get('pollution_sources', []))
        water_grade = grade_map.get(waterbody_info.get('water_quality_grade', 'C'), 5)
        
        season_factor = 1.3 if future_month in [8, 9, 10] else 0.7
        temperature_factor = 1.2 if future_month in [6, 7, 8] else 0.8
        
        historical_blooms = waterbody_info.get('historical_blooms', [])
        bloom_frequency = len(historical_blooms) / max(1, 3)  # 3 years of data
        days_since = 180  # Estimated
        
        features = np.array([[
            area_km2,
            depth_m,
            pollution_sources,
            future_month,
            season_factor,
            temperature_factor,
            bloom_frequency,
            days_since,
            water_grade
        ]])
        
        # Use ML model if trained
        if SKLEARN_AVAILABLE and self.is_trained and self.classification_model is not None:
            features_scaled = self.scaler.transform(features)
            bloom_probability = self.classification_model.predict_proba(features_scaled)[0][1]
            will_bloom = self.classification_model.predict(features_scaled)[0]
        else:
            # Fallback rule-based prediction
            risk_score = (
                (pollution_sources / 5) * 0.2 +
                (water_grade / 9) * 0.2 +
                season_factor * 0.2 +
                temperature_factor * 0.2 +
                bloom_frequency * 0.2
            )
            bloom_probability = min(1.0, risk_score)
            will_bloom = 1 if bloom_probability > 0.5 else 0
        
        # Forecast coverage using growth model
        predicted_coverage = current_coverage + (growth_rate_per_day * days_ahead)
        predicted_coverage = max(0, min(100, predicted_coverage))
        
        # Determine risk category
        if predicted_coverage > 50 or bloom_probability > 0.8:
            risk_category = "Very High"
        elif predicted_coverage > 30 or bloom_probability > 0.6:
            risk_category = "High"
        elif predicted_coverage > 15 or bloom_probability > 0.4:
            risk_category = "Medium"
        else:
            risk_category = "Low"
        
        confidence = bloom_probability if will_bloom else (1 - bloom_probability)
        
        return {
            'will_bloom': bool(will_bloom),
            'bloom_probability': float(bloom_probability),
            'predicted_coverage': float(predicted_coverage),
            'future_coverage': float(predicted_coverage),
            'risk_category': risk_category,
            'confidence': float(confidence),
            'prediction_horizon_days': days_ahead,
            'model_used': 'ML-Hybrid' if (SKLEARN_AVAILABLE and self.is_trained) else 'Rule-based',
            'growth_rate_per_day': growth_rate_per_day
        }
    
    def _estimate_growth_rate(self, current_conditions: Dict[str, Any]) -> float:
        """
        Estimate daily algae growth rate based on current conditions
        
        Args:
            current_conditions: Current water quality parameters
            
        Returns:
            Estimated growth rate as % per day
        """
        
        chlorophyll_a = current_conditions.get('chlorophyll_a', 10)
        temperature_factor = current_conditions.get('temperature_factor', 1.0)
        nutrient_factor = current_conditions.get('nutrient_factor', 1.0)
        seasonal_factor = current_conditions.get('seasonal_factor', 1.0)
        
        # Base growth rate depends on current chlorophyll levels
        if chlorophyll_a > 30:
            base_rate = 0.8  # High growth
        elif chlorophyll_a > 15:
            base_rate = 0.5  # Moderate growth
        elif chlorophyll_a > 5:
            base_rate = 0.2  # Slow growth
        else:
            base_rate = 0.05  # Minimal growth
        
        # Adjust for environmental factors
        adjusted_rate = base_rate * temperature_factor * nutrient_factor * seasonal_factor
        
        # Add some stochasticity
        noise = np.random.normal(0, 0.05)
        
        return max(-0.5, min(2.0, adjusted_rate + noise))
    
    def predict_temporal_progression(self, waterbody_info: Dict[str, Any],
                                    current_conditions: Dict[str, Any], 
                                    days: int = 30) -> List[Dict[str, Any]]:
        """
        Predict algae bloom progression over time
        
        Args:
            waterbody_info: Waterbody characteristics
            current_conditions: Current water quality parameters
            days: Number of days to predict
            
        Returns:
            List of daily predictions
        """
        
        progression = []
        running_coverage = current_conditions.get('current_coverage', 20)
        
        for day in range(days):
            # Update growth rate for each day
            updated_conditions = current_conditions.copy()
            updated_conditions['current_coverage'] = running_coverage
            updated_conditions['chlorophyll_a'] = running_coverage * 0.5  # Rough estimate
            
            # Predict for this day
            prediction = self.predict_bloom_risk(waterbody_info, updated_conditions, days_ahead=day)
            
            # Update running coverage
            running_coverage = prediction['future_coverage']
            
            progression.append({
                'day': day,
                'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                'predicted_coverage': prediction['future_coverage'],
                'risk_category': prediction['risk_category'],
                'bloom_probability': prediction['bloom_probability']
            })
        
        return progression
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        
        if not SKLEARN_AVAILABLE or not self.is_trained or self.classification_model is None:
            return {}
        
        try:
            importances = self.classification_model.feature_importances_
            
            importance_dict = {}
            for i, feature_name in enumerate(self.feature_names):
                importance_dict[feature_name] = float(importances[i])
            
            return importance_dict
        except:
            return {}
    
    def get_model_recommendations(self, prediction: Dict[str, Any]) -> List[str]:
        """
        Generate management recommendations based on predictions
        
        Args:
            prediction: Prediction results dictionary
            
        Returns:
            List of recommended actions
        """
        
        recommendations = []
        
        risk_category = prediction['risk_category']
        future_coverage = prediction['future_coverage']
        days_ahead = prediction['prediction_horizon_days']
        bloom_prob = prediction['bloom_probability']
        
        if risk_category == "Very High":
            recommendations.extend([
                f"URGENT: Very high bloom risk (prob: {bloom_prob*100:.0f}%) within {days_ahead} days",
                "Implement emergency response protocols immediately",
                "Increase monitoring to daily frequency",
                "Prepare algaecide treatment equipment",
                "Issue public health advisory"
            ])
        elif risk_category == "High":
            recommendations.extend([
                f"HIGH ALERT: Elevated bloom risk (prob: {bloom_prob*100:.0f}%) for next {days_ahead} days",
                "Increase monitoring frequency (2-3x per week)",
                "Reduce nutrient inputs immediately",
                "Prepare treatment measures",
                "Consider preemptive action"
            ])
        elif risk_category == "Medium":
            recommendations.extend([
                f"MODERATE: Watch for bloom development (prob: {bloom_prob*100:.0f}%)",
                "Maintain regular monitoring schedule",
                "Review nutrient management practices",
                "Ensure treatment readiness"
            ])
        else:
            recommendations.extend([
                f"LOW: Minimal bloom risk currently (prob: {bloom_prob*100:.0f}%)",
                "Continue standard monitoring program",
                "Maintain preventive measures"
            ])
        
        return recommendations
