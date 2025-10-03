"""
Google Earth Engine Helper Module
Handles satellite imagery retrieval and processing using GEE API
"""

import ee
import geemap
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class GEEHelper:
    """Helper class for Google Earth Engine operations"""
    
    def __init__(self):
        """Initialize GEE with authentication"""
        try:
            # Try to initialize Earth Engine
            ee.Initialize()
            self.authenticated = True
        except Exception as e:
            print(f"GEE Authentication failed: {str(e)}")
            self.authenticated = False
    
    def get_imagery(self, lat, lon, start_date, end_date, satellite="Sentinel-2", max_cloud=20):
        """
        Retrieve satellite imagery for specified location and date range
        
        Args:
            lat (float): Latitude of center point
            lon (float): Longitude of center point
            start_date (datetime): Start date for imagery search
            end_date (datetime): End date for imagery search
            satellite (str): Satellite platform ('Sentinel-2', 'Landsat 8/9', 'MODIS')
            max_cloud (int): Maximum cloud coverage percentage
        
        Returns:
            dict: Dictionary containing imagery data and metadata
        """
        
        if not self.authenticated:
            # Return simulated data if GEE is not available
            return self._generate_simulated_imagery(lat, lon, start_date, end_date, satellite)
        
        try:
            # Define area of interest
            point = ee.Geometry.Point([lon, lat])
            aoi = point.buffer(1000)  # 1km radius
            
            # Select appropriate collection based on satellite
            if satellite == "Sentinel-2":
                collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                    .filterBounds(aoi) \
                    .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))
                
            elif satellite == "Landsat 8/9":
                collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                    .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')) \
                    .filterBounds(aoi) \
                    .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
                    .filter(ee.Filter.lt('CLOUD_COVER', max_cloud))
                
            elif satellite == "MODIS":
                collection = ee.ImageCollection('MODIS/061/MOD09GA') \
                    .filterBounds(aoi) \
                    .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            # Get the most recent image
            image = collection.sort('system:time_start', False).first()
            
            if image is None:
                return None
            
            # Extract band data
            imagery_data = self._extract_band_data(image, aoi, satellite)
            
            return imagery_data
            
        except Exception as e:
            print(f"Error retrieving GEE imagery: {str(e)}")
            return self._generate_simulated_imagery(lat, lon, start_date, end_date, satellite)
    
    def _extract_band_data(self, image, aoi, satellite):
        """Extract band data from GEE image"""
        
        try:
            # Define band names based on satellite
            if satellite == "Sentinel-2":
                bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']  # Blue, Green, Red, NIR, SWIR1, SWIR2
                scale = 10
            elif satellite in ["Landsat 8/9"]:
                bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']  # Blue, Green, Red, NIR, SWIR1, SWIR2
                scale = 30
            else:  # MODIS
                bands = ['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b01', 'sur_refl_b02']  # Blue, Green, Red, NIR
                scale = 250
            
            # Get pixel values
            pixel_data = image.select(bands).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=scale,
                maxPixels=1e9
            ).getInfo()
            
            # Get image metadata
            properties = image.getInfo()['properties']
            
            return {
                'bands': pixel_data,
                'metadata': {
                    'satellite': satellite,
                    'date': properties.get('system:time_start'),
                    'cloud_cover': properties.get('CLOUDY_PIXEL_PERCENTAGE', properties.get('CLOUD_COVER', 0)),
                    'scale': scale
                },
                'geometry': aoi.getInfo()
            }
            
        except Exception as e:
            print(f"Error extracting band data: {str(e)}")
            return None
    
    def _generate_simulated_imagery(self, lat, lon, start_date, end_date, satellite):
        """Generate simulated imagery data when GEE is unavailable"""
        
        # Simulate realistic band values for water bodies
        if satellite == "Sentinel-2":
            bands = {
                'B2': np.random.uniform(800, 1200),   # Blue
                'B3': np.random.uniform(600, 1000),   # Green  
                'B4': np.random.uniform(400, 800),    # Red
                'B8': np.random.uniform(200, 600),    # NIR
                'B11': np.random.uniform(100, 400),   # SWIR1
                'B12': np.random.uniform(50, 300)     # SWIR2
            }
            scale = 10
        elif satellite in ["Landsat 8/9"]:
            bands = {
                'SR_B2': np.random.uniform(8000, 12000),  # Blue
                'SR_B3': np.random.uniform(6000, 10000),  # Green
                'SR_B4': np.random.uniform(4000, 8000),   # Red
                'SR_B5': np.random.uniform(2000, 6000),   # NIR
                'SR_B6': np.random.uniform(1000, 4000),   # SWIR1
                'SR_B7': np.random.uniform(500, 3000)     # SWIR2
            }
            scale = 30
        else:  # MODIS
            bands = {
                'sur_refl_b03': np.random.uniform(0.05, 0.15),  # Blue
                'sur_refl_b04': np.random.uniform(0.03, 0.12),  # Green
                'sur_refl_b01': np.random.uniform(0.02, 0.08),  # Red
                'sur_refl_b02': np.random.uniform(0.01, 0.06)   # NIR
            }
            scale = 250
        
        return {
            'bands': bands,
            'metadata': {
                'satellite': satellite,
                'date': int(datetime.now().timestamp() * 1000),
                'cloud_cover': np.random.uniform(5, 25),
                'scale': scale,
                'simulated': True
            },
            'geometry': {
                'type': 'Point',
                'coordinates': [lon, lat]
            }
        }
    
    def get_time_series(self, lat, lon, start_date, end_date, satellite="Sentinel-2"):
        """Get time series of imagery for temporal analysis"""
        
        if not self.authenticated:
            return self._generate_simulated_time_series(lat, lon, start_date, end_date)
        
        try:
            point = ee.Geometry.Point([lon, lat])
            aoi = point.buffer(1000)
            
            # Get collection
            if satellite == "Sentinel-2":
                collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                    .filterBounds(aoi) \
                    .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            
            # Extract time series data
            time_series = []
            images = collection.limit(20).getInfo()  # Limit to 20 most recent images
            
            for img_info in images['features']:
                img = ee.Image(img_info['id'])
                data = self._extract_band_data(img, aoi, satellite)
                if data:
                    time_series.append(data)
            
            return time_series
            
        except Exception as e:
            print(f"Error retrieving time series: {str(e)}")
            return self._generate_simulated_time_series(lat, lon, start_date, end_date)
    
    def _generate_simulated_time_series(self, lat, lon, start_date, end_date):
        """Generate simulated time series data"""
        
        time_series = []
        current_date = start_date
        
        while current_date <= end_date:
            # Add some temporal variation
            day_of_year = current_date.timetuple().tm_yday
            seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
            
            bands = {
                'B2': np.random.uniform(800, 1200) * seasonal_factor,
                'B3': np.random.uniform(600, 1000) * seasonal_factor,
                'B4': np.random.uniform(400, 800) * seasonal_factor,
                'B8': np.random.uniform(200, 600) * seasonal_factor,
                'B11': np.random.uniform(100, 400) * seasonal_factor,
                'B12': np.random.uniform(50, 300) * seasonal_factor
            }
            
            time_series.append({
                'bands': bands,
                'metadata': {
                    'satellite': 'Sentinel-2',
                    'date': int(current_date.timestamp() * 1000),
                    'cloud_cover': np.random.uniform(5, 25),
                    'scale': 10,
                    'simulated': True
                },
                'geometry': {
                    'type': 'Point',
                    'coordinates': [lon, lat]
                }
            })
            
            # Increment by ~10 days
            current_date += timedelta(days=10)
        
        return time_series
    
    def calculate_water_extent(self, imagery_data):
        """Calculate water extent using NDWI"""
        
        bands = imagery_data['bands']
        satellite = imagery_data['metadata']['satellite']
        
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
        
        # Water extent based on NDWI threshold
        water_pixels = 1 if ndwi > 0 else 0
        total_pixels = 1
        
        return {
            'water_extent_percent': (water_pixels / total_pixels) * 100,
            'ndwi': ndwi
        }
