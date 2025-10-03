"""
Uttarakhand Waterbodies Database
Contains information about major water bodies in Uttarakhand region for algae monitoring
"""

# Major water bodies in Uttarakhand/Roorkee region with their characteristics
UTTARAKHAND_WATERBODIES = {
    "Ganga Canal (Roorkee)": {
        "lat": 29.8543,
        "lon": 77.8880,
        "type": "Canal",
        "area_km2": 12.5,
        "depth_m": 3.2,
        "primary_use": "Irrigation and Industrial",
        "pollution_sources": ["Agricultural runoff", "Industrial discharge", "Urban waste"],
        "recent_issues": [
            "Increased algae growth during monsoon season",
            "Industrial effluent discharge from paper mills",
            "Agricultural pesticide contamination"
        ],
        "historical_blooms": [
            {"year": 2023, "severity": "Medium", "duration_days": 45},
            {"year": 2022, "severity": "High", "duration_days": 62},
            {"year": 2021, "severity": "Low", "duration_days": 28}
        ],
        "management_authority": "Irrigation Department, Uttarakhand",
        "monitoring_frequency": "Monthly",
        "water_quality_grade": "C",
        "seasonal_variation": {
            "monsoon": "High algae risk due to nutrient runoff",
            "post_monsoon": "Moderate algae growth",
            "winter": "Low algae activity",
            "summer": "Medium risk due to temperature rise"
        }
    },
    
    "Solani River": {
        "lat": 29.8156,
        "lon": 77.9311,
        "type": "River",
        "area_km2": 8.3,
        "depth_m": 2.1,
        "primary_use": "Domestic water supply",
        "pollution_sources": ["Sewage discharge", "Agricultural runoff", "Religious activities"],
        "recent_issues": [
            "Eutrophication due to sewage discharge",
            "Algal blooms affecting water treatment plants",
            "High coliform bacterial contamination"
        ],
        "historical_blooms": [
            {"year": 2023, "severity": "High", "duration_days": 38},
            {"year": 2022, "severity": "Medium", "duration_days": 51},
            {"year": 2021, "severity": "High", "duration_days": 44}
        ],
        "management_authority": "Uttarakhand Jal Sansthan",
        "monitoring_frequency": "Bi-weekly",
        "water_quality_grade": "D",
        "seasonal_variation": {
            "monsoon": "Severe algae blooms due to sewage overflow",
            "post_monsoon": "High algae concentration",
            "winter": "Moderate algae levels",
            "summer": "High risk due to low flow and high temperature"
        }
    },
    
    "Haridwar Canal System": {
        "lat": 29.9457,
        "lon": 78.1642,
        "type": "Canal Network",
        "area_km2": 45.7,
        "depth_m": 4.5,
        "primary_use": "Irrigation and Hydropower",
        "pollution_sources": ["Agricultural chemicals", "Urban runoff", "Industrial waste"],
        "recent_issues": [
            "Seasonal algae blooms affecting irrigation efficiency",
            "Pesticide residues from intensive farming",
            "Siltation reducing water flow"
        ],
        "historical_blooms": [
            {"year": 2023, "severity": "Medium", "duration_days": 55},
            {"year": 2022, "severity": "Low", "duration_days": 33},
            {"year": 2021, "severity": "Medium", "duration_days": 41}
        ],
        "management_authority": "Upper Ganga Canal Division",
        "monitoring_frequency": "Weekly during irrigation season",
        "water_quality_grade": "B",
        "seasonal_variation": {
            "monsoon": "High nutrient loading, moderate algae growth",
            "post_monsoon": "Peak algae season",
            "winter": "Minimal algae activity",
            "summer": "Increasing algae risk"
        }
    },
    
    "Bhimgoda Barrage": {
        "lat": 29.9558,
        "lon": 78.1734,
        "type": "Reservoir",
        "area_km2": 6.2,
        "depth_m": 8.7,
        "primary_use": "Water regulation and hydropower",
        "pollution_sources": ["Religious waste", "Tourism activities", "Upstream pollution"],
        "recent_issues": [
            "Floating algae mats during festivals",
            "High organic loading from religious activities",
            "Tourism-related pollution"
        ],
        "historical_blooms": [
            {"year": 2023, "severity": "Low", "duration_days": 22},
            {"year": 2022, "severity": "Medium", "duration_days": 35},
            {"year": 2021, "severity": "Low", "duration_days": 18}
        ],
        "management_authority": "Uttarakhand Irrigation Department",
        "monitoring_frequency": "Daily during festival periods",
        "water_quality_grade": "C+",
        "seasonal_variation": {
            "monsoon": "High flow reduces algae concentration",
            "post_monsoon": "Moderate algae growth",
            "winter": "Festival-related algae spikes",
            "summer": "Low water levels increase algae risk"
        }
    },
    
    "Eastern Yamuna Canal": {
        "lat": 29.8867,
        "lon": 77.7544,
        "type": "Canal",
        "area_km2": 18.9,
        "depth_m": 3.8,
        "primary_use": "Irrigation",
        "pollution_sources": ["Agricultural runoff", "Fertilizer leaching", "Pesticide contamination"],
        "recent_issues": [
            "Nitrogen and phosphorus enrichment from fertilizers",
            "Algae blooms clogging irrigation infrastructure",
            "Reduced oxygen levels affecting aquatic life"
        ],
        "historical_blooms": [
            {"year": 2023, "severity": "High", "duration_days": 67},
            {"year": 2022, "severity": "High", "duration_days": 58},
            {"year": 2021, "severity": "Medium", "duration_days": 39}
        ],
        "management_authority": "Eastern Yamuna Canal Division",
        "monitoring_frequency": "Bi-weekly",
        "water_quality_grade": "D+",
        "seasonal_variation": {
            "monsoon": "Peak nutrient runoff, severe algae blooms",
            "post_monsoon": "Continued high algae concentration",
            "winter": "Reduced algae activity",
            "summer": "Temperature-driven algae growth"
        }
    },
    
    "Raiwala Pond": {
        "lat": 30.0234,
        "lon": 78.1456,
        "type": "Pond",
        "area_km2": 1.2,
        "depth_m": 2.8,
        "primary_use": "Local water supply and fisheries",
        "pollution_sources": ["Domestic sewage", "Cattle washing", "Agricultural runoff"],
        "recent_issues": [
            "Severe eutrophication from sewage discharge",
            "Dense algae mats covering entire surface",
            "Fish kills due to oxygen depletion"
        ],
        "historical_blooms": [
            {"year": 2023, "severity": "Severe", "duration_days": 89},
            {"year": 2022, "severity": "High", "duration_days": 76},
            {"year": 2021, "severity": "High", "duration_days": 82}
        ],
        "management_authority": "Local Panchayat",
        "monitoring_frequency": "Monthly",
        "water_quality_grade": "E",
        "seasonal_variation": {
            "monsoon": "Dilution reduces algae but increases nutrients",
            "post_monsoon": "Explosive algae growth",
            "winter": "Persistent algae cover",
            "summer": "Extreme algae concentration"
        }
    },
    
    "Dehradun Canal": {
        "lat": 30.3165,
        "lon": 78.0322,
        "type": "Canal",
        "area_km2": 22.1,
        "depth_m": 4.1,
        "primary_use": "Urban water supply and irrigation",
        "pollution_sources": ["Urban sewage", "Industrial effluents", "Stormwater runoff"],
        "recent_issues": [
            "Industrial chemical contamination",
            "Urban stormwater carrying pollutants",
            "Algae blooms affecting water treatment"
        ],
        "historical_blooms": [
            {"year": 2023, "severity": "Medium", "duration_days": 42},
            {"year": 2022, "severity": "Low", "duration_days": 28},
            {"year": 2021, "severity": "Medium", "duration_days": 36}
        ],
        "management_authority": "Dehradun Municipal Corporation",
        "monitoring_frequency": "Weekly",
        "water_quality_grade": "C",
        "seasonal_variation": {
            "monsoon": "High pollution load, variable algae growth",
            "post_monsoon": "Moderate algae activity",
            "winter": "Lower algae levels",
            "summer": "Industrial discharge increases algae risk"
        }
    },
    
    "Rishikesh Ghat Complex": {
        "lat": 30.1030,
        "lon": 78.3017,
        "type": "River section",
        "area_km2": 5.8,
        "depth_m": 6.2,
        "primary_use": "Religious activities and tourism",
        "pollution_sources": ["Religious offerings", "Tourism waste", "Sewage discharge"],
        "recent_issues": [
            "Organic pollution from religious activities",
            "Tourist-generated waste",
            "Seasonal algae blooms during pilgrim seasons"
        ],
        "historical_blooms": [
            {"year": 2023, "severity": "Low", "duration_days": 25},
            {"year": 2022, "severity": "Medium", "duration_days": 31},
            {"year": 2021, "severity": "Low", "duration_days": 19}
        ],
        "management_authority": "Uttarakhand Tourism Board",
        "monitoring_frequency": "Daily during peak season",
        "water_quality_grade": "B",
        "seasonal_variation": {
            "monsoon": "High flow maintains water quality",
            "post_monsoon": "Moderate tourism impact",
            "winter": "Peak pilgrimage season, higher pollution",
            "summer": "High tourist season, increased waste"
        }
    },
    
    "Tehri Dam Reservoir": {
        "lat": 30.3773,
        "lon": 78.4804,
        "type": "Large reservoir",
        "area_km2": 52.0,
        "depth_m": 85.5,
        "primary_use": "Hydroelectric power generation",
        "pollution_sources": ["Upstream river pollution", "Tributary inflows", "Tourism activities"],
        "recent_issues": [
            "Stratification leading to algae blooms in upper layers",
            "Nutrient accumulation in dead storage zone",
            "Seasonal algae blooms affecting turbine efficiency"
        ],
        "historical_blooms": [
            {"year": 2023, "severity": "Low", "duration_days": 35},
            {"year": 2022, "severity": "Low", "duration_days": 29},
            {"year": 2021, "severity": "Medium", "duration_days": 47}
        ],
        "management_authority": "Tehri Hydro Development Corporation",
        "monitoring_frequency": "Weekly",
        "water_quality_grade": "B+",
        "seasonal_variation": {
            "monsoon": "High inflow brings nutrients, limited algae growth",
            "post_monsoon": "Thermal stratification promotes algae blooms",
            "winter": "Minimal algae activity due to cold temperatures",
            "summer": "Surface heating can trigger algae growth"
        }
    },
    
    "Mussoorie Lake": {
        "lat": 30.4598,
        "lon": 78.0644,
        "type": "Artificial lake",
        "area_km2": 0.8,
        "depth_m": 4.5,
        "primary_use": "Recreation and tourism",
        "pollution_sources": ["Tourist waste", "Hotel discharge", "Surface runoff"],
        "recent_issues": [
            "Eutrophication from hotel wastewater",
            "Tourist activities increasing organic load",
            "Algae blooms during tourist season"
        ],
        "historical_blooms": [
            {"year": 2023, "severity": "Medium", "duration_days": 52},
            {"year": 2022, "severity": "High", "duration_days": 48},
            {"year": 2021, "severity": "Medium", "duration_days": 41}
        ],
        "management_authority": "Mussoorie Municipal Board",
        "monitoring_frequency": "Bi-weekly during tourist season",
        "water_quality_grade": "C-",
        "seasonal_variation": {
            "monsoon": "Runoff increases nutrient loading",
            "post_monsoon": "Peak algae growth period",
            "winter": "Low tourist activity, reduced algae",
            "summer": "High tourist season, increased algae risk"
        }
    }
}

# Water quality grading system explanation
WATER_QUALITY_GRADES = {
    "A": {
        "description": "Excellent - Suitable for all purposes",
        "algae_risk": "Very Low",
        "treatment_required": "Minimal"
    },
    "B": {
        "description": "Good - Suitable for most purposes with basic treatment",
        "algae_risk": "Low",
        "treatment_required": "Basic filtration"
    },
    "B+": {
        "description": "Good+ - Above average quality",
        "algae_risk": "Low",
        "treatment_required": "Basic treatment"
    },
    "C": {
        "description": "Fair - Requires treatment for most uses",
        "algae_risk": "Medium",
        "treatment_required": "Standard treatment"
    },
    "C+": {
        "description": "Fair+ - Slightly above average",
        "algae_risk": "Medium",
        "treatment_required": "Standard treatment"
    },
    "C-": {
        "description": "Fair- - Below average quality",
        "algae_risk": "Medium-High",
        "treatment_required": "Enhanced treatment"
    },
    "D": {
        "description": "Poor - Significant treatment required",
        "algae_risk": "High",
        "treatment_required": "Advanced treatment"
    },
    "D+": {
        "description": "Poor+ - Slightly better than poor",
        "algae_risk": "High",
        "treatment_required": "Advanced treatment"
    },
    "E": {
        "description": "Very Poor - Extensive treatment required",
        "algae_risk": "Very High",
        "treatment_required": "Extensive treatment"
    }
}

# Regional characteristics affecting algae growth
REGIONAL_FACTORS = {
    "climate": {
        "temperature_range": "15-35°C",
        "monsoon_months": [6, 7, 8, 9],
        "peak_algae_season": [8, 9, 10],
        "low_algae_season": [12, 1, 2]
    },
    "geology": {
        "dominant_rock_type": "Sedimentary and metamorphic",
        "soil_type": "Alluvial plains with high nutrient content",
        "natural_phosphorus": "Medium levels from rock weathering"
    },
    "agriculture": {
        "fertilizer_use": "High - NPK fertilizers commonly used",
        "irrigation_type": "Canal and groundwater",
        "crop_seasons": ["Kharif (Jun-Oct)", "Rabi (Nov-Apr)"],
        "peak_runoff_period": "July-September"
    },
    "urbanization": {
        "sewage_treatment": "Limited - many areas discharge untreated sewage",
        "industrial_zones": ["Roorkee", "Haridwar", "Dehradun"],
        "population_growth": "Moderate to high in urban centers"
    }
}

# Pollution source categories with typical characteristics
POLLUTION_SOURCES = {
    "agricultural_runoff": {
        "primary_pollutants": ["Nitrogen", "Phosphorus", "Pesticides"],
        "seasonal_pattern": "Peak during monsoon and post-monsoon",
        "algae_impact": "High - provides essential nutrients for growth",
        "mitigation_strategies": [
            "Buffer strips along water bodies",
            "Controlled fertilizer application",
            "Organic farming practices"
        ]
    },
    "sewage_discharge": {
        "primary_pollutants": ["Organic matter", "Nitrogen", "Phosphorus", "Pathogens"],
        "seasonal_pattern": "Consistent year-round with monsoon dilution",
        "algae_impact": "Very High - rich in nutrients",
        "mitigation_strategies": [
            "Sewage treatment plants",
            "Constructed wetlands",
            "Septic system upgrades"
        ]
    },
    "industrial_discharge": {
        "primary_pollutants": ["Heavy metals", "Chemicals", "Organic compounds"],
        "seasonal_pattern": "Consistent with occasional peak discharges",
        "algae_impact": "Variable - can inhibit or promote growth",
        "mitigation_strategies": [
            "Effluent treatment plants",
            "Zero liquid discharge systems",
            "Regulatory monitoring"
        ]
    },
    "religious_activities": {
        "primary_pollutants": ["Organic matter", "Flowers", "Oil", "Ash"],
        "seasonal_pattern": "Peak during festivals and pilgrimage seasons",
        "algae_impact": "Medium - organic matter promotes growth",
        "mitigation_strategies": [
            "Eco-friendly materials",
            "Collection systems for offerings",
            "Public awareness campaigns"
        ]
    },
    "tourism_waste": {
        "primary_pollutants": ["Organic waste", "Plastics", "Personal care products"],
        "seasonal_pattern": "Peak during tourist seasons",
        "algae_impact": "Medium - organic fraction promotes growth",
        "mitigation_strategies": [
            "Waste collection systems",
            "Tourist education",
            "Sustainable tourism practices"
        ]
    }
}

def get_waterbody_info(name: str) -> dict:
    """
    Get detailed information about a specific waterbody
    
    Args:
        name: Name of the waterbody
        
    Returns:
        Dictionary containing waterbody information
    """
    return UTTARAKHAND_WATERBODIES.get(name, {})

def get_waterbodies_by_type(waterbody_type: str) -> dict:
    """
    Get all waterbodies of a specific type
    
    Args:
        waterbody_type: Type of waterbody (e.g., 'River', 'Canal', 'Pond')
        
    Returns:
        Dictionary of matching waterbodies
    """
    return {name: info for name, info in UTTARAKHAND_WATERBODIES.items() 
            if info.get('type') == waterbody_type}

def get_high_risk_waterbodies() -> dict:
    """
    Get waterbodies with high algae risk based on recent issues
    
    Returns:
        Dictionary of high-risk waterbodies
    """
    high_risk = {}
    for name, info in UTTARAKHAND_WATERBODIES.items():
        # Consider high risk if recent blooms were severe or if water quality is poor
        recent_blooms = info.get('historical_blooms', [])
        if recent_blooms:
            latest_bloom = recent_blooms[0]  # Most recent year
            if (latest_bloom.get('severity') in ['High', 'Severe'] or 
                info.get('water_quality_grade', 'C') in ['D', 'D+', 'E']):
                high_risk[name] = info
    
    return high_risk

def get_seasonal_risk_assessment(month: int) -> dict:
    """
    Get seasonal algae risk assessment for all waterbodies
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Dictionary with seasonal risk information
    """
    seasonal_info = {}
    
    # Determine season
    if month in [6, 7, 8, 9]:
        season = "monsoon"
    elif month in [10, 11]:
        season = "post_monsoon"
    elif month in [12, 1, 2]:
        season = "winter"
    else:
        season = "summer"
    
    for name, info in UTTARAKHAND_WATERBODIES.items():
        seasonal_variation = info.get('seasonal_variation', {})
        seasonal_info[name] = {
            'current_season': season,
            'risk_description': seasonal_variation.get(season, 'No seasonal data available'),
            'base_risk': info.get('water_quality_grade', 'C')
        }
    
    return seasonal_info

