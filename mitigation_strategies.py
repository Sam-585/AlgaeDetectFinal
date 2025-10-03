"""
Mitigation Strategies for Algae Bloom Management
Provides actionable recommendations based on risk levels and conditions
"""

# Mitigation strategies organized by risk level
MITIGATION_STRATEGIES = {
    "Minimal": [
        {
            "title": "Preventive Monitoring",
            "description": "Implement regular water quality monitoring to detect early signs of nutrient enrichment or algae growth before problems develop.",
            "cost": "Low ($500-1,500/month)",
            "timeline": "Ongoing",
            "priority": "Medium",
            "effectiveness": "High for prevention",
            "implementation_steps": [
                "Set up monthly water sampling schedule",
                "Monitor key parameters: chlorophyll-a, nutrients, turbidity",
                "Establish baseline water quality data",
                "Train local staff in basic water quality assessment"
            ]
        },
        {
            "title": "Nutrient Source Control",
            "description": "Identify and minimize nutrient inputs from agricultural runoff, sewage, and other pollution sources in the watershed.",
            "cost": "Medium ($2,000-8,000)",
            "timeline": "3-6 months",
            "priority": "Medium",
            "effectiveness": "Very High",
            "implementation_steps": [
                "Conduct watershed pollution source survey",
                "Implement best management practices for agriculture",
                "Upgrade sewage collection and treatment systems",
                "Create buffer zones around water bodies"
            ]
        },
        {
            "title": "Community Education",
            "description": "Educate local communities about activities that contribute to algae growth and promote water-friendly practices.",
            "cost": "Low ($1,000-3,000)",
            "timeline": "2-4 weeks",
            "priority": "Medium",
            "effectiveness": "Medium-High",
            "implementation_steps": [
                "Develop educational materials in local languages",
                "Conduct community workshops",
                "Partner with local schools for awareness programs",
                "Create signage around water bodies"
            ]
        }
    ],
    
    "Low": [
        {
            "title": "Enhanced Monitoring Program",
            "description": "Increase monitoring frequency and add advanced parameters to track water quality trends and detect early warning signs.",
            "cost": "Medium ($3,000-6,000/month)",
            "timeline": "Immediate implementation",
            "priority": "High",
            "effectiveness": "High",
            "implementation_steps": [
                "Increase sampling frequency to bi-weekly",
                "Add satellite imagery monitoring",
                "Install automated water quality sensors",
                "Establish early warning thresholds"
            ]
        },
        {
            "title": "Watershed Management",
            "description": "Implement comprehensive watershed management practices to reduce nutrient loading and improve water quality.",
            "cost": "High ($10,000-25,000)",
            "timeline": "6-12 months",
            "priority": "High",
            "effectiveness": "Very High",
            "implementation_steps": [
                "Develop watershed management plan",
                "Implement erosion control measures",
                "Establish riparian buffer zones",
                "Regulate agricultural chemical use"
            ]
        },
        {
            "title": "Aeration Systems",
            "description": "Install mechanical aeration systems to increase dissolved oxygen levels and prevent anaerobic conditions that promote algae growth.",
            "cost": "Medium ($5,000-15,000)",
            "timeline": "1-3 months",
            "priority": "Medium",
            "effectiveness": "Medium-High",
            "implementation_steps": [
                "Assess water body size and depth",
                "Select appropriate aeration technology",
                "Install and commission aeration systems",
                "Monitor dissolved oxygen levels regularly"
            ]
        },
        {
            "title": "Biological Controls",
            "description": "Introduce beneficial microorganisms or aquatic plants that compete with algae for nutrients.",
            "cost": "Low-Medium ($2,000-8,000)",
            "timeline": "2-6 months",
            "priority": "Low",
            "effectiveness": "Medium",
            "implementation_steps": [
                "Study ecosystem compatibility",
                "Select appropriate biological agents",
                "Implement pilot program",
                "Monitor ecosystem response"
            ]
        }
    ],
    
    "Medium": [
        {
            "title": "Immediate Water Use Restrictions",
            "description": "Implement temporary restrictions on water use for drinking, recreation, and agriculture until algae levels decrease.",
            "cost": "Low (administrative costs)",
            "timeline": "Immediate",
            "priority": "Critical",
            "effectiveness": "High for public safety",
            "implementation_steps": [
                "Issue public health advisory",
                "Restrict recreational water activities",
                "Provide alternative water sources",
                "Post warning signs at water access points"
            ]
        },
        {
            "title": "Chemical Treatment",
            "description": "Apply approved algaecides or other chemical treatments to reduce existing algae populations while addressing root causes.",
            "cost": "Medium ($3,000-10,000)",
            "timeline": "1-2 weeks",
            "priority": "High",
            "effectiveness": "High (short-term)",
            "implementation_steps": [
                "Obtain necessary permits for chemical treatment",
                "Select appropriate algaecide for local conditions",
                "Apply treatment according to manufacturer guidelines",
                "Monitor treatment effectiveness and side effects"
            ]
        },
        {
            "title": "Nutrient Precipitation",
            "description": "Use chemical precipitation to remove excess phosphorus from the water column and sediments.",
            "cost": "High ($8,000-20,000)",
            "timeline": "2-4 weeks",
            "priority": "High",
            "effectiveness": "High",
            "implementation_steps": [
                "Conduct water chemistry analysis",
                "Select appropriate precipitation agent (alum, iron salts)",
                "Calculate required dosage",
                "Apply treatment and monitor results"
            ]
        },
        {
            "title": "Sediment Removal",
            "description": "Remove nutrient-rich sediments that serve as internal nutrient source for algae growth.",
            "cost": "Very High ($20,000-50,000)",
            "timeline": "3-6 months",
            "priority": "Medium",
            "effectiveness": "Very High (long-term)",
            "implementation_steps": [
                "Assess sediment nutrient content",
                "Obtain environmental permits",
                "Contract specialized dredging equipment",
                "Properly dispose of removed sediments"
            ]
        },
        {
            "title": "Alternative Water Supply",
            "description": "Establish temporary alternative water sources for critical uses while treating the affected water body.",
            "cost": "High ($10,000-30,000)",
            "timeline": "1-4 weeks",
            "priority": "High",
            "effectiveness": "High for continuity",
            "implementation_steps": [
                "Identify alternative water sources",
                "Install temporary distribution systems",
                "Ensure water quality meets standards",
                "Coordinate with local authorities"
            ]
        }
    ],
    
    "High": [
        {
            "title": "Emergency Response Activation",
            "description": "Activate emergency response protocols and notify all relevant authorities and affected communities immediately.",
            "cost": "Low (administrative)",
            "timeline": "Immediate (within 24 hours)",
            "priority": "Critical",
            "effectiveness": "Essential for safety",
            "implementation_steps": [
                "Activate emergency response team",
                "Notify health authorities and media",
                "Issue public health emergency declaration",
                "Coordinate with disaster management agencies"
            ]
        },
        {
            "title": "Complete Water Access Prohibition",
            "description": "Prohibit all human and animal contact with water until algae toxin levels return to safe ranges.",
            "cost": "Medium ($5,000-15,000 for enforcement)",
            "timeline": "Immediate",
            "priority": "Critical",
            "effectiveness": "Essential",
            "implementation_steps": [
                "Install physical barriers at access points",
                "Deploy security personnel if necessary",
                "Issue legal notices and penalties",
                "Provide emergency alternative water sources"
            ]
        },
        {
            "title": "Intensive Chemical Treatment",
            "description": "Apply intensive multi-phase chemical treatment including algaecides, coagulants, and oxidizers.",
            "cost": "Very High ($15,000-40,000)",
            "timeline": "1-3 weeks",
            "priority": "Critical",
            "effectiveness": "High",
            "implementation_steps": [
                "Engage specialized treatment contractors",
                "Implement staged treatment protocol",
                "Monitor treatment progress daily",
                "Test for harmful byproducts"
            ]
        },
        {
            "title": "Water Body Isolation",
            "description": "Physically isolate affected water body to prevent spread of algae and toxins to connected water systems.",
            "cost": "High ($20,000-60,000)",
            "timeline": "1-2 weeks",
            "priority": "High",
            "effectiveness": "High for containment",
            "implementation_steps": [
                "Install temporary barriers or dams",
                "Divert clean water sources",
                "Implement bypass systems",
                "Monitor downstream water quality"
            ]
        },
        {
            "title": "Emergency Water Treatment Plant",
            "description": "Install temporary advanced water treatment facilities to provide safe water for essential needs.",
            "cost": "Very High ($50,000-150,000)",
            "timeline": "2-6 weeks",
            "priority": "High",
            "effectiveness": "High",
            "implementation_steps": [
                "Deploy mobile treatment units",
                "Install advanced filtration systems",
                "Implement multi-barrier treatment approach",
                "Establish quality control laboratory"
            ]
        },
        {
            "title": "Ecosystem Restoration",
            "description": "Begin immediate ecosystem restoration to address fundamental causes of severe algae blooms.",
            "cost": "Very High ($75,000-200,000)",
            "timeline": "6-24 months",
            "priority": "Medium (long-term)",
            "effectiveness": "Very High (sustainable)",
            "implementation_steps": [
                "Conduct comprehensive ecosystem assessment",
                "Develop restoration master plan",
                "Implement habitat restoration projects",
                "Establish long-term monitoring program"
            ]
        }
    ]
}

# Technology-specific mitigation approaches
TECHNOLOGY_BASED_SOLUTIONS = {
    "mechanical": {
        "aeration": {
            "description": "Install fountains, diffused air systems, or mechanical aerators to increase dissolved oxygen",
            "suitable_for": ["Ponds", "Small lakes", "Reservoirs"],
            "cost_range": "$5,000-25,000",
            "maintenance": "Medium",
            "effectiveness": "High for oxygen depletion, Medium for algae"
        },
        "harvesting": {
            "description": "Physical removal of algae biomass using specialized harvesting equipment",
            "suitable_for": ["Large water bodies", "Dense algae mats"],
            "cost_range": "$10,000-50,000",
            "maintenance": "High",
            "effectiveness": "High (immediate), Low (long-term without nutrient control)"
        },
        "ultrasonic": {
            "description": "Use ultrasonic devices to disrupt algae cell walls and prevent bloom formation",
            "suitable_for": ["Small to medium water bodies", "Drinking water reservoirs"],
            "cost_range": "$8,000-30,000",
            "maintenance": "Low",
            "effectiveness": "Medium-High for certain algae species"
        }
    },
    
    "chemical": {
        "algaecides": {
            "description": "Application of EPA-approved algaecides to kill existing algae populations",
            "suitable_for": ["Emergency situations", "Small water bodies"],
            "cost_range": "$2,000-15,000",
            "maintenance": "Low",
            "effectiveness": "High (short-term), requires repeated applications"
        },
        "coagulation": {
            "description": "Use aluminum or iron salts to remove algae and nutrients from water column",
            "suitable_for": ["Turbid water", "High nutrient water bodies"],
            "cost_range": "$5,000-20,000",
            "maintenance": "Medium",
            "effectiveness": "High for algae removal, Medium for nutrient control"
        },
        "oxidation": {
            "description": "Apply hydrogen peroxide or ozone to oxidize algae and organic compounds",
            "suitable_for": ["Small to medium water bodies", "Emergency treatment"],
            "cost_range": "$8,000-35,000",
            "maintenance": "Medium",
            "effectiveness": "High for algae control, degrades quickly"
        }
    },
    
    "biological": {
        "bioaugmentation": {
            "description": "Introduce beneficial bacteria to compete with algae and break down nutrients",
            "suitable_for": ["Eutrophic water bodies", "Long-term treatment"],
            "cost_range": "$3,000-12,000",
            "maintenance": "Low",
            "effectiveness": "Medium-High (sustainable approach)"
        },
        "aquatic_plants": {
            "description": "Establish aquatic plants that absorb nutrients and compete with algae",
            "suitable_for": ["Shallow water bodies", "Constructed wetlands"],
            "cost_range": "$2,000-10,000",
            "maintenance": "Medium",
            "effectiveness": "High (long-term), requires proper species selection"
        },
        "fish_stocking": {
            "description": "Introduce grass carp or other herbivorous fish to control aquatic vegetation",
            "suitable_for": ["Large water bodies", "Established ecosystems"],
            "cost_range": "$1,000-5,000",
            "maintenance": "Low",
            "effectiveness": "Medium, requires careful ecosystem management"
        }
    }
}

# Prevention strategies for different pollution sources
PREVENTION_STRATEGIES = {
    "agricultural": {
        "nutrient_management": {
            "description": "Implement precision agriculture techniques to optimize fertilizer use",
            "actions": [
                "Soil testing to determine actual nutrient needs",
                "Split application of fertilizers",
                "Use slow-release fertilizer formulations",
                "Implement cover cropping"
            ],
            "cost_savings": "15-30% reduction in fertilizer costs",
            "environmental_benefit": "50-70% reduction in nutrient runoff"
        },
        "buffer_strips": {
            "description": "Establish vegetated buffer zones between agricultural areas and water bodies",
            "actions": [
                "Plant native grasses and trees along waterways",
                "Maintain minimum 30-meter buffer width",
                "Prevent cultivation within buffer zones",
                "Regular maintenance of vegetation"
            ],
            "cost_range": "$500-2,000 per acre",
            "effectiveness": "60-90% reduction in sediment and nutrient loading"
        },
        "conservation_tillage": {
            "description": "Reduce soil erosion through minimal tillage practices",
            "actions": [
                "Adopt no-till or minimum till practices",
                "Maintain crop residue on fields",
                "Use cover crops between growing seasons",
                "Implement contour farming on slopes"
            ],
            "cost_impact": "Neutral to positive (reduced fuel costs)",
            "effectiveness": "40-70% reduction in soil erosion"
        }
    },
    
    "urban": {
        "stormwater_management": {
            "description": "Implement green infrastructure to manage urban runoff",
            "actions": [
                "Install rain gardens and bioswales",
                "Implement permeable pavement",
                "Create detention ponds",
                "Upgrade storm drain systems"
            ],
            "cost_range": "$10,000-100,000 per system",
            "effectiveness": "30-80% reduction in pollutant loading"
        },
        "sewage_upgrades": {
            "description": "Upgrade sewage treatment facilities to remove nutrients",
            "actions": [
                "Install biological nutrient removal systems",
                "Upgrade to tertiary treatment",
                "Implement decentralized treatment",
                "Regular maintenance and monitoring"
            ],
            "cost_range": "$50,000-500,000 per plant",
            "effectiveness": "80-95% nutrient removal efficiency"
        },
        "source_control": {
            "description": "Control pollution at the source through regulations and education",
            "actions": [
                "Implement fertilizer ordinances",
                "Promote native landscaping",
                "Educate residents about lawn care",
                "Regulate industrial discharges"
            ],
            "cost_range": "$5,000-20,000 for program development",
            "effectiveness": "20-50% reduction in nutrient inputs"
        }
    },
    
    "industrial": {
        "pretreatment": {
            "description": "Require industrial pretreatment before discharge to municipal systems",
            "actions": [
                "Install industry-specific treatment systems",
                "Regular monitoring and compliance",
                "Implement best management practices",
                "Upgrade to zero liquid discharge where feasible"
            ],
            "cost_range": "$25,000-250,000 per facility",
            "effectiveness": "70-99% pollutant removal depending on treatment level"
        },
        "process_optimization": {
            "description": "Optimize industrial processes to reduce waste generation",
            "actions": [
                "Implement cleaner production techniques",
                "Recycle and reuse process water",
                "Substitute harmful chemicals",
                "Improve operational efficiency"
            ],
            "cost_impact": "Often positive through reduced raw material use",
            "effectiveness": "30-80% reduction in waste generation"
        }
    }
}

# Emergency response protocols
EMERGENCY_PROTOCOLS = {
    "immediate_response": {
        "first_24_hours": [
            "Assess algae bloom extent and severity",
            "Collect water samples for toxin analysis",
            "Issue public health advisory",
            "Restrict water access and use",
            "Notify relevant authorities",
            "Activate emergency response team"
        ],
        "first_week": [
            "Implement emergency treatment measures",
            "Establish alternative water sources",
            "Continue monitoring and assessment",
            "Coordinate media communications",
            "Begin investigation of causes"
        ]
    },
    
    "health_protection": {
        "public_advisories": [
            "Post warning signs at all water access points",
            "Issue media alerts and press releases",
            "Coordinate with health departments",
            "Provide information on symptoms of algae toxin exposure",
            "Establish hotline for public questions"
        ],
        "alternative_water": [
            "Identify safe alternative water sources",
            "Arrange temporary water distribution",
            "Test alternative sources for safety",
            "Coordinate with emergency management agencies"
        ]
    },
    
    "treatment_priorities": {
        "immediate": [
            "Stop algae toxin production",
            "Reduce algae biomass",
            "Protect downstream water bodies",
            "Maintain emergency water supplies"
        ],
        "short_term": [
            "Restore water quality to safe levels",
            "Address immediate pollution sources",
            "Implement temporary prevention measures",
            "Monitor ecosystem recovery"
        ],
        "long_term": [
            "Comprehensive watershed management",
            "Infrastructure improvements",
            "Regulatory framework development",
            "Sustainable prevention strategies"
        ]
    }
}

def get_recommendations_by_risk_level(risk_level: str) -> list:
    """
    Get mitigation strategies for a specific risk level
    
    Args:
        risk_level: Risk level ('Minimal', 'Low', 'Medium', 'High')
        
    Returns:
        List of recommended strategies
    """
    return MITIGATION_STRATEGIES.get(risk_level, MITIGATION_STRATEGIES['Medium'])

def get_technology_recommendations(water_body_type: str, budget_range: str) -> dict:
    """
    Get technology recommendations based on water body characteristics and budget
    
    Args:
        water_body_type: Type of water body ('Pond', 'Lake', 'River', 'Canal')
        budget_range: Budget range ('Low', 'Medium', 'High')
        
    Returns:
        Dictionary of suitable technologies
    """
    suitable_technologies = {}
    
    budget_limits = {
        'Low': 10000,
        'Medium': 50000,
        'High': 200000
    }
    
    max_budget = budget_limits.get(budget_range, 50000)
    
    for category, technologies in TECHNOLOGY_BASED_SOLUTIONS.items():
        for tech_name, tech_info in technologies.items():
            # Parse cost range
            cost_str = tech_info['cost_range'].replace('$', '').replace(',', '')
            cost_parts = cost_str.split('-')
            min_cost = int(cost_parts[0])
            
            # Check if suitable for water body type and within budget
            if (water_body_type in tech_info.get('suitable_for', []) and 
                min_cost <= max_budget):
                suitable_technologies[tech_name] = tech_info
    
    return suitable_technologies

def get_prevention_plan(pollution_sources: list) -> dict:
    """
    Generate prevention plan based on identified pollution sources
    
    Args:
        pollution_sources: List of pollution source types
        
    Returns:
        Dictionary containing prevention strategies
    """
    prevention_plan = {}
    
    for source in pollution_sources:
        if source.lower() in ['agricultural', 'agriculture', 'farm']:
            prevention_plan['agricultural'] = PREVENTION_STRATEGIES['agricultural']
        elif source.lower() in ['urban', 'municipal', 'city']:
            prevention_plan['urban'] = PREVENTION_STRATEGIES['urban']
        elif source.lower() in ['industrial', 'factory', 'industry']:
            prevention_plan['industrial'] = PREVENTION_STRATEGIES['industrial']
    
    return prevention_plan

def estimate_implementation_cost(strategies: list) -> dict:
    """
    Estimate total implementation cost for selected strategies
    
    Args:
        strategies: List of strategy names
        
    Returns:
        Dictionary with cost breakdown
    """
    total_cost = {'min': 0, 'max': 0}
    cost_breakdown = {}
    
    # Search through all risk levels for matching strategies
    for risk_level, strategy_list in MITIGATION_STRATEGIES.items():
        for strategy in strategy_list:
            if strategy['title'] in strategies:
                # Parse cost string
                cost_str = strategy['cost']
                if '$' in cost_str:
                    # Extract numeric values
                    import re
                    numbers = re.findall(r'[\d,]+', cost_str)
                    if len(numbers) >= 2:
                        min_cost = int(numbers[0].replace(',', ''))
                        max_cost = int(numbers[1].replace(',', ''))
                    elif len(numbers) == 1:
                        min_cost = max_cost = int(numbers[0].replace(',', ''))
                    else:
                        min_cost = max_cost = 5000  # Default estimate
                    
                    total_cost['min'] += min_cost
                    total_cost['max'] += max_cost
                    cost_breakdown[strategy['title']] = {'min': min_cost, 'max': max_cost}
    
    return {
        'total_range': f"${total_cost['min']:,} - ${total_cost['max']:,}",
        'breakdown': cost_breakdown,
        'average_estimate': f"${(total_cost['min'] + total_cost['max']) // 2:,}"
    }

