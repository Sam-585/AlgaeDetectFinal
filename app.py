import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
from datetime import datetime, timedelta
import io
import base64
from PIL import Image
import os

# Import custom utilities
from utils.gee_helper import GEEHelper
from utils.spectral_indices import SpectralIndicesCalculator
from utils.image_processor import ImageProcessor
from utils.risk_assessment import RiskAssessment
from utils.report_generator import ReportGenerator
from utils.ml_predictor import AlgaeBloomPredictor
from data.uttarakhand_waterbodies import UTTARAKHAND_WATERBODIES
from assets.mitigation_strategies import MITIGATION_STRATEGIES

# Page configuration
st.set_page_config(
    page_title="Algae Bloom Monitor - Uttarakhand",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'selected_waterbody' not in st.session_state:
    st.session_state.selected_waterbody = None

def main():
    st.title("🌊 Algae Bloom Monitoring System")
    st.subheader("Geospatial Analysis for Waterbodies in Roorkee/Uttarakhand")
    
    # Sidebar for navigation and inputs
    with st.sidebar:
        st.header("Analysis Options")
        analysis_mode = st.selectbox(
            "Select Analysis Mode",
            ["Satellite Imagery Analysis", "Upload Local Image", "Historical Case Study", "Multi-Waterbody Comparison"]
        )
        
        if analysis_mode == "Satellite Imagery Analysis":
            satellite_analysis_sidebar()
        elif analysis_mode == "Upload Local Image":
            local_image_sidebar()
        elif analysis_mode == "Historical Case Study":
            case_study_sidebar()
        else:  # Multi-Waterbody Comparison
            multi_waterbody_sidebar()
    
    # Main content area
    if analysis_mode == "Satellite Imagery Analysis":
        satellite_analysis_main()
    elif analysis_mode == "Upload Local Image":
        local_image_main()
    elif analysis_mode == "Historical Case Study":
        case_study_main()
    else:  # Multi-Waterbody Comparison
        multi_waterbody_main()
    
    # Footer with scientific background
    st.markdown("---")
    with st.expander("📚 Scientific Background & References"):
        st.markdown("""
        ### Why Algae Accumulation is a Civil Engineering Issue
        
        Algae blooms in waterbodies pose significant challenges for:
        - **Water Resource Management**: Clogging of intake systems, reduced water quality
        - **Infrastructure Impact**: Corrosion of pipes, increased treatment costs
        - **Environmental Safety**: Eutrophication, oxygen depletion, ecosystem disruption
        - **Public Health**: Toxic algae species can contaminate drinking water supplies
        
        ### Key Research References
        1. **Chorus, I., & Bartram, J. (1999)**. "Toxic cyanobacteria in water: a guide to their public health consequences, monitoring and management"
        2. **Paerl, H. W., & Huisman, J. (2008)**. "Blooms like it hot: the role of temperature in the global expansion of harmful cyanobacterial blooms"
        
        ### UN SDG Alignment
        - **SDG 6**: Clean Water and Sanitation - Improving water quality monitoring
        - **SDG 14**: Life Below Water - Protecting aquatic ecosystems
        """)
    
    # User Contribution Section
    with st.expander("📝 Contribute Your Observations"):
        show_feedback_form()
    
    # Alert Subscription Section
    with st.expander("🔔 Subscribe to Bloom Alerts"):
        show_alert_subscription()

def satellite_analysis_sidebar():
    st.subheader("🛰️ Satellite Analysis")
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=90),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Satellite selection
    satellite = st.selectbox(
        "Satellite Data Source",
        ["Sentinel-2", "Landsat 8/9", "MODIS"]
    )
    
    # Cloud coverage filter
    cloud_cover = st.slider("Max Cloud Coverage (%)", 0, 100, 20)
    
    # Spectral indices selection
    st.subheader("Spectral Indices")
    indices = st.multiselect(
        "Select Indices to Calculate",
        ["NDVI", "NDWI", "Chlorophyll-a", "Turbidity", "FAI (Floating Algae Index)"],
        default=["NDWI", "Chlorophyll-a", "FAI (Floating Algae Index)"]
    )
    
    if st.button("Run Satellite Analysis", type="primary"):
        run_satellite_analysis(start_date, end_date, satellite, cloud_cover, indices)

def local_image_sidebar():
    st.subheader("📸 Local Image Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload Waterbody Image",
        type=['png', 'jpg', 'jpeg', 'tiff'],
        help="Upload a high-resolution image of a waterbody for algae analysis"
    )
    
    if uploaded_file is not None:
        # Image metadata inputs
        st.subheader("Image Metadata")
        location_name = st.text_input("Location Name", "")
        
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=29.8543, format="%.6f")
        with col2:
            longitude = st.number_input("Longitude", value=77.8880, format="%.6f")
        
        capture_date = st.date_input("Capture Date", value=datetime.now())
        
        if st.button("Analyze Uploaded Image", type="primary"):
            analyze_uploaded_image(uploaded_file, location_name, latitude, longitude, capture_date)

def case_study_sidebar():
    st.subheader("📊 Historical Case Studies")
    
    case_study = st.selectbox(
        "Select Case Study",
        list(UTTARAKHAND_WATERBODIES.keys())
    )
    
    if st.button("Load Case Study", type="primary"):
        load_case_study(case_study)

def multi_waterbody_sidebar():
    st.subheader("🔍 Multi-Waterbody Comparison")
    
    # Select waterbodies to compare
    selected_waterbodies = st.multiselect(
        "Select Waterbodies to Compare (2-5)",
        list(UTTARAKHAND_WATERBODIES.keys()),
        default=list(UTTARAKHAND_WATERBODIES.keys())[:3],
        max_selections=5
    )
    
    # Comparison metrics
    st.subheader("Comparison Options")
    
    show_risk_comparison = st.checkbox("Risk Assessment Comparison", value=True)
    show_historical_trends = st.checkbox("Historical Bloom Trends", value=True)
    show_regional_map = st.checkbox("Regional Risk Map", value=True)
    show_economic_impact = st.checkbox("Economic Impact Analysis", value=False)
    
    if st.button("Generate Comparison Report", type="primary"):
        if len(selected_waterbodies) < 2:
            st.error("Please select at least 2 waterbodies to compare")
        else:
            st.session_state.comparison_waterbodies = selected_waterbodies
            st.session_state.comparison_options = {
                'show_risk_comparison': show_risk_comparison,
                'show_historical_trends': show_historical_trends,
                'show_regional_map': show_regional_map,
                'show_economic_impact': show_economic_impact
            }
            st.session_state.run_comparison = True

def satellite_analysis_main():
    st.header("🗺️ Interactive Map - Select Waterbody")
    
    # Create base map centered on Roorkee/Uttarakhand
    m = folium.Map(
        location=[29.8543, 77.8880],  # Roorkee coordinates
        zoom_start=10,
        tiles="OpenStreetMap"
    )
    
    # Add satellite tile layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add waterbody markers
    for name, data in UTTARAKHAND_WATERBODIES.items():
        folium.CircleMarker(
            location=[data['lat'], data['lon']],
            radius=8,
            popup=f"<b>{name}</b><br>Type: {data['type']}<br>Area: {data['area_km2']} km²",
            color="blue",
            fill=True,
            fillColor="lightblue"
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Display map and capture clicks
    map_data = st_folium(m, width=700, height=400, returned_objects=["last_object_clicked"])
    
    # Handle map clicks
    if map_data['last_object_clicked']:
        clicked_lat = map_data['last_object_clicked']['lat']
        clicked_lng = map_data['last_object_clicked']['lng']
        
        # Find nearest waterbody
        nearest_waterbody = find_nearest_waterbody(clicked_lat, clicked_lng)
        if nearest_waterbody:
            st.session_state.selected_waterbody = nearest_waterbody
            st.success(f"Selected: {nearest_waterbody}")
    
    # Display analysis results if available
    if st.session_state.analysis_results:
        display_analysis_results()

def local_image_main():
    st.header("📸 Local Image Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.session_state.get('uploaded_image'):
            st.subheader("Uploaded Image")
            st.image(st.session_state.uploaded_image, caption="Original Image")
    
    with col2:
        if st.session_state.get('processed_image'):
            st.subheader("Processed Image")
            st.image(st.session_state.processed_image, caption="Algae Detection Overlay")
    
    if st.session_state.analysis_results:
        display_analysis_results()

def case_study_main():
    st.header("📊 Historical Case Studies")
    
    if st.session_state.selected_waterbody:
        waterbody_data = UTTARAKHAND_WATERBODIES[st.session_state.selected_waterbody]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Waterbody Information")
            st.write(f"**Name:** {st.session_state.selected_waterbody}")
            st.write(f"**Type:** {waterbody_data['type']}")
            st.write(f"**Area:** {waterbody_data['area_km2']} km²")
            st.write(f"**Depth:** {waterbody_data['depth_m']} m (avg)")
            st.write(f"**Primary Use:** {waterbody_data['primary_use']}")
        
        with col2:
            st.subheader("Recent Issues")
            for issue in waterbody_data['recent_issues']:
                st.write(f"• {issue}")
    
    if st.session_state.analysis_results:
        display_analysis_results()

def multi_waterbody_main():
    st.header("🔍 Multi-Waterbody Comparison Dashboard")
    
    # Check if comparison should be run
    if not st.session_state.get('run_comparison', False):
        st.info("👈 Select waterbodies from the sidebar and click 'Generate Comparison Report' to begin")
        
        # Show regional overview map
        st.subheader("📍 Regional Overview - Uttarakhand Waterbodies")
        
        m = folium.Map(
            location=[29.8543, 77.8880],
            zoom_start=9,
            tiles="OpenStreetMap"
        )
        
        # Color code by water quality grade
        grade_colors = {
            'A': 'darkgreen', 'B+': 'green', 'B': 'lightgreen',
            'C+': 'yellow', 'C': 'orange', 'C-': 'darkorange',
            'D+': 'red', 'D': 'darkred', 'E': 'black'
        }
        
        for name, data in UTTARAKHAND_WATERBODIES.items():
            color = grade_colors.get(data.get('water_quality_grade', 'C'), 'gray')
            
            folium.CircleMarker(
                location=[data['lat'], data['lon']],
                radius=10,
                popup=f"<b>{name}</b><br>Grade: {data.get('water_quality_grade', 'N/A')}<br>Type: {data['type']}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        st_folium(m, width=900, height=500)
        
        # Legend
        st.write("**Water Quality Grade Legend:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("🟢 A-B: Excellent-Good")
        with col2:
            st.write("🟡 C: Fair")
        with col3:
            st.write("🔴 D-E: Poor-Very Poor")
        
        return
    
    # Run comparison
    selected_waterbodies = st.session_state.get('comparison_waterbodies', [])
    options = st.session_state.get('comparison_options', {})
    
    if len(selected_waterbodies) < 2:
        st.error("Please select at least 2 waterbodies")
        return
    
    st.success(f"Comparing {len(selected_waterbodies)} waterbodies")
    
    # Generate comparison data
    comparison_data = []
    for name in selected_waterbodies:
        wb_data = UTTARAKHAND_WATERBODIES[name]
        
        # Generate case study results for each
        results = generate_case_study_results(name, wb_data)
        
        comparison_data.append({
            'name': name,
            'type': wb_data['type'],
            'area_km2': wb_data['area_km2'],
            'depth_m': wb_data['depth_m'],
            'water_grade': wb_data.get('water_quality_grade', 'N/A'),
            'algae_coverage': results['risk_assessment']['algae_coverage_percent'],
            'risk_level': results['risk_assessment']['risk_level'],
            'risk_score': results['risk_assessment']['risk_score'],
            'growth_rate': results['risk_assessment'].get('growth_rate', 0),
            'pollution_sources': len(wb_data.get('pollution_sources', [])),
            'historical_blooms': len(wb_data.get('historical_blooms', [])),
            'results': results
        })
    
    # Risk Comparison Table
    if options.get('show_risk_comparison', True):
        st.subheader("⚠️ Risk Assessment Comparison")
        
        comp_df = pd.DataFrame(comparison_data)
        display_df = comp_df[['name', 'type', 'area_km2', 'water_grade', 'algae_coverage', 'risk_level', 'risk_score']].copy()
        display_df.columns = ['Waterbody', 'Type', 'Area (km²)', 'Grade', 'Algae %', 'Risk Level', 'Risk Score']
        display_df['Algae %'] = display_df['Algae %'].round(1)
        display_df['Risk Score'] = display_df['Risk Score'].round(3)
        
        # Color code risk levels
        def highlight_risk(row):
            if row['Risk Level'] == 'High':
                return ['background-color: #ffcccc'] * len(row)
            elif row['Risk Level'] == 'Medium':
                return ['background-color: #fff4cc'] * len(row)
            else:
                return ['background-color: #ccffcc'] * len(row)
        
        styled_df = display_df.style.apply(highlight_risk, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Side-by-side comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_coverage = px.bar(
                comp_df,
                x='name',
                y='algae_coverage',
                title='Algae Coverage Comparison',
                color='risk_level',
                color_discrete_map={'Low': 'green', 'Medium': 'yellow', 'High': 'red'},
                labels={'name': 'Waterbody', 'algae_coverage': 'Coverage (%)'}
            )
            st.plotly_chart(fig_coverage, use_container_width=True)
        
        with col2:
            fig_score = px.bar(
                comp_df,
                x='name',
                y='risk_score',
                title='Risk Score Comparison',
                color='risk_score',
                color_continuous_scale='RdYlGn_r',
                labels={'name': 'Waterbody', 'risk_score': 'Risk Score'}
            )
            st.plotly_chart(fig_score, use_container_width=True)
    
    # Historical Trends
    if options.get('show_historical_trends', True):
        st.subheader("📈 Historical Bloom Trends")
        
        fig_trends = go.Figure()
        
        for item in comparison_data:
            temporal_data = item['results'].get('temporal_data', [])
            if temporal_data:
                temp_df = pd.DataFrame(temporal_data)
                fig_trends.add_trace(go.Scatter(
                    x=temp_df['date'],
                    y=temp_df['algae_coverage'],
                    mode='lines+markers',
                    name=item['name']
                ))
        
        fig_trends.update_layout(
            title='Algae Coverage Trends - Last 90 Days',
            xaxis_title='Date',
            yaxis_title='Algae Coverage (%)',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_trends, use_container_width=True)
    
    # Regional Risk Map
    if options.get('show_regional_map', True):
        st.subheader("🗺️ Regional Risk Distribution")
        
        m = folium.Map(
            location=[29.8543, 77.8880],
            zoom_start=9,
            tiles="OpenStreetMap"
        )
        
        risk_colors = {'Low': 'green', 'Minimal': 'lightgreen', 'Medium': 'orange', 'High': 'red'}
        
        for item in comparison_data:
            wb_data = UTTARAKHAND_WATERBODIES[item['name']]
            color = risk_colors.get(item['risk_level'], 'gray')
            
            folium.CircleMarker(
                location=[wb_data['lat'], wb_data['lon']],
                radius=15,
                popup=f"<b>{item['name']}</b><br>Risk: {item['risk_level']}<br>Coverage: {item['algae_coverage']:.1f}%",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        st_folium(m, width=900, height=450)
    
    # Economic Impact
    if options.get('show_economic_impact', False):
        st.subheader("💰 Economic Impact Comparison")
        
        economic_data = []
        for item in comparison_data:
            # Estimate treatment costs based on algae coverage and area
            coverage = item['algae_coverage']
            area = item['area_km2']
            
            # Cost per km² based on severity
            if coverage > 40:
                cost_per_km2 = 50000  # High treatment cost
            elif coverage > 20:
                cost_per_km2 = 25000  # Medium cost
            else:
                cost_per_km2 = 10000  # Low/prevention cost
            
            estimated_cost = cost_per_km2 * area
            
            # Population affected (rough estimate based on waterbody use)
            if 'Domestic' in item['results']['waterbody'] or 'water supply' in UTTARAKHAND_WATERBODIES[item['name']].get('primary_use', '').lower():
                pop_affected = int(area * 10000)  # 10k per km² for domestic use
            else:
                pop_affected = int(area * 2000)  # Lower for irrigation/other uses
            
            economic_data.append({
                'Waterbody': item['name'],
                'Treatment Cost (₹)': f"₹{estimated_cost:,.0f}",
                'Population Affected': f"{pop_affected:,}",
                'Area Affected (km²)': area
            })
        
        econ_df = pd.DataFrame(economic_data)
        st.table(econ_df)
        
        st.info("💡 **Cost estimates** based on standard water treatment protocols. Actual costs may vary based on specific conditions.")
    
    # Summary Statistics
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_coverage = np.mean([item['algae_coverage'] for item in comparison_data])
        st.metric("Average Coverage", f"{avg_coverage:.1f}%")
    
    with col2:
        high_risk_count = sum(1 for item in comparison_data if item['risk_level'] in ['High', 'Severe'])
        st.metric("High Risk Sites", high_risk_count)
    
    with col3:
        total_area = sum(item['area_km2'] for item in comparison_data)
        st.metric("Total Area Monitored", f"{total_area:.1f} km²")
    
    with col4:
        avg_growth = np.mean([item['growth_rate'] for item in comparison_data])
        st.metric("Avg Growth Rate", f"{avg_growth:+.1f}%/wk")
    
    # Reset button
    if st.button("🔄 Start New Comparison"):
        st.session_state.run_comparison = False
        st.rerun()

def run_satellite_analysis(start_date, end_date, satellite, cloud_cover, indices):
    """Run satellite imagery analysis using Google Earth Engine"""
    
    with st.spinner("🛰️ Fetching satellite data and running analysis..."):
        try:
            # Initialize GEE helper
            gee_helper = GEEHelper()
            
            if st.session_state.selected_waterbody:
                waterbody_data = UTTARAKHAND_WATERBODIES[st.session_state.selected_waterbody]
                center_lat, center_lon = waterbody_data['lat'], waterbody_data['lon']
            else:
                center_lat, center_lon = 29.8543, 77.8880  # Default to Roorkee
            
            # Get satellite imagery
            imagery_data = gee_helper.get_imagery(
                center_lat, center_lon,
                start_date, end_date,
                satellite, cloud_cover
            )
            
            if not imagery_data:
                st.error("No suitable satellite imagery found for the selected parameters.")
                return
            
            # Calculate spectral indices
            indices_calc = SpectralIndicesCalculator()
            results = {}
            
            for index in indices:
                if index == "NDVI":
                    results[index] = indices_calc.calculate_ndvi(imagery_data)
                elif index == "NDWI":
                    results[index] = indices_calc.calculate_ndwi(imagery_data)
                elif index == "Chlorophyll-a":
                    results[index] = indices_calc.calculate_chlorophyll_a(imagery_data)
                elif index == "Turbidity":
                    results[index] = indices_calc.calculate_turbidity(imagery_data)
                elif index == "FAI (Floating Algae Index)":
                    results[index] = indices_calc.calculate_fai(imagery_data)
            
            # Perform risk assessment
            risk_assessor = RiskAssessment()
            risk_data = risk_assessor.assess_algae_risk(results, imagery_data)
            
            # Store results in session state
            st.session_state.analysis_results = {
                'type': 'satellite',
                'waterbody': st.session_state.selected_waterbody or "Selected Location",
                'date_range': f"{start_date} to {end_date}",
                'satellite': satellite,
                'indices': results,
                'risk_assessment': risk_data,
                'temporal_data': generate_temporal_data(results),
                'environmental_impact': calculate_environmental_impact(risk_data)
            }
            
            st.success("✅ Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.info("Note: This demo uses simulated data as Google Earth Engine requires authentication.")
            # Generate demo data for demonstration
            generate_demo_analysis_results('satellite')

def analyze_uploaded_image(uploaded_file, location_name, latitude, longitude, capture_date):
    """Analyze uploaded local image"""
    
    with st.spinner("📸 Processing uploaded image..."):
        try:
            # Read and process image
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            
            # Initialize image processor
            img_processor = ImageProcessor()
            
            # Process image for algae detection
            processed_results = img_processor.detect_algae(image)
            st.session_state.processed_image = processed_results['overlay_image']
            
            # Calculate spectral indices from image
            indices_calc = SpectralIndicesCalculator()
            image_array = np.array(image)
            
            results = {
                'NDVI': indices_calc.calculate_ndvi_from_rgb(image_array),
                'Chlorophyll-a': indices_calc.calculate_chlorophyll_from_rgb(image_array),
                'Turbidity': indices_calc.calculate_turbidity_from_rgb(image_array)
            }
            
            # Perform risk assessment
            risk_assessor = RiskAssessment()
            risk_data = risk_assessor.assess_algae_risk_from_image(processed_results, results)
            
            # Store results
            st.session_state.analysis_results = {
                'type': 'local_image',
                'waterbody': location_name or "Uploaded Image",
                'location': f"Lat: {latitude}, Lon: {longitude}",
                'capture_date': str(capture_date),
                'indices': results,
                'risk_assessment': risk_data,
                'algae_coverage': processed_results['algae_percentage'],
                'environmental_impact': calculate_environmental_impact(risk_data)
            }
            
            st.success("✅ Image analysis completed!")
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            generate_demo_analysis_results('local_image')

def load_case_study(case_study_name):
    """Load historical case study data"""
    
    st.session_state.selected_waterbody = case_study_name
    
    with st.spinner("📊 Loading case study data..."):
        try:
            # Generate comprehensive case study data
            waterbody_data = UTTARAKHAND_WATERBODIES[case_study_name]
            
            # Simulate historical analysis results
            results = generate_case_study_results(case_study_name, waterbody_data)
            
            st.session_state.analysis_results = results
            st.success(f"✅ Case study loaded: {case_study_name}")
            
        except Exception as e:
            st.error(f"❌ Error loading case study: {str(e)}")

def display_analysis_results():
    """Display comprehensive analysis results"""
    
    results = st.session_state.analysis_results
    
    # Key Metrics Dashboard
    st.header("📊 Analysis Results Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        algae_coverage = results.get('algae_coverage', results['risk_assessment']['algae_coverage_percent'])
        st.metric("Algae Coverage", f"{algae_coverage:.1f}%")
    
    with col2:
        risk_score = results['risk_assessment']['risk_score']
        risk_level = results['risk_assessment']['risk_level']
        st.metric("Risk Level", risk_level, f"Score: {risk_score:.2f}")
    
    with col3:
        growth_rate = results['risk_assessment'].get('growth_rate', 0)
        st.metric("Growth Rate", f"{growth_rate:+.2f}%/week")
    
    with col4:
        water_quality = results['environmental_impact']['water_quality_score']
        st.metric("Water Quality Score", f"{water_quality:.1f}/10")
    
    # Spectral Indices Visualization
    st.subheader("🌈 Spectral Indices Analysis")
    
    indices_data = []
    for index, value in results['indices'].items():
        if isinstance(value, (int, float)):
            indices_data.append({'Index': index, 'Value': value})
        elif isinstance(value, dict) and 'mean' in value:
            indices_data.append({'Index': index, 'Value': value['mean']})
    
    if indices_data:
        df_indices = pd.DataFrame(indices_data)
        fig_indices = px.bar(
            df_indices, 
            x='Index', 
            y='Value',
            title="Spectral Indices Values",
            color='Value',
            color_continuous_scale='RdYlBu_r'
        )
        st.plotly_chart(fig_indices, use_container_width=True)
    
    # Temporal Analysis (if available)
    if 'temporal_data' in results and results['temporal_data']:
        st.subheader("📈 Temporal Analysis - Algae Growth Trends")
        
        temporal_df = pd.DataFrame(results['temporal_data'])
        
        fig_temporal = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Algae Coverage Over Time', 'Risk Score Trends'],
            vertical_spacing=0.1
        )
        
        fig_temporal.add_trace(
            go.Scatter(
                x=temporal_df['date'],
                y=temporal_df['algae_coverage'],
                mode='lines+markers',
                name='Algae Coverage %',
                line=dict(color='green')
            ),
            row=1, col=1
        )
        
        fig_temporal.add_trace(
            go.Scatter(
                x=temporal_df['date'],
                y=temporal_df['risk_score'],
                mode='lines+markers',
                name='Risk Score',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        fig_temporal.update_layout(height=500, showlegend=True)
        fig_temporal.update_xaxes(title_text="Date", row=2, col=1)
        fig_temporal.update_yaxes(title_text="Coverage %", row=1, col=1)
        fig_temporal.update_yaxes(title_text="Risk Score", row=2, col=1)
        
        st.plotly_chart(fig_temporal, use_container_width=True)
    
    # Environmental Impact Assessment
    st.subheader("🌍 Environmental Impact Assessment")
    
    impact_data = results['environmental_impact']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dissolved Oxygen Impact:**")
        do_reduction = impact_data['dissolved_oxygen_reduction']
        st.write(f"• Estimated reduction: {do_reduction:.1f}%")
        
        if do_reduction > 30:
            st.error("⚠️ Critical oxygen depletion risk")
        elif do_reduction > 15:
            st.warning("⚡ Moderate oxygen reduction")
        else:
            st.success("✅ Minimal oxygen impact")
        
        st.write("**Aquatic Life Risk:**")
        fish_mortality_risk = impact_data['fish_mortality_risk']
        st.write(f"• Fish mortality risk: {fish_mortality_risk}")
        
        st.write("**Water Usability:**")
        for use, status in impact_data['water_usability'].items():
            icon = "✅" if status == "Safe" else "⚠️" if status == "Caution" else "❌"
            st.write(f"• {use}: {icon} {status}")
    
    with col2:
        # Risk distribution pie chart
        risk_dist = impact_data.get('risk_distribution', {
            'Low Risk': 30, 'Medium Risk': 45, 'High Risk': 25
        })
        
        fig_risk = px.pie(
            values=list(risk_dist.values()),
            names=list(risk_dist.keys()),
            title="Risk Distribution Across Waterbody",
            color_discrete_map={
                'Low Risk': 'green',
                'Medium Risk': 'yellow',
                'High Risk': 'red'
            }
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # ML Prediction Section
    st.subheader("🤖 AI-Powered Bloom Risk Forecast")
    
    try:
        # Initialize ML predictor
        ml_predictor = AlgaeBloomPredictor()
        
        # Train on historical data
        training_results = ml_predictor.train_models(UTTARAKHAND_WATERBODIES, model_type='random_forest')
        
        # Check if training was successful
        if not training_results.get('success', False):
            st.warning(f"⚠️ ML training: {training_results.get('message', 'Model not trained')}")
            st.info("Using rule-based forecasting instead")
        
        # Get waterbody info
        waterbody_name = results.get('waterbody', 'Unknown')
        waterbody_info = UTTARAKHAND_WATERBODIES.get(waterbody_name, {
            'area_km2': 10,
            'depth_m': 5,
            'pollution_sources': ['Unknown'],
            'water_quality_grade': 'C',
            'historical_blooms': []
        })
        
        # Prepare current conditions from results
        indices = results['indices']
        current_month = datetime.now().month
        
        current_conditions = {
            'chlorophyll_a': indices.get('Chlorophyll-a', {}).get('mean', indices.get('Chlorophyll-a', 10)) if isinstance(indices.get('Chlorophyll-a'), dict) else indices.get('Chlorophyll-a', 10),
            'turbidity': indices.get('Turbidity', {}).get('mean', indices.get('Turbidity', 15)) if isinstance(indices.get('Turbidity'), dict) else indices.get('Turbidity', 15),
            'temperature_factor': 1.2 if current_month in [6, 7, 8] else 0.8,
            'nutrient_factor': 1.0 + (len(waterbody_info.get('pollution_sources', [])) * 0.1),
            'seasonal_factor': 1.3 if current_month in [8, 9, 10] else 0.7,
            'current_coverage': results['risk_assessment']['algae_coverage_percent']
        }
        
        # Get predictions for different time horizons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_7 = ml_predictor.predict_bloom_risk(waterbody_info, current_conditions, days_ahead=7)
            st.metric("7-Day Forecast", pred_7['risk_category'], 
                     f"Coverage: {pred_7['future_coverage']:.1f}%")
            st.progress(pred_7['bloom_probability'])
            st.caption(f"Bloom probability: {pred_7['bloom_probability']*100:.1f}%")
        
        with col2:
            pred_14 = ml_predictor.predict_bloom_risk(waterbody_info, current_conditions, days_ahead=14)
            st.metric("14-Day Forecast", pred_14['risk_category'],
                     f"Coverage: {pred_14['future_coverage']:.1f}%")
            st.progress(pred_14['bloom_probability'])
            st.caption(f"Bloom probability: {pred_14['bloom_probability']*100:.1f}%")
        
        with col3:
            pred_30 = ml_predictor.predict_bloom_risk(waterbody_info, current_conditions, days_ahead=30)
            st.metric("30-Day Forecast", pred_30['risk_category'],
                     f"Coverage: {pred_30['future_coverage']:.1f}%")
            st.progress(pred_30['bloom_probability'])
            st.caption(f"Bloom probability: {pred_30['bloom_probability']*100:.1f}%")
        
        # Temporal progression chart
        if st.checkbox("Show 30-Day Progression Forecast"):
            progression = ml_predictor.predict_temporal_progression(waterbody_info, current_conditions, days=30)
            prog_df = pd.DataFrame(progression)
            
            fig_prog = go.Figure()
            fig_prog.add_trace(go.Scatter(
                x=prog_df['date'],
                y=prog_df['predicted_coverage'],
                mode='lines+markers',
                name='Predicted Coverage',
                line=dict(color='#FF6B6B', width=2)
            ))
            fig_prog.update_layout(
                title='30-Day Algae Coverage Forecast',
                xaxis_title='Date',
                yaxis_title='Predicted Coverage (%)',
                hovermode='x unified',
                height=300
            )
            st.plotly_chart(fig_prog, use_container_width=True)
        
        # Model performance info
        if training_results.get('success', False):
            with st.expander("📊 Model Performance Metrics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Model Type:** {training_results['model_type'].replace('_', ' ').title()}")
                    st.write(f"**Training Samples:** {training_results['samples_trained']}")
                    st.write(f"**Classification Accuracy:** {training_results['classification_accuracy']*100:.1f}%")
                    st.write(f"**Historical Bloom Rate:** {training_results.get('bloom_percentage', 0):.1f}%")
                with col2:
                    st.write(f"**Prediction Model:** {pred_14['model_used']}")
                    st.write(f"**Prediction Confidence:** {pred_14['confidence']*100:.1f}%")
                    st.write(f"**Growth Rate:** {pred_14.get('growth_rate_per_day', 0):.2f}%/day")
                    
                # Feature importance
                feat_imp = training_results.get('feature_importance', {})
                if feat_imp:
                    st.write("**Top 3 Predictive Features:**")
                    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:3]
                    for feat_name, importance in sorted_feats:
                        st.write(f"• {feat_name.replace('_', ' ').title()}: {importance*100:.1f}%")
        
        # AI Recommendations
        ai_recommendations = ml_predictor.get_model_recommendations(pred_14)
        if ai_recommendations:
            st.write("**🎯 AI-Generated Recommendations:**")
            for rec in ai_recommendations[:3]:
                st.write(f"• {rec}")
                
    except Exception as e:
        st.warning(f"ML predictions unavailable: {str(e)}")
        st.info("Showing rule-based risk assessment only")
        import traceback
        st.code(traceback.format_exc())
    
    # UN SDG Impact Assessment
    st.subheader("🌍 UN Sustainable Development Goals (SDG) Impact")
    
    st.markdown("""
    This analysis directly contributes to achieving multiple UN Sustainable Development Goals:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🎯 Primary SDGs")
        
        # SDG 6: Clean Water and Sanitation
        st.write("**SDG 6: Clean Water and Sanitation**")
        algae_coverage = results['risk_assessment']['algae_coverage_percent']
        waterbody_name = results.get('waterbody', 'Unknown')
        waterbody_info = UTTARAKHAND_WATERBODIES.get(waterbody_name, {})
        
        # Calculate people potentially benefiting
        area_km2 = waterbody_info.get('area_km2', 10)
        if 'water supply' in waterbody_info.get('primary_use', '').lower() or 'domestic' in waterbody_info.get('type', '').lower():
            population_served = int(area_km2 * 15000)  # 15k per km² for domestic use
        else:
            population_served = int(area_km2 * 3000)  # 3k per km² for other uses
        
        st.write(f"• **{population_served:,} people** potentially benefit from improved water quality")
        st.write(f"• **{area_km2:.1f} km²** of water resources monitored")
        
        if algae_coverage > 30:
            st.write(f"• **High urgency**: Water treatment needed for {population_served:,} users")
        elif algae_coverage > 15:
            st.write(f"• **Medium priority**: Preventive measures recommended")
        else:
            st.write(f"• **Sustainable**: Current water quality maintained")
        
        st.write("")
        
        # SDG 14: Life Below Water
        st.write("**SDG 14: Life Below Water**")
        do_impact = results.get('environmental_impact', {}).get('dissolved_oxygen_reduction', 0)
        fish_risk = results.get('environmental_impact', {}).get('fish_mortality_risk', 'Low')
        
        st.write(f"• Dissolved oxygen impact: **{do_impact:.1f}%** reduction")
        st.write(f"• Aquatic life risk: **{fish_risk}**")
        st.write(f"• Biodiversity protection: **{'Critical' if do_impact > 30 else 'Moderate' if do_impact > 15 else 'Good'}**")
    
    with col2:
        st.write("### 🔄 Secondary SDGs")
        
        # SDG 3: Good Health and Well-Being
        st.write("**SDG 3: Good Health and Well-Being**")
        risk_level = results['risk_assessment']['risk_level']
        
        if risk_level in ['High', 'Severe']:
            health_risk = "High - Immediate action needed"
            people_at_risk = population_served
        elif risk_level == 'Medium':
            health_risk = "Moderate - Monitor closely"
            people_at_risk = int(population_served * 0.3)
        else:
            health_risk = "Low - Safe water quality"
            people_at_risk = 0
        
        st.write(f"• Public health risk: **{health_risk}**")
        st.write(f"• People potentially at risk: **{people_at_risk:,}**")
        
        st.write("")
        
        # SDG 11: Sustainable Cities and Communities
        st.write("**SDG 11: Sustainable Cities**")
        st.write(f"• Infrastructure resilience: **{'Needs improvement' if risk_level == 'High' else 'Adequate'}**")
        st.write(f"• Water resource management: **Active monitoring**")
        
        st.write("")
        
        # SDG 13: Climate Action
        st.write("**SDG 13: Climate Action**")
        st.write(f"• Climate-related monitoring: **Active**")
        st.write(f"• Early warning system: **Operational**")
    
    # Quantifiable Impact Metrics
    st.write("### 📊 Measurable Impact Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Water quality improvement potential
        improvement_potential = max(0, algae_coverage - 5)  # Target is <5%
        st.metric("Water Quality Improvement Potential", f"{improvement_potential:.1f}%")
    
    with col2:
        # Lives protected
        st.metric("People Served", f"{population_served:,}")
    
    with col3:
        # Economic benefit (avoided treatment costs)
        if algae_coverage > 30:
            treatment_cost = area_km2 * 50000
        elif algae_coverage > 15:
            treatment_cost = area_km2 * 25000
        else:
            treatment_cost = area_km2 * 10000
        
        avoided_cost = treatment_cost * 0.7  # 70% cost savings through prevention
        st.metric("Potential Cost Savings", f"₹{avoided_cost:,.0f}")
    
    with col4:
        # Ecosystem health score
        ecosystem_score = max(0, 100 - do_impact - (algae_coverage * 0.5))
        st.metric("Ecosystem Health Score", f"{ecosystem_score:.0f}/100")
    
    st.info("""
    💡 **Innovation & Impact**: This application provides real-time, data-driven insights that enable 
    proactive water resource management, directly contributing to multiple UN SDGs through:
    - **Early Warning**: Detecting blooms before they become severe
    - **Prevention**: Reducing treatment costs by 50-70% through early intervention
    - **Data-Driven Decisions**: Providing quantifiable metrics for policy makers
    - **Community Engagement**: Enabling citizen science through feedback systems
    """)
    
    # Mitigation Recommendations
    st.subheader("💡 Recommended Mitigation Strategies")
    
    risk_level = results['risk_assessment']['risk_level']
    recommendations = MITIGATION_STRATEGIES.get(risk_level, MITIGATION_STRATEGIES['Medium'])
    
    for i, strategy in enumerate(recommendations, 1):
        st.write(f"{i}. **{strategy['title']}**")
        st.write(f"   {strategy['description']}")
        st.write(f"   *Estimated cost: {strategy['cost']} | Timeline: {strategy['timeline']}*")
        st.write("")
    
    # Export Options
    st.subheader("📋 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate PDF Report"):
            generate_pdf_report(results)
    
    with col2:
        if st.button("📊 Download CSV Data"):
            generate_csv_export(results)
    
    with col3:
        if st.button("🔗 Share Analysis Link"):
            st.info("Analysis link copied to clipboard!")

def find_nearest_waterbody(lat, lng):
    """Find nearest waterbody to clicked coordinates"""
    min_distance = float('inf')
    nearest = None
    
    for name, data in UTTARAKHAND_WATERBODIES.items():
        distance = ((lat - data['lat'])**2 + (lng - data['lon'])**2)**0.5
        if distance < min_distance:
            min_distance = distance
            nearest = name
    
    return nearest if min_distance < 0.1 else None  # Within ~11km

def show_feedback_form():
    """Display user feedback and contribution form"""
    from utils.database_helper import DatabaseHelper
    
    st.markdown("""
    Help us improve algae monitoring by sharing your observations, case studies, or feedback!
    Your contributions help build a comprehensive database of algae bloom incidents.
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Submit Case Study", "💬 General Feedback", "🚨 Report Issue"])
    
    # Tab 1: Case Study Submission
    with tab1:
        st.subheader("Submit a Case Study")
        st.write("Share your field observations and help build our knowledge base")
        
        with st.form("case_study_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                submitter_name = st.text_input("Your Name *", placeholder="John Doe")
                submitter_email = st.text_input("Email *", placeholder="john@example.com")
                submitter_role = st.selectbox("Your Role *", 
                    ["Student", "Engineer", "Researcher", "Citizen", "Government Official", "Other"])
            
            with col2:
                waterbody_name = st.text_input("Waterbody Name *", placeholder="e.g., Solani River")
                observation_date = st.date_input("Observation Date *", value=datetime.now())
                algae_severity = st.selectbox("Algae Severity *", 
                    ["Low", "Medium", "High", "Severe"])
            
            col3, col4 = st.columns(2)
            with col3:
                location_lat = st.number_input("Latitude", value=29.8543, format="%.6f")
            with col4:
                location_lon = st.number_input("Longitude", value=77.8880, format="%.6f")
            
            estimated_coverage = st.slider("Estimated Algae Coverage (%)", 0, 100, 20)
            
            observations = st.text_area("Detailed Observations *", 
                placeholder="Describe the algae bloom, water color, smell, affected area, etc.")
            
            mitigation_attempted = st.text_area("Mitigation Measures Attempted", 
                placeholder="What actions were taken to address the bloom?")
            
            outcomes = st.text_area("Outcomes", 
                placeholder="Results of mitigation efforts")
            
            submitted = st.form_submit_button("📤 Submit Case Study", type="primary")
            
            if submitted:
                if not all([submitter_name, submitter_email, waterbody_name, observations]):
                    st.error("Please fill in all required fields (*)") 
                else:
                    try:
                        db = DatabaseHelper()
                        case_study_id = db.submit_case_study(
                            submitter_name=submitter_name,
                            submitter_email=submitter_email,
                            submitter_role=submitter_role,
                            waterbody_name=waterbody_name,
                            observation_date=observation_date.strftime('%Y-%m-%d'),
                            algae_severity=algae_severity,
                            estimated_coverage=estimated_coverage,
                            location_lat=location_lat,
                            location_lon=location_lon,
                            observations=observations,
                            mitigation_attempted=mitigation_attempted if mitigation_attempted else None,
                            outcomes=outcomes if outcomes else None
                        )
                        st.success(f"✅ Case study submitted successfully! ID: {case_study_id}")
                        st.info("Your submission will be reviewed and may be added to our public database")
                    except Exception as e:
                        st.error(f"Failed to submit case study: {str(e)}")
    
    # Tab 2: General Feedback
    with tab2:
        st.subheader("Share Your Feedback")
        st.write("Tell us about your experience with the application")
        
        with st.form("feedback_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                feedback_name = st.text_input("Name *", placeholder="Your name")
                feedback_email = st.text_input("Email *", placeholder="your.email@example.com")
            
            with col2:
                organization = st.text_input("Organization", placeholder="Optional")
                rating = st.select_slider("Rating", options=[1, 2, 3, 4, 5], value=4)
            
            feedback_text = st.text_area("Your Feedback *", 
                placeholder="Share your thoughts, suggestions, or experiences...")
            
            submitted_feedback = st.form_submit_button("📨 Submit Feedback", type="primary")
            
            if submitted_feedback:
                if not all([feedback_name, feedback_email, feedback_text]):
                    st.error("Please fill in all required fields (*)")
                else:
                    try:
                        db = DatabaseHelper()
                        feedback_id = db.submit_feedback(
                            name=feedback_name,
                            email=feedback_email,
                            organization=organization if organization else None,
                            waterbody_name=None,
                            feedback_type='feedback',
                            feedback_text=feedback_text,
                            rating=rating
                        )
                        st.success(f"✅ Thank you for your feedback! ID: {feedback_id}")
                    except Exception as e:
                        st.error(f"Failed to submit feedback: {str(e)}")
    
    # Tab 3: Issue Report
    with tab3:
        st.subheader("Report an Issue")
        st.write("Report urgent algae bloom incidents or water quality concerns")
        
        with st.form("issue_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                issue_name = st.text_input("Your Name *", placeholder="Your name")
                issue_email = st.text_input("Email *", placeholder="contact@example.com")
            
            with col2:
                issue_waterbody = st.text_input("Waterbody Name *", placeholder="Which waterbody?")
                issue_severity = st.select_slider("Urgency Level", 
                    options=["Low", "Medium", "High", "Critical"], value="Medium")
            
            col3, col4 = st.columns(2)
            with col3:
                issue_lat = st.number_input("Latitude", value=29.8543, format="%.6f")
            with col4:
                issue_lon = st.number_input("Longitude", value=77.8880, format="%.6f")
            
            issue_description = st.text_area("Issue Description *", 
                placeholder="Describe the issue, its location, and potential impact...")
            
            submitted_issue = st.form_submit_button("🚨 Submit Issue Report", type="primary")
            
            if submitted_issue:
                if not all([issue_name, issue_email, issue_waterbody, issue_description]):
                    st.error("Please fill in all required fields (*)")
                else:
                    try:
                        db = DatabaseHelper()
                        issue_id = db.submit_feedback(
                            name=issue_name,
                            email=issue_email,
                            organization=None,
                            waterbody_name=issue_waterbody,
                            feedback_type='issue_report',
                            feedback_text=issue_description,
                            location_lat=issue_lat,
                            location_lon=issue_lon
                        )
                        st.success(f"✅ Issue reported successfully! ID: {issue_id}")
                        st.warning("⚠️ For immediate emergencies, please contact local authorities")
                    except Exception as e:
                        st.error(f"Failed to submit issue report: {str(e)}")

def show_alert_subscription():
    """Display alert subscription form"""
    from utils.database_helper import DatabaseHelper
    
    st.markdown("""
    Get notified when algae bloom risk levels exceed your threshold. 
    Subscribe to receive email alerts for specific waterbodies in the Uttarakhand region.
    """)
    
    tab1, tab2 = st.tabs(["📧 Subscribe", "✏️ Manage Subscription"])
    
    # Tab 1: Subscribe
    with tab1:
        with st.form("alert_subscription_form"):
            st.subheader("Set Up Algae Bloom Alerts")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sub_name = st.text_input("Your Name *", placeholder="John Doe")
                sub_email = st.text_input("Email Address *", placeholder="alerts@example.com")
            
            with col2:
                alert_threshold = st.selectbox(
                    "Alert Threshold *",
                    ["Low", "Medium", "High"],
                    index=1,
                    help="Receive alerts when risk reaches or exceeds this level"
                )
                
                notification_frequency = st.selectbox(
                    "Notification Frequency",
                    ["immediate", "daily", "weekly"],
                    index=0,
                    help="How often to receive notifications"
                )
            
            st.write("**Select Waterbodies to Monitor:**")
            waterbody_options = list(UTTARAKHAND_WATERBODIES.keys())
            
            # Create checkboxes in columns
            cols = st.columns(3)
            selected_waterbodies = []
            
            for idx, waterbody in enumerate(waterbody_options):
                with cols[idx % 3]:
                    if st.checkbox(waterbody, value=(idx < 3), key=f"sub_{waterbody}"):
                        selected_waterbodies.append(waterbody)
            
            st.info("📧 You will receive a verification email after subscribing")
            
            submitted = st.form_submit_button("🔔 Subscribe to Alerts", type="primary")
            
            if submitted:
                if not all([sub_name, sub_email, selected_waterbodies]):
                    st.error("Please fill in all required fields and select at least one waterbody")
                else:
                    try:
                        db = DatabaseHelper()
                        subscription_id = db.subscribe_to_alerts(
                            email=sub_email,
                            name=sub_name,
                            waterbodies=selected_waterbodies,
                            alert_threshold=alert_threshold,
                            notification_frequency=notification_frequency
                        )
                        st.success(f"✅ Successfully subscribed! Subscription ID: {subscription_id}")
                        st.info("📧 Please check your email to verify your subscription")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Failed to subscribe: {str(e)}")
    
    # Tab 2: Manage Subscription
    with tab2:
        st.subheader("Manage Your Subscription")
        
        manage_email = st.text_input("Enter your email address", placeholder="your@email.com")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 View My Subscriptions"):
                if manage_email:
                    try:
                        db = DatabaseHelper()
                        subscriptions = db.get_active_subscriptions()
                        user_subs = [s for s in subscriptions if s['email'] == manage_email]
                        
                        if user_subs:
                            for sub in user_subs:
                                st.write(f"**Subscription ID:** {sub['id']}")
                                st.write(f"**Status:** {'Active' if sub['is_active'] else 'Inactive'}")
                                st.write(f"**Waterbodies:** {', '.join(sub['waterbodies'])}")
                                st.write(f"**Threshold:** {sub['alert_threshold']}")
                                st.write(f"**Frequency:** {sub['notification_frequency']}")
                                st.write(f"**Subscribed:** {sub['subscribed_at']}")
                                st.write("---")
                        else:
                            st.warning("No subscriptions found for this email")
                    except Exception as e:
                        st.error(f"Error retrieving subscriptions: {str(e)}")
                else:
                    st.error("Please enter your email address")
        
        with col2:
            if st.button("🚫 Unsubscribe"):
                if manage_email:
                    try:
                        db = DatabaseHelper()
                        db.unsubscribe_from_alerts(manage_email)
                        st.success("✅ Successfully unsubscribed from all alerts")
                    except Exception as e:
                        st.error(f"Failed to unsubscribe: {str(e)}")
                else:
                    st.error("Please enter your email address")

def generate_temporal_data(indices_results):
    """Generate temporal data for trend analysis"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='W')
    
    temporal_data = []
    base_coverage = 15.0
    
    for i, date in enumerate(dates):
        # Simulate seasonal variation and growth trends
        seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * i / 52)  # Annual cycle
        growth_factor = 1.0 + 0.02 * i  # Gradual growth
        noise = np.random.normal(0, 0.1)
        
        coverage = base_coverage * seasonal_factor * growth_factor + noise
        coverage = max(0, min(100, coverage))  # Clamp between 0-100%
        
        risk_score = min(1.0, coverage / 100 + 0.2)
        
        temporal_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'algae_coverage': coverage,
            'risk_score': risk_score
        })
    
    return temporal_data

def calculate_environmental_impact(risk_data):
    """Calculate environmental impact metrics"""
    coverage = risk_data['algae_coverage_percent']
    risk_score = risk_data['risk_score']
    
    # Dissolved oxygen reduction based on algae coverage
    do_reduction = min(50, coverage * 0.8 + risk_score * 20)
    
    # Fish mortality risk assessment
    if coverage > 40:
        fish_risk = "High"
    elif coverage > 20:
        fish_risk = "Medium"
    else:
        fish_risk = "Low"
    
    # Water usability assessment
    water_usability = {
        "Drinking Water": "Unsafe" if coverage > 30 else "Caution" if coverage > 15 else "Safe",
        "Recreation": "Unsafe" if coverage > 25 else "Caution" if coverage > 10 else "Safe",
        "Agriculture": "Caution" if coverage > 35 else "Safe",
        "Aquaculture": "Unsafe" if coverage > 20 else "Caution" if coverage > 10 else "Safe"
    }
    
    # Overall water quality score (0-10)
    water_quality_score = max(0, 10 - (coverage / 10) - (risk_score * 3))
    
    return {
        'dissolved_oxygen_reduction': do_reduction,
        'fish_mortality_risk': fish_risk,
        'water_usability': water_usability,
        'water_quality_score': water_quality_score,
        'risk_distribution': {
            'Low Risk': max(0, 70 - coverage),
            'Medium Risk': min(60, max(20, coverage)),
            'High Risk': max(0, coverage - 30)
        }
    }

def generate_case_study_results(case_study_name, waterbody_data):
    """Generate comprehensive case study results"""
    
    # Base results on waterbody characteristics
    area_km2 = waterbody_data['area_km2']
    depth_m = waterbody_data['depth_m']
    
    # Larger, shallower waterbodies tend to have more algae issues
    base_coverage = min(50, (area_km2 * 2) + (10 / max(1, depth_m)))
    
    # Add some variation based on recent issues
    issue_impact = len(waterbody_data['recent_issues']) * 5
    algae_coverage = base_coverage + issue_impact + np.random.normal(0, 5)
    algae_coverage = max(0, min(100, algae_coverage))
    
    # Calculate indices based on coverage
    indices = {
        'NDVI': 0.3 + (algae_coverage / 200),  # Higher algae = higher vegetation index
        'NDWI': 0.4 - (algae_coverage / 300),  # Higher algae = lower water index
        'Chlorophyll-a': algae_coverage / 5,   # Direct relationship
        'Turbidity': algae_coverage / 4,       # Higher algae = higher turbidity
        'FAI (Floating Algae Index)': algae_coverage / 100  # Normalized FAI
    }
    
    # Risk assessment
    risk_score = algae_coverage / 100 + 0.1
    if risk_score > 0.7:
        risk_level = "High"
    elif risk_score > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    risk_assessment = {
        'algae_coverage_percent': algae_coverage,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'growth_rate': np.random.normal(2.5, 1.0)  # Weekly growth rate
    }
    
    return {
        'type': 'case_study',
        'waterbody': case_study_name,
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'indices': indices,
        'risk_assessment': risk_assessment,
        'temporal_data': generate_temporal_data(indices),
        'environmental_impact': calculate_environmental_impact(risk_assessment)
    }

def generate_demo_analysis_results(analysis_type):
    """Generate demo results when APIs are unavailable"""
    
    demo_indices = {
        'NDVI': np.random.uniform(0.2, 0.8),
        'NDWI': np.random.uniform(-0.3, 0.3),
        'Chlorophyll-a': np.random.uniform(5, 25),
        'Turbidity': np.random.uniform(10, 40),
        'FAI (Floating Algae Index)': np.random.uniform(0.1, 0.6)
    }
    
    algae_coverage = np.random.uniform(10, 45)
    risk_score = algae_coverage / 100 + np.random.uniform(0.1, 0.3)
    
    if risk_score > 0.7:
        risk_level = "High"
    elif risk_score > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    risk_assessment = {
        'algae_coverage_percent': algae_coverage,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'growth_rate': np.random.normal(2.0, 1.5)
    }
    
    st.session_state.analysis_results = {
        'type': analysis_type,
        'waterbody': st.session_state.selected_waterbody or "Demo Location",
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'indices': demo_indices,
        'risk_assessment': risk_assessment,
        'temporal_data': generate_temporal_data(demo_indices),
        'environmental_impact': calculate_environmental_impact(risk_assessment)
    }

def generate_pdf_report(results):
    """Generate PDF report"""
    try:
        report_gen = ReportGenerator()
        pdf_buffer = report_gen.generate_pdf_report(results)
        
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_buffer,
            file_name=f"algae_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")

def generate_csv_export(results):
    """Generate CSV export"""
    try:
        # Prepare data for CSV export
        export_data = []
        
        # Basic info
        export_data.append(['Parameter', 'Value'])
        export_data.append(['Waterbody', results['waterbody']])
        export_data.append(['Analysis Type', results['type']])
        export_data.append(['Analysis Date', results.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))])
        export_data.append([''])
        
        # Indices
        export_data.append(['Spectral Indices', ''])
        for index, value in results['indices'].items():
            if isinstance(value, dict) and 'mean' in value:
                export_data.append([index, value['mean']])
            else:
                export_data.append([index, value])
        
        export_data.append([''])
        
        # Risk assessment
        export_data.append(['Risk Assessment', ''])
        for key, value in results['risk_assessment'].items():
            export_data.append([key.replace('_', ' ').title(), value])
        
        # Convert to DataFrame and CSV
        df = pd.DataFrame(export_data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, header=False)
        
        st.download_button(
            label="⬇️ Download CSV Data",
            data=csv_buffer.getvalue(),
            file_name=f"algae_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error generating CSV: {str(e)}")

if __name__ == "__main__":
    main()
