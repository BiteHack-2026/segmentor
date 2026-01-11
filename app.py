"""
Land Cover Change Analysis App
Interactive map to select a location and analyze historical land cover changes (2015-2019).
Uses Copernicus Global Land Cover (CGLS-LC100) data at 100m resolution.
"""

import streamlit as st
import ee
import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Load environment variables
load_dotenv()
GEE_PROJECT = os.getenv("GEE_PROJECT")

# Page config
st.set_page_config(
    page_title="Land Cover Change Tracker",
    page_icon="🌍",
    layout="wide"
)

# MODIS MCD12Q1 IGBP Land Cover Classes (17 classes + water/unclassified)
# This dataset has annual data from 2001-2022 (20+ years!)
MODIS_IGBP_CLASSES = {
    1: ("Evergreen Needleleaf Forests", "#086a1a"),
    2: ("Evergreen Broadleaf Forests", "#0e6313"),
    3: ("Deciduous Needleleaf Forests", "#07a318"),
    4: ("Deciduous Broadleaf Forests", "#0cab1d"),
    5: ("Mixed Forests", "#1ac926"),
    6: ("Closed Shrublands", "#98920c"),
    7: ("Open Shrublands", "#d5ce05"),
    8: ("Woody Savannas", "#c2a80e"),
    9: ("Savannas", "#f7dc05"),
    10: ("Grasslands", "#ebe50b"),
    11: ("Permanent Wetlands", "#08aec4"),
    12: ("Croplands", "#f7a2da"),
    13: ("Urban and Built-up Lands", "#fa0000"),
    14: ("Cropland/Natural Vegetation Mosaics", "#e95fe0"),
    15: ("Permanent Snow and Ice", "#f0f0f0"),
    16: ("Barren", "#b4b4b4"),
    17: ("Water Bodies", "#0064c8"),
}

# Aggregated classes for simpler analysis (ignoring wetland and barren as requested)
AGGREGATED_CLASSES = {
    "Forest": [1, 2, 3, 4, 5],
    "Shrubland": [6, 7],
    "Savanna": [8, 9],
    "Grassland": [10],
    "Cropland": [12, 14],
    "Built-up": [13],
    "Water": [17],
    "Other": [15],
}

AGGREGATED_COLORS = {
    "Forest": "#228B22",
    "Shrubland": "#DAA520",
    "Savanna": "#F0E68C",
    "Grassland": "#ADFF2F",
    "Cropland": "#FF69B4",
    "Built-up": "#DC143C",
    "Water": "#1E90FF",
    "Other": "#A9A9A9",
}

# Custom CSS for premium styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    h1 {
        background: linear-gradient(90deg, #00d9ff, #00ff88, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }
    .subtitle {
        color: #a8b2d1;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    .change-positive {
        color: #ff6b6b;
        font-weight: bold;
    }
    .change-negative {
        color: #4ade80;
        font-weight: bold;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        margin-right: 15px;
        margin-bottom: 8px;
    }
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 4px;
        margin-right: 6px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .metric-container {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        border-radius: 12px;
        padding: 1rem 1.5rem;
        flex: 1;
        min-width: 150px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-label {
        color: #8892b0;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        color: #ccd6f6;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-delta-up {
        color: #ff6b6b;
        font-size: 0.9rem;
    }
    .metric-delta-down {
        color: #4ade80;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_gee():
    """Initialize Google Earth Engine."""
    if not GEE_PROJECT:
        return False
    try:
        ee.Initialize(project=GEE_PROJECT)
        return True
    except ee.EEException:
        try:
            ee.Authenticate()
            ee.Initialize(project=GEE_PROJECT)
            return True
        except:
            return False


def download_satellite_image(center_lon, center_lat, size_meters=5000, image_size=512, year=2019):
    """Download satellite image from GEE and return as PIL Image.
    
    Uses Sentinel-2 as primary source, falls back to Landsat 8 for older years
    or when Sentinel-2 data is unavailable.
    """
    center = ee.Geometry.Point([center_lon, center_lat])
    region = center.buffer(size_meters / 2).bounds()
    
    # Try Sentinel-2 first (available from mid-2015)
    try:
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(center)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        )
        
        # Check if collection has images
        count = collection.size().getInfo()
        if count > 0:
            image = collection.median().select(["B4", "B3", "B2"])
            vis_image = image.visualize(min=0, max=3000)
            
            url = vis_image.getThumbURL({
                "region": region,
                "dimensions": f"{image_size}x{image_size}",
                "format": "png",
            })
            
            response = requests.get(url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
    except:
        pass
    
    # Fallback to Landsat 8 (available from 2013)
    try:
        collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(center)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filter(ee.Filter.lt("CLOUD_COVER", 30))
        )
        
        count = collection.size().getInfo()
        if count > 0:
            # Apply scaling factors for Landsat
            def apply_scale(img):
                optical = img.select("SR_B.*").multiply(0.0000275).add(-0.2)
                return optical.clamp(0, 1)
            
            image = collection.map(apply_scale).median()
            # Landsat 8 RGB bands: SR_B4 (Red), SR_B3 (Green), SR_B2 (Blue)
            vis_image = image.select(["SR_B4", "SR_B3", "SR_B2"]).visualize(
                min=0, max=0.3
            )
            
            url = vis_image.getThumbURL({
                "region": region,
                "dimensions": f"{image_size}x{image_size}",
                "format": "png",
            })
            
            response = requests.get(url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
    except:
        pass
    
    # If both fail, raise an exception
    raise Exception(f"No satellite imagery available for {year} in this region")


def download_modis_landcover(center_lon, center_lat, year, size_meters=5000, image_size=512):
    """Download MODIS MCD12Q1 Land Cover data for a specific year.
    
    MODIS data available from 2001-2022 at 500m resolution.
    Uses IGBP classification scheme.
    """
    center = ee.Geometry.Point([center_lon, center_lat])
    region = center.buffer(size_meters / 2).bounds()
    
    # Load MODIS Land Cover Type Yearly Global 500m
    # LC_Type1 = IGBP classification (17 classes)
    landcover = (
        ee.ImageCollection("MODIS/061/MCD12Q1")
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .first()
        .select("LC_Type1")
    )
    
    # Download as numpy array
    url = landcover.getDownloadURL({
        "region": region,
        "dimensions": f"{image_size}x{image_size}",
        "format": "NPY",
    })
    
    response = requests.get(url)
    response.raise_for_status()
    
    # Load and extract the data
    raw_data = np.load(BytesIO(response.content))
    
    if raw_data.dtype.names:
        lc_data = raw_data["LC_Type1"].astype(np.uint8)
    else:
        lc_data = raw_data.astype(np.uint8)
    
    return lc_data


def calculate_class_percentages(lc_data):
    """Calculate percentages for each aggregated class."""
    total_pixels = lc_data.size
    percentages = {}
    
    for agg_name, class_ids in AGGREGATED_CLASSES.items():
        mask = np.isin(lc_data, class_ids)
        count = np.sum(mask)
        percentages[agg_name] = (count / total_pixels) * 100
    
    return percentages


def create_colored_mask(lc_data):
    """Create a colored RGB mask from MODIS IGBP land cover class data."""
    # Create color lookup table for MODIS IGBP classes
    lut = np.zeros((256, 3), dtype=np.uint8)
    for class_id, (name, hex_color) in MODIS_IGBP_CLASSES.items():
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        lut[class_id] = [r, g, b]
    
    # Apply color mapping
    colored = lut[lc_data]
    return Image.fromarray(colored)


def analyze_changes(yearly_data):
    """Analyze changes between years."""
    years = sorted(yearly_data.keys())
    df_list = []
    
    for year in years:
        percentages = calculate_class_percentages(yearly_data[year])
        for class_name, pct in percentages.items():
            df_list.append({
                "Year": year,
                "Class": class_name,
                "Percentage": pct
            })
    
    return pd.DataFrame(df_list)


def create_trend_chart(df):
    """Create an interactive line chart showing land cover trends."""
    fig = px.line(
        df,
        x="Year",
        y="Percentage",
        color="Class",
        color_discrete_map=AGGREGATED_COLORS,
        markers=True,
        title="Land Cover Change Over Time"
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccd6f6"),
        title=dict(font=dict(size=18, color="#ccd6f6")),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            tickmode="linear"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            title="Percentage (%)"
        ),
        hovermode="x unified"
    )
    
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=10)
    )
    
    return fig


def create_bar_chart(df):
    """Create a grouped bar chart comparing first and last year."""
    years = sorted(df["Year"].unique())
    first_year = years[0]
    last_year = years[-1]
    
    comparison_df = df[df["Year"].isin([first_year, last_year])].copy()
    comparison_df["Year"] = comparison_df["Year"].astype(str)
    
    fig = px.bar(
        comparison_df,
        x="Class",
        y="Percentage",
        color="Year",
        barmode="group",
        color_discrete_sequence=["#6366f1", "#22d3ee"],
        title=f"Land Cover Comparison: {first_year} vs {last_year}"
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccd6f6"),
        title=dict(font=dict(size=18, color="#ccd6f6")),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=False,
            tickangle=-45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            title="Percentage (%)"
        )
    )
    
    return fig


def create_change_chart(df):
    """Create a bar chart showing net change for each class."""
    years = sorted(df["Year"].unique())
    first_year = years[0]
    last_year = years[-1]
    
    changes = []
    for class_name in AGGREGATED_CLASSES.keys():
        first_val = df[(df["Year"] == first_year) & (df["Class"] == class_name)]["Percentage"].values
        last_val = df[(df["Year"] == last_year) & (df["Class"] == class_name)]["Percentage"].values
        
        if len(first_val) > 0 and len(last_val) > 0:
            change = last_val[0] - first_val[0]
            changes.append({
                "Class": class_name,
                "Change": change,
                "Color": "#ff6b6b" if change > 0 else "#4ade80"
            })
    
    change_df = pd.DataFrame(changes)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=change_df["Class"],
        y=change_df["Change"],
        marker_color=change_df["Color"],
        text=[f"{c:+.2f}%" for c in change_df["Change"]],
        textposition="outside",
        textfont=dict(color="#ccd6f6", size=12)
    ))
    
    fig.update_layout(
        title=f"Net Change in Land Cover ({first_year} → {last_year})",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccd6f6"),
        title_font=dict(size=18, color="#ccd6f6"),
        xaxis=dict(
            showgrid=False,
            tickangle=-45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            title="Change (%)",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.3)",
            zerolinewidth=2
        ),
        showlegend=False
    )
    
    return fig, changes


def create_area_chart(df):
    """Create a stacked area chart showing land cover composition over time."""
    pivot_df = df.pivot(index="Year", columns="Class", values="Percentage").reset_index()
    
    fig = go.Figure()
    
    for class_name in AGGREGATED_CLASSES.keys():
        if class_name in pivot_df.columns:
            fig.add_trace(go.Scatter(
                x=pivot_df["Year"],
                y=pivot_df[class_name],
                mode="lines",
                stackgroup="one",
                name=class_name,
                line=dict(width=0.5, color=AGGREGATED_COLORS[class_name]),
                fillcolor=AGGREGATED_COLORS[class_name],
                hovertemplate=f"{class_name}: " + "%{y:.1f}%<extra></extra>"
            ))
    
    fig.update_layout(
        title="Land Cover Composition Over Time",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccd6f6"),
        title_font=dict(size=18, color="#ccd6f6"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=False,
            tickmode="linear"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            title="Percentage (%)",
            range=[0, 100]
        ),
        hovermode="x unified"
    )
    
    return fig


def main():
    # Header
    st.markdown("# 🌍 Land Cover Change Tracker")
    st.markdown('<p class="subtitle">Analyze 20+ years of land cover changes (2001-2022) using MODIS satellite data</p>', unsafe_allow_html=True)
    
    # Initialize GEE
    gee_ready = initialize_gee()
    if not gee_ready:
        st.error("⚠️ Google Earth Engine not configured. Please set GEE_PROJECT in your .env file.")
        st.stop()
    
    # Sidebar settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        size_km = st.slider(
            "Area size (km)",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Size of the square area to analyze. Larger = better for regional changes (MODIS is 500m resolution)"
        )
        size_meters = size_km * 1000
        
        # Calculate approximate pixels at MODIS resolution
        approx_pixels = size_km * 1000 / 500
        st.caption(f"≈ {int(approx_pixels)}×{int(approx_pixels)} pixels at MODIS 500m resolution")
        
        image_size = st.select_slider(
            "Output image size",
            options=[256, 512, 768, 1024],
            value=512,
            help="Output image dimensions in pixels"
        )
        
        st.markdown("---")
        st.markdown("### 📅 Time Range")
        st.info("📆 MODIS data: **2001-2022** (20+ years!)")
        
        years_to_analyze = st.multiselect(
            "Select years",
            options=list(range(2001, 2023)),
            default=[2001, 2005, 2010, 2015, 2020],
            help="Select years to include in the analysis (recommend 5+ year gaps)"
        )
        
        st.markdown("---")
        st.markdown("### 🎨 Aggregated Classes")
        st.markdown("""
        **MODIS IGBP Classification:**
        - **Forest**: All forest types
        - **Shrubland**: Open & closed shrubs
        - **Savanna**: Woody & herbaceous
        - **Grassland**: Grasslands
        - **Cropland**: Agricultural areas
        - **Built-up**: Urban/developed
        - **Water**: Water bodies
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Ignored Classes")
        st.caption("These classes are excluded from the analysis:")
        st.markdown("- Herbaceous wetland\n- Bare / sparse vegetation")
        
        st.markdown("---")
        st.markdown("### 🔥 Hotspot Locations")
        st.caption("Areas with significant changes (2015-2019)")
        
        # Define hotspot locations - precise coords for documented changes
        HOTSPOTS = {
            "-- Select a hotspot --": None,
            # Urban expansion
            "🏗️ Shenzhen Suburbs, China": {"lat": 22.65, "lon": 113.85, "desc": "Rapid urban sprawl visible"},
            "🏗️ Bangalore Outskirts, India": {"lat": 13.15, "lon": 77.45, "desc": "IT corridor expansion"},
            "🏗️ Lagos-Ikorodu, Nigeria": {"lat": 6.62, "lon": 3.50, "desc": "Megacity expansion"},
            "🏗️ Chengdu New District, China": {"lat": 30.80, "lon": 104.25, "desc": "New urban development"},
            # Deforestation hotspots - precise frontier coordinates
            "🌳 BR-319 Road, Amazonas": {"lat": -6.5, "lon": -62.5, "desc": "Highway deforestation corridor"},
            "🌳 Novo Progresso, Pará Brazil": {"lat": -7.0, "lon": -55.4, "desc": "Cattle frontier hotspot"},
            "🌳 Sinop Region, Mato Grosso": {"lat": -11.85, "lon": -55.5, "desc": "Soy expansion frontier"},
            "🌳 Merauke, Papua Indonesia": {"lat": -7.5, "lon": 140.0, "desc": "Palm oil expansion zone"},
            "🌳 Jambi, Sumatra Indonesia": {"lat": -1.6, "lon": 103.6, "desc": "Plantation conversion"},
            # Agricultural expansion
            "🌾 Santa Cruz, Bolivia": {"lat": -17.5, "lon": -62.5, "desc": "Soy expansion visible"},
            "🌾 Chaco, Paraguay": {"lat": -22.0, "lon": -60.0, "desc": "Cattle ranching expansion"},
        }
        
        selected_hotspot = st.selectbox(
            "Jump to hotspot",
            options=list(HOTSPOTS.keys()),
            index=0,
            help="Select a location known for major changes"
        )
        
        if HOTSPOTS[selected_hotspot] is not None:
            hotspot = HOTSPOTS[selected_hotspot]
            if st.button("📍 Use This Location", use_container_width=True):
                st.session_state.selected_lat = hotspot["lat"]
                st.session_state.selected_lon = hotspot["lon"]
                st.session_state.last_lat = hotspot["lat"]
                st.session_state.last_lon = hotspot["lon"]
                st.session_state.hotspot_used = selected_hotspot
                st.rerun()
            st.caption(f"*{hotspot['desc']}*")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **MODIS MCD12Q1** is a 20+ year global land cover product.
        
        - **Years**: 2001-2022
        - **Resolution**: 500m
        - **Classification**: IGBP (17 classes)
        - **Source**: Terra & Aqua satellites
        
        *Long time range = dramatic changes visible!*
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📍 Select Location")
        
        # Initialize map centered on default location
        default_lat = st.session_state.get("last_lat", 50.0)
        default_lon = st.session_state.get("last_lon", 19.0)
        
        m = folium.Map(
            location=[default_lat, default_lon],
            zoom_start=8,
            tiles="OpenStreetMap"
        )
        
        # Add marker if location selected
        if "selected_lat" in st.session_state:
            folium.Marker(
                [st.session_state.selected_lat, st.session_state.selected_lon],
                popup=f"Selected: {st.session_state.selected_lat:.4f}, {st.session_state.selected_lon:.4f}",
                icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")
            ).add_to(m)
        
        # Display map
        map_data = st_folium(m, width=None, height=400, returned_objects=["last_clicked"])
        
        # Handle click
        if map_data and map_data.get("last_clicked"):
            clicked = map_data["last_clicked"]
            st.session_state.selected_lat = clicked["lat"]
            st.session_state.selected_lon = clicked["lng"]
            st.session_state.last_lat = clicked["lat"]
            st.session_state.last_lon = clicked["lng"]
        
        # Show selected coordinates
        if "selected_lat" in st.session_state:
            st.info(f"📌 Selected: **{st.session_state.selected_lat:.4f}°, {st.session_state.selected_lon:.4f}°**")
            
            if len(years_to_analyze) < 2:
                st.warning("Please select at least 2 years to analyze changes.")
            else:
                if st.button("🚀 Analyze Historical Changes", type="primary", use_container_width=True):
                    st.session_state.analyze = True
    
    with col2:
        st.markdown("### 🖼️ Preview")
        
        if st.session_state.get("analyze") and "selected_lat" in st.session_state:
            yearly_data = {}
            yearly_images = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_steps = len(years_to_analyze) * 2 + 1  # landcover + satellite for each year + analysis
            current_step = 0
            
            # Download data for each year
            for i, year in enumerate(sorted(years_to_analyze)):
                status_text.text(f"📥 Downloading {year} MODIS land cover data...")
                try:
                    lc_data = download_modis_landcover(
                        st.session_state.selected_lon,
                        st.session_state.selected_lat,
                        year,
                        size_meters,
                        image_size
                    )
                    yearly_data[year] = lc_data
                    yearly_images[year] = create_colored_mask(lc_data)
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                except Exception as e:
                    st.error(f"Failed to download {year} data: {e}")
                    st.session_state.analyze = False
                    st.stop()
                
                # Download satellite image for the year
                status_text.text(f"🛰️ Downloading {year} satellite image...")
                try:
                    sat_img = download_satellite_image(
                        st.session_state.selected_lon,
                        st.session_state.selected_lat,
                        size_meters,
                        image_size,
                        year
                    )
                    st.session_state[f"sat_img_{year}"] = sat_img
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                except Exception as e:
                    st.warning(f"Could not download satellite image for {year}")
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
            
            status_text.text("📊 Analyzing changes...")
            df = analyze_changes(yearly_data)
            current_step += 1
            progress_bar.progress(1.0)
            
            # Store results in session state
            st.session_state.yearly_data = yearly_data
            st.session_state.yearly_images = yearly_images
            st.session_state.change_df = df
            st.session_state.analysis_complete = True
            
            progress_bar.empty()
            status_text.empty()
            st.session_state.analyze = False
        
        # Display results if available
        if st.session_state.get("analysis_complete"):
            years = sorted(st.session_state.yearly_images.keys())
            
            # Show satellite and land cover images for first and last year
            first_year = years[0]
            last_year = years[-1]
            
            st.markdown(f"**{first_year} → {last_year} Comparison**")
            
            # Placeholder HTML for missing satellite images
            placeholder_html = """
                <div style="
                    width: 100%;
                    aspect-ratio: 1;
                    background: linear-gradient(135deg, rgba(100,100,100,0.2), rgba(50,50,50,0.3));
                    border: 2px dashed rgba(255,255,255,0.2);
                    border-radius: 8px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: rgba(255,255,255,0.4);
                    font-size: 2rem;
                    margin-bottom: 0.5rem;
                ">📷</div>
                <p style="text-align: center; color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 0;">No satellite image</p>
            """
            
            # First year column
            img_cols = st.columns(2)
            with img_cols[0]:
                st.markdown(f"**{first_year}**")
                # Show satellite image if available
                sat_key = f"sat_img_{first_year}"
                if sat_key in st.session_state and st.session_state[sat_key] is not None:
                    st.image(st.session_state[sat_key], use_container_width=True, caption="Satellite")
                else:
                    st.markdown(placeholder_html, unsafe_allow_html=True)
                st.image(st.session_state.yearly_images[first_year], use_container_width=True, caption="Land Cover")
            
            with img_cols[1]:
                st.markdown(f"**{last_year}**")
                # Show satellite image if available
                sat_key = f"sat_img_{last_year}"
                if sat_key in st.session_state and st.session_state[sat_key] is not None:
                    st.image(st.session_state[sat_key], use_container_width=True, caption="Satellite")
                else:
                    st.markdown(placeholder_html, unsafe_allow_html=True)
                st.image(st.session_state.yearly_images[last_year], use_container_width=True, caption="Land Cover")
        else:
            st.markdown(
                """
                <div style="
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 300px;
                    background: rgba(255,255,255,0.02);
                    border: 2px dashed rgba(255,255,255,0.1);
                    border-radius: 12px;
                    color: #8892b0;
                ">
                    <p style="font-size: 3rem; margin-bottom: 0.5rem;">👆</p>
                    <p>Click on the map to select a location</p>
                    <p style="font-size: 0.9rem;">Then click "Analyze Historical Changes"</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Show charts if analysis is complete
    if st.session_state.get("analysis_complete"):
        st.markdown("---")
        st.markdown("## 📊 Land Cover Change Analysis")
        
        df = st.session_state.change_df
        years = sorted(df["Year"].unique())
        first_year = years[0]
        last_year = years[-1]
        
        # Key metrics
        st.markdown("### 🔑 Key Findings")
        
        # Calculate changes for key classes (safely)
        def get_class_value(df, year, class_name):
            vals = df[(df["Year"] == year) & (df["Class"] == class_name)]["Percentage"].values
            return vals[0] if len(vals) > 0 else 0
        
        tree_first = get_class_value(df, first_year, "Forest")
        tree_last = get_class_value(df, last_year, "Forest")
        tree_change = tree_last - tree_first
        
        buildup_first = get_class_value(df, first_year, "Built-up")
        buildup_last = get_class_value(df, last_year, "Built-up")
        buildup_change = buildup_last - buildup_first
        
        cropland_first = get_class_value(df, first_year, "Cropland")
        cropland_last = get_class_value(df, last_year, "Cropland")
        cropland_change = cropland_last - cropland_first
        
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            delta_color = "inverse" if tree_change < 0 else "normal"
            st.metric(
                label="🌲 Forest",
                value=f"{tree_last:.1f}%",
                delta=f"{tree_change:+.2f}%",
                delta_color=delta_color
            )
        
        with metric_cols[1]:
            delta_color = "normal" if buildup_change > 0 else "inverse"
            st.metric(
                label="🏘️ Built-up",
                value=f"{buildup_last:.1f}%",
                delta=f"{buildup_change:+.2f}%",
                delta_color=delta_color
            )
        
        with metric_cols[2]:
            st.metric(
                label="🌾 Cropland",
                value=f"{cropland_last:.1f}%",
                delta=f"{cropland_change:+.2f}%"
            )
        
        with metric_cols[3]:
            total_years = last_year - first_year
            st.metric(
                label="📅 Analysis Period",
                value=f"{total_years} years",
                delta=f"{first_year}-{last_year}"
            )
        
        # Charts
        st.markdown("### 📈 Trend Analysis")
        
        chart_tabs = st.tabs(["📈 Trends", "📊 Comparison", "📉 Net Change", "🗺️ Composition"])
        
        with chart_tabs[0]:
            fig = create_trend_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_tabs[1]:
            fig = create_bar_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_tabs[2]:
            fig, changes = create_change_chart(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary text
            st.markdown("#### 📝 Summary")
            
            increasing = [c for c in changes if c["Change"] > 0.1]
            decreasing = [c for c in changes if c["Change"] < -0.1]
            
            if increasing:
                inc_text = ", ".join([f"**{c['Class']}** (+{c['Change']:.2f}%)" for c in increasing])
                st.markdown(f"📈 **Increasing:** {inc_text}")
            
            if decreasing:
                dec_text = ", ".join([f"**{c['Class']}** ({c['Change']:.2f}%)" for c in decreasing])
                st.markdown(f"📉 **Decreasing:** {dec_text}")
            
            if not increasing and not decreasing:
                st.info("No significant changes detected (threshold: ±0.1%)")
        
        with chart_tabs[3]:
            fig = create_area_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Year-by-year gallery
        st.markdown("### 🖼️ Year-by-Year Gallery")
        st.markdown("*Satellite imagery (top) and land cover classification (bottom)*")
        
        year_cols = st.columns(len(years))
        for i, year in enumerate(years):
            with year_cols[i]:
                st.markdown(f"**{year}**")
                
                # Show satellite image if available, otherwise show placeholder
                sat_key = f"sat_img_{year}"
                if sat_key in st.session_state and st.session_state[sat_key] is not None:
                    st.image(st.session_state[sat_key], use_container_width=True, caption="Satellite")
                else:
                    # Create a placeholder box to maintain alignment
                    st.markdown(
                        """
                        <div style="
                            width: 100%;
                            aspect-ratio: 1;
                            background: linear-gradient(135deg, rgba(100,100,100,0.2), rgba(50,50,50,0.3));
                            border: 2px dashed rgba(255,255,255,0.2);
                            border-radius: 8px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            color: rgba(255,255,255,0.4);
                            font-size: 2rem;
                            margin-bottom: 0.5rem;
                        ">📷</div>
                        <p style="text-align: center; color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 0;">No satellite image</p>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Show land cover classification
                st.image(st.session_state.yearly_images[year], use_container_width=True, caption="Land Cover")
                
                # Show percentages for this year
                year_data = df[df["Year"] == year]
                with st.expander("📊 Details"):
                    for _, row in year_data.iterrows():
                        if row["Percentage"] > 0.5:
                            st.caption(f"{row['Class']}: {row['Percentage']:.1f}%")


if __name__ == "__main__":
    main()
