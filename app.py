"""
Land Cover Segmentation App with ESA WorldCover
Interactive map to select a location and view ESA WorldCover land cover classification.
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

# Load environment variables
load_dotenv()
GEE_PROJECT = os.getenv("GEE_PROJECT")

# Page config
st.set_page_config(
    page_title="Land Cover Segmenter",
    page_icon="🛰️",
    layout="wide"
)

# ESA WorldCover classes and colors
WORLDCOVER_CLASSES = {
    10: ("Tree cover", "#006400"),
    20: ("Shrubland", "#FFBB22"),
    30: ("Grassland", "#FFFF4C"),
    40: ("Cropland", "#F096FF"),
    50: ("Built-up", "#FA0000"),
    60: ("Bare / sparse vegetation", "#B4B4B4"),
    70: ("Snow and ice", "#F0F0F0"),
    80: ("Permanent water bodies", "#0064C8"),
    90: ("Herbaceous wetland", "#0096A0"),
    95: ("Mangroves", "#00CF75"),
    100: ("Moss and lichen", "#FAE6A0"),
}

# Custom CSS for premium styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    h1 {
        background: linear-gradient(90deg, #00d9ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #8892b0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
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
    .legend-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
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


def download_satellite_image(center_lon, center_lat, size_meters=5000, image_size=512):
    """Download satellite image from GEE and return as PIL Image."""
    center = ee.Geometry.Point([center_lon, center_lat])
    region = center.buffer(size_meters / 2).bounds()
    
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(center)
        .filterDate("2024-01-01", "2024-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
    )
    
    image = collection.select(["B4", "B3", "B2"])
    vis_image = image.visualize(min=0, max=3000)
    
    url = vis_image.getThumbURL({
        "region": region,
        "dimensions": f"{image_size}x{image_size}",
        "format": "png",
    })
    
    response = requests.get(url)
    response.raise_for_status()
    
    return Image.open(BytesIO(response.content)).convert("RGB")


def download_worldcover(center_lon, center_lat, size_meters=5000, image_size=512):
    """Download ESA WorldCover land cover data."""
    center = ee.Geometry.Point([center_lon, center_lat])
    region = center.buffer(size_meters / 2).bounds()
    
    # Load ESA WorldCover 10m v200 (2021)
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
    lc_image = worldcover.select("Map")
    
    # Download as numpy array
    url = lc_image.getDownloadURL({
        "region": region,
        "dimensions": f"{image_size}x{image_size}",
        "format": "NPY",
    })
    
    response = requests.get(url)
    response.raise_for_status()
    
    # Load and extract the data
    raw_data = np.load(BytesIO(response.content))
    
    if raw_data.dtype.names:
        lc_data = raw_data["Map"].astype(np.uint8)
    else:
        lc_data = raw_data.astype(np.uint8)
    
    return lc_data


def create_colored_mask(lc_data):
    """Create a colored RGB mask from WorldCover class data."""
    # Create color lookup table
    lut = np.zeros((256, 3), dtype=np.uint8)
    for class_id, (name, hex_color) in WORLDCOVER_CLASSES.items():
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        lut[class_id] = [r, g, b]
    
    # Apply color mapping
    colored = lut[lc_data]
    
    # Calculate class percentages
    unique, counts = np.unique(lc_data, return_counts=True)
    total_pixels = lc_data.size
    class_percentages = {}
    for cls, count in zip(unique, counts):
        if cls in WORLDCOVER_CLASSES:
            class_percentages[cls] = (count / total_pixels) * 100
    
    return Image.fromarray(colored), class_percentages


def main():
    # Header
    st.markdown("# 🛰️ Land Cover Segmenter")
    st.markdown('<p class="subtitle">Click on the map to view ESA WorldCover land cover classification (10m resolution)</p>', unsafe_allow_html=True)
    
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
            min_value=1,
            max_value=20,
            value=5,
            help="Size of the square area to analyze"
        )
        size_meters = size_km * 1000
        
        image_size = st.select_slider(
            "Image resolution",
            options=[256, 512, 768, 1024],
            value=512,
            help="Higher resolution = more detail but slower"
        )
        
        st.markdown("---")
        st.markdown("### 🎨 ESA WorldCover Legend")
        
        for class_id, (name, color) in WORLDCOVER_CLASSES.items():
            st.markdown(
                f'<div class="legend-item">'
                f'<div class="legend-color" style="background-color: {color};"></div>'
                f'<span style="color: white;">{name}</span></div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **ESA WorldCover** is a global land cover map at 10m resolution based on Sentinel-1 and Sentinel-2 data.
        
        - **Year**: 2021
        - **Resolution**: 10m
        - **Classes**: 11
        - **Accuracy**: ~75% overall
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📍 Select Location")
        
        # Initialize map centered on default location
        default_lat = st.session_state.get("last_lat", 37.7749)
        default_lon = st.session_state.get("last_lon", -122.4194)
        
        m = folium.Map(
            location=[default_lat, default_lon],
            zoom_start=10,
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
            
            if st.button("🚀 Analyze This Location", type="primary", use_container_width=True):
                st.session_state.analyze = True
    
    with col2:
        st.markdown("### 🖼️ Results")
        
        if st.session_state.get("analyze") and "selected_lat" in st.session_state:
            with st.spinner("🛰️ Downloading satellite image..."):
                try:
                    satellite_img = download_satellite_image(
                        st.session_state.selected_lon,
                        st.session_state.selected_lat,
                        size_meters,
                        image_size
                    )
                except Exception as e:
                    st.error(f"Failed to download satellite image: {e}")
                    st.session_state.analyze = False
                    st.stop()
            
            with st.spinner("🗺️ Downloading ESA WorldCover data..."):
                try:
                    worldcover_data = download_worldcover(
                        st.session_state.selected_lon,
                        st.session_state.selected_lat,
                        size_meters,
                        image_size
                    )
                    segmented_img, class_percentages = create_colored_mask(worldcover_data)
                except Exception as e:
                    st.error(f"Failed to download WorldCover data: {e}")
                    st.session_state.analyze = False
                    st.stop()
            
            # Display images side by side
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.markdown("**Satellite Image**")
                st.image(satellite_img, use_container_width=True)
            
            with img_col2:
                st.markdown("**ESA WorldCover Classification**")
                st.image(segmented_img, use_container_width=True)
            
            # Show statistics
            st.markdown("### 📊 Land Cover Distribution")
            
            # Sort by percentage
            sorted_classes = sorted(class_percentages.items(), key=lambda x: x[1], reverse=True)
            
            # Display as progress bars
            for class_id, percentage in sorted_classes:
                if class_id in WORLDCOVER_CLASSES:
                    name, color = WORLDCOVER_CLASSES[class_id]
                    
                    col_name, col_bar, col_pct = st.columns([2, 4, 1])
                    with col_name:
                        st.markdown(f"<span style='color: {color}; font-weight: bold;'>■</span> {name}", unsafe_allow_html=True)
                    with col_bar:
                        st.progress(percentage / 100)
                    with col_pct:
                        st.markdown(f"**{percentage:.1f}%**")
            
            st.session_state.analyze = False
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
                    <p style="font-size: 0.9rem;">Then click "Analyze This Location"</p>
                </div>
                """,
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
