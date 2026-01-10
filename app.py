"""
Land Cover Segmentation App
Interactive map to select a location, download satellite imagery, and segment land cover.
"""

import streamlit as st
import ee
import os
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
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
def load_segmentation_model():
    """Load the segmentation model (cached for performance)."""
    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = AutoModelForSemanticSegmentation.from_pretrained(
        "florian-morel22/segformer-b0-deepglobe-land-cover"
    )
    return processor, model


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
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )
    
    image = collection.select(["B4", "B3", "B2"])
    vis_image = image.visualize(min=0, max=3000, bands=["B4", "B3", "B2"])
    
    url = vis_image.getThumbURL({
        "region": region,
        "dimensions": f"{image_size}x{image_size}",
        "format": "png",
    })
    
    response = requests.get(url)
    response.raise_for_status()
    
    return Image.open(BytesIO(response.content)).convert("RGB")


def segment_image(image, processor, model):
    """Run segmentation on the image and return colored mask."""
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred_mask = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # Color palette
    palette = np.array([
        [0, 255, 255],   # 0: Urban
        [255, 255, 0],   # 1: Agriculture
        [255, 0, 255],   # 2: Rangeland
        [0, 255, 0],     # 3: Forest
        [0, 0, 255],     # 4: Water
        [255, 255, 255], # 5: Barren
        [0, 0, 0]        # 6: Unknown
    ], dtype=np.uint8)
    
    color_mask = palette[pred_mask]
    
    # Calculate class percentages
    unique, counts = np.unique(pred_mask, return_counts=True)
    total_pixels = pred_mask.size
    class_percentages = {int(u): (c / total_pixels) * 100 for u, c in zip(unique, counts)}
    
    return Image.fromarray(color_mask), class_percentages


def main():
    # Header
    st.markdown("# 🛰️ Land Cover Segmenter")
    st.markdown('<p class="subtitle">Click on the map to analyze satellite imagery with AI-powered land cover segmentation</p>', unsafe_allow_html=True)
    
    # Initialize GEE
    gee_ready = initialize_gee()
    if not gee_ready:
        st.error("⚠️ Google Earth Engine not configured. Please set GEE_PROJECT in your .env file.")
        st.stop()
    
    # Load model
    with st.spinner("Loading AI model..."):
        processor, model = load_segmentation_model()
    
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
        st.markdown("### 🎨 Legend")
        
        legend_data = [
            ("Urban", "#00FFFF"),
            ("Agriculture", "#FFFF00"),
            ("Rangeland", "#FF00FF"),
            ("Forest", "#00FF00"),
            ("Water", "#0000FF"),
            ("Barren", "#FFFFFF"),
            ("Unknown", "#000000"),
        ]
        
        for name, color in legend_data:
            st.markdown(
                f'<div class="legend-item">'
                f'<div class="legend-color" style="background-color: {color};"></div>'
                f'<span style="color: white;">{name}</span></div>',
                unsafe_allow_html=True
            )
    
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
                    st.error(f"Failed to download image: {e}")
                    st.session_state.analyze = False
                    st.stop()
            
            with st.spinner("🧠 Running AI segmentation..."):
                segmented_img, class_percentages = segment_image(satellite_img, processor, model)
            
            # Display images side by side
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.markdown("**Satellite Image**")
                st.image(satellite_img, use_container_width=True)
            
            with img_col2:
                st.markdown("**Land Cover Segmentation**")
                st.image(segmented_img, use_container_width=True)
            
            # Show statistics
            st.markdown("### 📊 Land Cover Distribution")
            
            class_names = ["Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"]
            class_colors = ["#00FFFF", "#FFFF00", "#FF00FF", "#00FF00", "#0000FF", "#FFFFFF", "#000000"]
            
            cols = st.columns(len(class_percentages))
            for i, (class_id, percentage) in enumerate(sorted(class_percentages.items())):
                with cols[i % len(cols)]:
                    st.metric(
                        label=class_names[class_id],
                        value=f"{percentage:.1f}%"
                    )
            
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
