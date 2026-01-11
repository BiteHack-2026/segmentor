"""
Download ESA WorldCover land cover data and Sentinel-2 imagery from Google Earth Engine.

ESA WorldCover provides global land cover maps at 10m resolution (2020/2021).
Based on Sentinel-1 and Sentinel-2 data with 11 land cover classes.

Requires: earthengine-api, python-dotenv, requests, numpy, pillow
"""

import ee
import os
import requests
import numpy as np
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image

# Load environment variables
load_dotenv()

# Get GEE project name from .env
GEE_PROJECT = os.getenv("GEE_PROJECT")

if not GEE_PROJECT:
    raise ValueError("GEE_PROJECT not found in .env file!")

# ESA WorldCover class definitions
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland", 
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def initialize_gee():
    """Initialize Google Earth Engine with project credentials."""
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"✓ GEE initialized with project: {GEE_PROJECT}")
    except ee.EEException:
        print("Authenticating with Google Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
        print(f"✓ GEE initialized with project: {GEE_PROJECT}")


def download_worldcover(
    center_lon: float,
    center_lat: float,
    size_meters: int = 5000,
    output_path: str = "worldcover.npy",
    image_size: int = 512,
):
    """
    Download ESA WorldCover land cover data.
    
    Args:
        center_lon: Longitude of the center point
        center_lat: Latitude of the center point
        size_meters: Size of the square area in meters
        output_path: Path to save the numpy array
        image_size: Pixel dimensions of the output
    
    Returns:
        numpy array of land cover classes
    """
    center = ee.Geometry.Point([center_lon, center_lat])
    region = center.buffer(size_meters / 2).bounds()
    
    # Load ESA WorldCover 10m v200 (2021)
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
    
    # Get the Map band (contains class values)
    lc_image = worldcover.select("Map")
    
    # Download as numpy array
    url = lc_image.getDownloadURL({
        "region": region,
        "dimensions": f"{image_size}x{image_size}",
        "format": "NPY",
    })
    
    print(f"Downloading WorldCover data...")
    response = requests.get(url)
    response.raise_for_status()
    
    # Load and extract the data
    raw_data = np.load(BytesIO(response.content))
    
    # Extract the Map band from structured array
    if raw_data.dtype.names:
        lc_data = raw_data["Map"].astype(np.uint8)
    else:
        lc_data = raw_data.astype(np.uint8)
    
    # Save to file
    np.save(output_path, lc_data)
    
    print(f"✓ WorldCover data saved to: {output_path}")
    print(f"  Shape: {lc_data.shape}")
    print(f"  Center: ({center_lat}, {center_lon})")
    print(f"  Coverage: {size_meters}m x {size_meters}m")
    
    # Print class statistics
    unique, counts = np.unique(lc_data, return_counts=True)
    print(f"\n  Land cover classes found:")
    for cls, count in zip(unique, counts):
        if cls in WORLDCOVER_CLASSES:
            pct = (count / lc_data.size) * 100
            print(f"    {WORLDCOVER_CLASSES[cls]}: {pct:.1f}%")
    
    return lc_data


def download_rgb_image(
    center_lon: float,
    center_lat: float,
    size_meters: int = 5000,
    output_path: str = "satellite_rgb.jpg",
    image_size: int = 512,
    year: int = 2024,
):
    """Download RGB Sentinel-2 image for visualization."""
    center = ee.Geometry.Point([center_lon, center_lat])
    region = center.buffer(size_meters / 2).bounds()
    
    # Load Sentinel-2 imagery
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(center)
        .filterDate(f"{year}-06-01", f"{year}-08-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
    )
    
    # Create RGB visualization
    vis_image = collection.select(["B4", "B3", "B2"]).visualize(
        min=0,
        max=3000,
    )
    
    url = vis_image.getThumbURL({
        "region": region,
        "dimensions": f"{image_size}x{image_size}",
        "format": "jpg",
    })
    
    print(f"Downloading RGB satellite image...")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    print(f"✓ RGB image saved to: {output_path}")
    
    return output_path


def main():
    # Initialize GEE
    initialize_gee()
    
    # Choose a location with diverse land cover
    # Example: Agricultural area in Poland (for hackathon relevance)
    # center_lon = 20.0  # Poland
    # center_lat = 52.0
    
    # Or: California Central Valley (diverse agriculture)
    center_lon = -120.5
    center_lat = 36.5
    
    # Or: Amazon rainforest edge (forest + agriculture)
    # center_lon = -55.0
    # center_lat = -12.0
    
    size_meters = 10000  # 10km x 10km
    image_size = 512
    
    print(f"\n{'='*60}")
    print(f"Downloading data for location: ({center_lat}, {center_lon})")
    print(f"Area: {size_meters/1000}km x {size_meters/1000}km")
    print(f"{'='*60}\n")
    
    # Download WorldCover land cover data
    download_worldcover(
        center_lon=center_lon,
        center_lat=center_lat,
        size_meters=size_meters,
        output_path="worldcover.npy",
        image_size=image_size,
    )
    
    print()
    
    # Download RGB image for comparison
    download_rgb_image(
        center_lon=center_lon,
        center_lat=center_lat,
        size_meters=size_meters,
        output_path="satellite_rgb.jpg",
        image_size=image_size,
    )
    
    print(f"\n{'='*60}")
    print("Download complete! Run main.py to visualize results.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
