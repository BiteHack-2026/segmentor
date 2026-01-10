"""
Download a square JPG image from Google Earth Engine.
Requires: earthengine-api, python-dotenv, requests
"""

import ee
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get GEE project name from .env
GEE_PROJECT = os.getenv("GEE_PROJECT")

if not GEE_PROJECT:
    raise ValueError("GEE_PROJECT not found in .env file!")


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


def download_satellite_image(
    center_lon: float = -122.4194,  # Default: San Francisco
    center_lat: float = 37.7749,
    size_meters: int = 5000,        # Size of the square in meters
    output_path: str = "satellite_image.jpg",
    image_size: int = 512,          # Output image dimensions (square)
):
    """
    Download a square satellite image from Google Earth Engine.
    
    Args:
        center_lon: Longitude of the center point
        center_lat: Latitude of the center point
        size_meters: Size of the square area in meters
        output_path: Path to save the JPG image
        image_size: Pixel dimensions of the output image (width=height)
    """
    # Define center point
    center = ee.Geometry.Point([center_lon, center_lat])
    
    # Create a square region of interest
    region = center.buffer(size_meters / 2).bounds()
    
    # Load Sentinel-2 imagery (most recent cloud-free composite)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(center)
        .filterDate("2024-01-01", "2024-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )
    
    # Select RGB bands and scale to 0-255
    image = collection.select(["B4", "B3", "B2"])  # Red, Green, Blue
    
    # Apply visualization parameters
    vis_image = image.visualize(
        min=0,
        max=3000,
        bands=["B4", "B3", "B2"]
    )
    
    # Get download URL
    url = vis_image.getThumbURL({
        "region": region,
        "dimensions": f"{image_size}x{image_size}",
        "format": "jpg",
    })
    
    print(f"Downloading image from: {url[:80]}...")
    
    # Download and save the image
    response = requests.get(url)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    print(f"✓ Image saved to: {output_path}")
    print(f"  Dimensions: {image_size}x{image_size} pixels")
    print(f"  Center: ({center_lat}, {center_lon})")
    print(f"  Coverage: {size_meters}m x {size_meters}m")


def main():
    # Initialize GEE
    initialize_gee()
    
    # Download a sample image
    # You can customize these parameters:
    download_satellite_image(
        center_lon=-122.4194,    # San Francisco longitude
        center_lat=37.7749,       # San Francisco latitude
        size_meters=10000,        # 10km x 10km square
        output_path="satellite_image.jpg",
        image_size=512,           # 512x512 pixels
    )


if __name__ == "__main__":
    main()
