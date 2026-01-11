"""
Visualize ESA WorldCover land cover classification.

ESA WorldCover is a global land cover product at 10m resolution
based on Sentinel-1 and Sentinel-2 data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import os

# ESA WorldCover classes and official colors
WORLDCOVER_CLASSES = {
    10: ("Tree cover", "#006400"),                    # Dark Green
    20: ("Shrubland", "#FFBB22"),                     # Orange
    30: ("Grassland", "#FFFF4C"),                     # Yellow
    40: ("Cropland", "#F096FF"),                      # Pink
    50: ("Built-up", "#FA0000"),                      # Red
    60: ("Bare / sparse vegetation", "#B4B4B4"),      # Gray
    70: ("Snow and ice", "#F0F0F0"),                  # White
    80: ("Permanent water bodies", "#0064C8"),        # Blue
    90: ("Herbaceous wetland", "#0096A0"),            # Teal
    95: ("Mangroves", "#00CF75"),                     # Light Green
    100: ("Moss and lichen", "#FAE6A0"),              # Beige
}

# Create color lookup table
def create_color_lut():
    """Create a color lookup table for WorldCover classes."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for class_id, (name, hex_color) in WORLDCOVER_CLASSES.items():
        # Convert hex to RGB
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        lut[class_id] = [r, g, b]
    return lut


def load_data():
    """Load the WorldCover and RGB data."""
    if not os.path.exists("worldcover.npy"):
        raise FileNotFoundError(
            "worldcover.npy not found. Run download_gee_image.py first!"
        )
    
    worldcover = np.load("worldcover.npy")
    
    rgb_image = None
    if os.path.exists("satellite_rgb.jpg"):
        rgb_image = np.array(Image.open("satellite_rgb.jpg"))
    
    return worldcover, rgb_image


def visualize_worldcover(worldcover, rgb_image=None, output_path="landcover_result.png"):
    """Create visualization of WorldCover data."""
    
    # Create color map
    color_lut = create_color_lut()
    colored_map = color_lut[worldcover]
    
    # Find unique classes present
    unique_classes = np.unique(worldcover)
    
    # Create legend patches
    legend_patches = []
    for cls in sorted(unique_classes):
        if cls in WORLDCOVER_CLASSES:
            name, hex_color = WORLDCOVER_CLASSES[cls]
            # Convert hex to RGB normalized [0,1]
            r = int(hex_color[1:3], 16) / 255
            g = int(hex_color[3:5], 16) / 255
            b = int(hex_color[5:7], 16) / 255
            legend_patches.append(Patch(facecolor=(r, g, b), label=name))
    
    # Create figure
    if rgb_image is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # RGB satellite image
        axes[0].imshow(rgb_image)
        axes[0].set_title("Sentinel-2 RGB", fontsize=12, fontweight='bold')
        axes[0].axis("off")
        
        # Land cover classification
        axes[1].imshow(colored_map)
        axes[1].set_title("ESA WorldCover Classification", fontsize=12, fontweight='bold')
        axes[1].axis("off")
        
        # Side-by-side blend
        # Resize RGB to match worldcover if needed
        if rgb_image.shape[:2] != worldcover.shape:
            from PIL import Image as PILImage
            rgb_resized = np.array(
                PILImage.fromarray(rgb_image).resize(
                    (worldcover.shape[1], worldcover.shape[0])
                )
            )
        else:
            rgb_resized = rgb_image
        
        # Create blended overlay
        alpha = 0.5
        blended = (rgb_resized * (1 - alpha) + colored_map * alpha).astype(np.uint8)
        axes[2].imshow(blended)
        axes[2].set_title("Overlay", fontsize=12, fontweight='bold')
        axes[2].axis("off")
        
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(colored_map)
        ax.set_title("ESA WorldCover Land Cover Classification", fontsize=14, fontweight='bold')
        ax.axis("off")
        axes = [ax]
    
    # Add legend as a separate box on the last plot
    last_ax = axes[-1]
    
    # Create legend with better styling
    legend = last_ax.legend(
        handles=legend_patches,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        title="Land Cover Classes",
        title_fontsize=11,
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=1,
        labelspacing=0.8,
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    
    plt.suptitle("ESA WorldCover 10m Land Cover Classification", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.show()
    
    print(f"\n✓ Visualization saved to: {output_path}")


def print_statistics(worldcover):
    """Print land cover statistics."""
    total_pixels = worldcover.size
    unique, counts = np.unique(worldcover, return_counts=True)
    
    print("\n" + "=" * 50)
    print("LAND COVER STATISTICS")
    print("=" * 50)
    
    # Sort by percentage
    stats = []
    for cls, count in zip(unique, counts):
        if cls in WORLDCOVER_CLASSES:
            name, _ = WORLDCOVER_CLASSES[cls]
            pct = (count / total_pixels) * 100
            stats.append((pct, name, count))
    
    stats.sort(reverse=True)
    
    for pct, name, count in stats:
        bar = "█" * int(pct / 2)
        print(f"{name:30s} {pct:5.1f}% {bar}")
    
    print("=" * 50)


def main():
    print("=" * 60)
    print("ESA WorldCover Land Cover Visualization")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    worldcover, rgb_image = load_data()
    print(f"   WorldCover shape: {worldcover.shape}")
    if rgb_image is not None:
        print(f"   RGB image shape: {rgb_image.shape}")
    
    # Print statistics
    print_statistics(worldcover)
    
    # Visualize
    print("\n2. Creating visualization...")
    visualize_worldcover(worldcover, rgb_image)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()