"""
Filmstrip Generator Module for Satellite Analysis Agent
Phase 2: Extracts frames from GIF and creates a 1x3 vertical filmstrip

Author: Hackathon Team
Timeline: H 2-4
"""

from PIL import Image, ImageDraw, ImageFont
import os
import tempfile
from typing import Tuple, Optional, List


class FilmstripError(Exception):
    """Custom exception for filmstrip generation errors"""
    pass


def validate_gif_path(gif_path: str) -> None:
    """
    Validates that the GIF file exists and is accessible.
    
    Args:
        gif_path: Path to the GIF file
        
    Raises:
        FilmstripError: If file doesn't exist or is not a valid GIF
    """
    if not gif_path:
        raise FilmstripError("GIF path is empty or None")
    
    if not os.path.exists(gif_path):
        raise FilmstripError(f"GIF file not found: {gif_path}")
    
    if not gif_path.lower().endswith('.gif'):
        raise FilmstripError(f"File is not a GIF: {gif_path}")


def extract_frames(gif_path: str, num_frames: int = 3) -> List[Image.Image]:
    """
    Extracts evenly spaced frames from a GIF file.
    
    Args:
        gif_path: Path to the GIF file
        num_frames: Number of frames to extract (default: 3 for start, middle, end)
        
    Returns:
        List of PIL Image objects
        
    Raises:
        FilmstripError: If GIF cannot be loaded or has insufficient frames
    """
    validate_gif_path(gif_path)
    
    try:
        gif = Image.open(gif_path)
    except Exception as e:
        raise FilmstripError(f"Failed to open GIF: {str(e)}")
    
    # Count total frames
    total_frames = 0
    try:
        while True:
            gif.seek(total_frames)
            total_frames += 1
    except EOFError:
        pass  # Reached the end of frames
    
    if total_frames < num_frames:
        raise FilmstripError(
            f"GIF has only {total_frames} frames, need at least {num_frames}"
        )
    
    # Calculate frame indices to extract
    if num_frames == 1:
        indices = [0]
    elif num_frames == 2:
        indices = [0, total_frames - 1]
    else:
        # For 3+ frames: start, evenly spaced middle frames, end
        step = (total_frames - 1) / (num_frames - 1)
        indices = [int(i * step) for i in range(num_frames)]
    
    # Extract frames
    frames = []
    for idx in indices:
        gif.seek(idx)
        # Convert to RGB (GIFs might be in palette mode)
        frame = gif.convert('RGB')
        frames.append(frame.copy())
    
    gif.close()
    
    return frames


def add_frame_labels(
    frames: List[Image.Image],
    labels: Optional[List[str]] = None,
    font_size: int = 20
) -> List[Image.Image]:
    """
    Adds text labels to frames (e.g., "Start", "Middle", "End").
    
    Args:
        frames: List of PIL Image objects
        labels: Optional list of label strings (defaults to Start, Middle, End)
        font_size: Font size for labels
        
    Returns:
        List of frames with labels added
    """
    if labels is None:
        if len(frames) == 3:
            labels = ["Start", "Middle", "End"]
        else:
            labels = [f"Frame {i+1}" for i in range(len(frames))]
    
    labeled_frames = []
    
    for frame, label in zip(frames, labels):
        # Create a copy to avoid modifying original
        labeled_frame = frame.copy()
        draw = ImageDraw.Draw(labeled_frame)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position (top-left corner with padding)
        padding = 10
        
        # Get text bounding box for background
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw semi-transparent background rectangle
        bg_position = [
            padding,
            padding,
            padding + text_width + 10,
            padding + text_height + 10
        ]
        draw.rectangle(bg_position, fill=(0, 0, 0, 180))
        
        # Draw text
        text_position = (padding + 5, padding + 5)
        draw.text(text_position, label, fill="white", font=font)
        
        labeled_frames.append(labeled_frame)
    
    return labeled_frames


def create_vertical_filmstrip(
    frames: List[Image.Image],
    add_labels: bool = True,
    spacing: int = 5,
    background_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Creates a vertical filmstrip from a list of frames.
    
    Args:
        frames: List of PIL Image objects
        add_labels: Whether to add "Start", "Middle", "End" labels
        spacing: Pixels of spacing between frames
        background_color: RGB tuple for background/spacing color
        
    Returns:
        PIL Image object of the vertical filmstrip
    """
    if not frames:
        raise FilmstripError("No frames provided to create filmstrip")
    
    # Add labels if requested
    if add_labels:
        frames = add_frame_labels(frames)
    
    # Get dimensions (assume all frames are same size)
    frame_width = frames[0].width
    frame_height = frames[0].height
    
    # Calculate filmstrip dimensions
    total_spacing = spacing * (len(frames) - 1)
    filmstrip_width = frame_width
    filmstrip_height = (frame_height * len(frames)) + total_spacing
    
    # Create blank filmstrip canvas
    filmstrip = Image.new('RGB', (filmstrip_width, filmstrip_height), background_color)
    
    # Paste frames vertically
    current_y = 0
    for frame in frames:
        filmstrip.paste(frame, (0, current_y))
        current_y += frame_height + spacing
    
    return filmstrip


def generate_filmstrip(
    gif_path: str,
    output_path: str,
    num_frames: int = 3,
    add_labels: bool = True,
    spacing: int = 5
) -> str:
    """
    Main function: Generates a vertical filmstrip from a GIF file.
    
    Args:
        gif_path: Path to input GIF file
        output_path: Path to save output PNG filmstrip
        num_frames: Number of frames to extract (default: 3)
        add_labels: Whether to add frame labels (default: True)
        spacing: Pixels of spacing between frames (default: 5)
        
    Returns:
        Path to the generated filmstrip PNG
        
    Raises:
        FilmstripError: If generation fails
        
    Example:
        >>> filmstrip_path = generate_filmstrip(
        ...     "satellite_timelapse.gif",
        ...     "filmstrip.png"
        ... )
    """
    # Validate input
    validate_gif_path(gif_path)
    
    # Extract frames
    frames = extract_frames(gif_path, num_frames=num_frames)
    
    # Create filmstrip
    filmstrip = create_vertical_filmstrip(
        frames,
        add_labels=add_labels,
        spacing=spacing
    )
    
    # Save to output path
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        filmstrip.save(output_path, format='PNG')
    except Exception as e:
        raise FilmstripError(f"Failed to save filmstrip: {str(e)}")
    
    return output_path


def node_2_filmstrip_generator(state: dict) -> dict:
    """
    LangGraph Node 2: Filmstrip Generator
    
    Takes the GIF path from state and generates a vertical filmstrip,
    saving the path back to state.
    
    Args:
        state: Agent state dictionary containing 'gif_path'
        
    Returns:
        Updated state with 'filmstrip_path' populated
    """
    try:
        gif_path = state.get('gif_path')
        
        if not gif_path:
            raise FilmstripError("No GIF path found in state['gif_path']")
        
        # Generate output path
        # If area_name is available, use it in filename
        area_name = state.get('area_name', 'area')
        safe_area_name = area_name.replace(' ', '_').lower()
        output_filename = f"filmstrip_{safe_area_name}.png"
        
        # Default output location
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        # Generate filmstrip
        filmstrip_path = generate_filmstrip(
            gif_path=gif_path,
            output_path=output_path,
            num_frames=3,
            add_labels=True
        )
        
        # Update state
        state['filmstrip_path'] = filmstrip_path
        
        # Clear any previous errors
        if 'error' in state:
            state['error'] = None
        
        return state
        
    except Exception as e:
        # Store error in state
        state['error'] = f"Filmstrip Generator Error: {str(e)}"
        state['filmstrip_path'] = ""
        return state


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("FILMSTRIP GENERATOR TEST")
    print("=" * 60)
    
    # Create a test GIF for demonstration
    print("\n1. Creating test GIF...")
    
    # Create 5 simple frames with changing colors
    test_frames = []
    colors = [
        (50, 100, 50),   # Dark green
        (80, 130, 80),   # Medium green
        (110, 160, 110), # Light green
        (140, 140, 140), # Gray
        (170, 170, 170)  # Light gray
    ]
    
    for i, color in enumerate(colors):
        frame = Image.new('RGB', (400, 300), color)
        draw = ImageDraw.Draw(frame)
        
        # Add some text to each frame
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        text = f"Year {2019 + i}"
        draw.text((150, 130), text, fill="white", font=font)
        test_frames.append(frame)
    
    # Save as GIF
    test_gif_path = "/tmp/test_satellite.gif"
    test_frames[0].save(
        test_gif_path,
        save_all=True,
        append_images=test_frames[1:],
        duration=500,
        loop=0
    )
    print(f"   ✓ Test GIF created: {test_gif_path}")
    
    # Test filmstrip generation
    print("\n2. Generating filmstrip...")
    try:
        filmstrip_path = generate_filmstrip(
            gif_path=test_gif_path,
            output_path="/tmp/test_filmstrip.png",
            num_frames=3
        )
        print(f"   ✓ Filmstrip created: {filmstrip_path}")
        
        # Check file size
        file_size = os.path.getsize(filmstrip_path) / 1024  # KB
        print(f"   ✓ File size: {file_size:.2f} KB")
        
    except FilmstripError as e:
        print(f"   ✗ Error: {e}")
    
    # Test LangGraph node
    print("\n3. LangGraph Node Test:")
    test_state = {
        'gif_path': test_gif_path,
        'area_name': 'Amazon Rainforest'
    }
    result_state = node_2_filmstrip_generator(test_state)
    
    if result_state.get('error'):
        print(f"   ✗ ERROR: {result_state['error']}")
    else:
        print(f"   ✓ SUCCESS! Filmstrip generated:")
        print(f"   ✓ Path: {result_state['filmstrip_path']}")
        print(f"   ✓ File exists: {os.path.exists(result_state['filmstrip_path'])}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - Test GIF: {test_gif_path}")
    print(f"  - Test Filmstrip: /tmp/test_filmstrip.png")
    print(f"  - Node Output: {result_state.get('filmstrip_path', 'N/A')}")