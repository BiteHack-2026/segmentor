"""
Report Assembler Module for Satellite Analysis Agent
Phase 4: Uses Jinja2 to generate professional HTML reports

Author: Hackathon Team
Timeline: H 6-9
"""

import base64
import os
import tempfile
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template


class ReportAssemblerError(Exception):
    """Custom exception for report assembly errors"""

    pass


def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes an image file to base64 for embedding in HTML.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string with data URI prefix

    Raises:
        ReportAssemblerError: If image cannot be read
    """
    try:
        # Determine mime type from extension
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.gif':
            mime_type = 'image/gif'
        elif ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        else:
            mime_type = 'image/png'

        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        raise ReportAssemblerError(f"Failed to encode image: {str(e)}")


def prepare_template_context(
    data_df: pd.DataFrame,
    narrative_json: Dict[str, str],
    area_name: str,
    time_period: Optional[Dict[str, str]] = None,
    filmstrip_path: Optional[str] = None,
    embed_images: bool = True,
) -> Dict:
    """
    Prepares the context dictionary for Jinja2 template rendering.

    Args:
        data_df: DataFrame with YoY data
        narrative_json: Dictionary with analysis sections
        area_name: Name of the geographic area
        time_period: Dict with 'start' and 'end' years
        filmstrip_path: Path to the filmstrip image
        embed_images: If True, embed images as base64; if False, use file paths

    Returns:
        Context dictionary for template rendering
    """
    context = {
        "area_name": area_name,
        "time_period": time_period,
        "narrative": narrative_json,
        "data_df": data_df,
        "data_columns": list(data_df.columns),
        "generation_timestamp": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
    }

    # Handle filmstrip image
    if filmstrip_path and os.path.exists(filmstrip_path):
        if embed_images:
            # Embed as base64 for standalone HTML
            context["filmstrip_path"] = encode_image_to_base64(filmstrip_path)
        else:
            # Use relative or absolute path
            context["filmstrip_path"] = filmstrip_path
    else:
        context["filmstrip_path"] = None

    return context


def load_template(template_path: Optional[str] = None) -> Template:
    """
    Loads the Jinja2 template from file or uses default.

    Args:
        template_path: Path to custom template file (optional)

    Returns:
        Jinja2 Template object

    Raises:
        ReportAssemblerError: If template cannot be loaded
    """
    if template_path and os.path.exists(template_path):
        # Load custom template
        template_dir = os.path.dirname(template_path)
        template_name = os.path.basename(template_path)

        try:
            env = Environment(loader=FileSystemLoader(template_dir))
            template = env.get_template(template_name)
            return template
        except Exception as e:
            raise ReportAssemblerError(f"Failed to load template: {str(e)}")
    else:
        # Use default template in same directory
        default_template_path = os.path.join(
            os.path.dirname(__file__), "report_template.html"
        )

        if os.path.exists(default_template_path):
            template_dir = os.path.dirname(default_template_path)
            env = Environment(loader=FileSystemLoader(template_dir))
            template = env.get_template("report_template.html")
            return template
        else:
            raise ReportAssemblerError(
                f"Default template not found at {default_template_path}"
            )


def render_report(
    data_df: pd.DataFrame,
    narrative_json: Dict[str, str],
    area_name: str,
    time_period: Optional[Dict[str, str]] = None,
    filmstrip_path: Optional[str] = None,
    template_path: Optional[str] = None,
    embed_images: bool = True,
) -> str:
    """
    Renders the HTML report using Jinja2 template.

    Args:
        data_df: DataFrame with YoY data
        narrative_json: Dictionary with analysis sections
        area_name: Name of the geographic area
        time_period: Dict with 'start' and 'end' years
        filmstrip_path: Path to the filmstrip image
        template_path: Path to custom template (optional)
        embed_images: If True, embed images as base64

    Returns:
        Rendered HTML string

    Raises:
        ReportAssemblerError: If rendering fails
    """
    # Load template
    template = load_template(template_path)

    # Prepare context
    context = prepare_template_context(
        data_df=data_df,
        narrative_json=narrative_json,
        area_name=area_name,
        time_period=time_period,
        filmstrip_path=filmstrip_path,
        embed_images=embed_images,
    )

    # Render template
    try:
        html_content = template.render(**context)
        return html_content
    except Exception as e:
        raise ReportAssemblerError(f"Failed to render template: {str(e)}")


def assemble_report(
    data_df: pd.DataFrame,
    narrative_json: Dict[str, str],
    area_name: str,
    output_path: str,
    time_period: Optional[Dict[str, str]] = None,
    filmstrip_path: Optional[str] = None,
    template_path: Optional[str] = None,
    embed_images: bool = True,
) -> str:
    """
    Main function: Assembles and saves the HTML report.

    Args:
        data_df: DataFrame with YoY data
        narrative_json: Dictionary with analysis sections
        area_name: Name of the geographic area
        output_path: Path to save the HTML report
        time_period: Dict with 'start' and 'end' years
        filmstrip_path: Path to the filmstrip image
        template_path: Path to custom template (optional)
        embed_images: If True, embed images as base64

    Returns:
        Path to the generated HTML report

    Raises:
        ReportAssemblerError: If assembly fails

    Example:
        >>> report_path = assemble_report(
        ...     data_df=df,
        ...     narrative_json=narrative,
        ...     area_name="Amazon Basin",
        ...     output_path="report.html",
        ...     filmstrip_path="filmstrip.png"
        ... )
    """
    # Render HTML
    html_content = render_report(
        data_df=data_df,
        narrative_json=narrative_json,
        area_name=area_name,
        time_period=time_period,
        filmstrip_path=filmstrip_path,
        template_path=template_path,
        embed_images=embed_images,
    )

    # Save to file
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        raise ReportAssemblerError(f"Failed to save report: {str(e)}")

    return output_path


def node_4_report_assembler(state: dict, template_path: Optional[str] = None) -> dict:
    """
    LangGraph Node 4: Report Assembler

    Takes all components from state (data, narrative, filmstrip) and
    assembles them into a professional HTML report.

    Args:
        state: Agent state dictionary
        template_path: Optional path to custom template

    Returns:
        Updated state with 'html_path' populated
    """
    try:
        # Extract required fields from state
        data_df = state.get("data_df")
        narrative_json = state.get("narrative_json")
        area_name = state.get("area_name", "Study Area")
        time_period = state.get("time_period")
        
        # Use GIF for HTML report if available (better visualization)
        # Fallback to generated filmstrip if GIF is missing
        gif_path = state.get("gif_path")
        filmstrip_path = state.get("filmstrip_path")
        
        report_image_path = gif_path if (gif_path and os.path.exists(gif_path)) else filmstrip_path

        # Validate inputs
        if data_df is None:
            raise ReportAssemblerError("No DataFrame found in state['data_df']")

        if not narrative_json:
            raise ReportAssemblerError("No narrative found in state['narrative_json']")

        # Generate output path
        safe_area_name = area_name.replace(" ", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"report_{safe_area_name}_{timestamp}.html"

        # Default output location
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        # Assemble report
        html_path = assemble_report(
            data_df=data_df,
            narrative_json=narrative_json,
            area_name=area_name,
            output_path=output_path,
            time_period=time_period,
            filmstrip_path=report_image_path,
            template_path=template_path,
            embed_images=True,  # Embed for standalone HTML
        )

        # Update state
        state["html_path"] = html_path

        # Clear any previous errors
        if "error" in state:
            state["error"] = None

        return state

    except Exception as e:
        # Store error in state
        state["error"] = f"Report Assembler Error: {str(e)}"
        state["html_path"] = ""
        return state


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("REPORT ASSEMBLER TEST")
    print("=" * 60)

    # Sample data (from Phase 1)
    sample_data = pd.DataFrame(
        {
            "year": [2019, 2020, 2021, 2022, 2023],
            "forest_pct_change": [-1.2, -2.1, -1.5, -2.8, -3.1],
            "urban_pct_change": [3.5, 5.0, 4.2, 6.1, 5.8],
            "water_pct_change": [0.1, -0.2, 0.3, -0.1, 0.2],
        }
    )

    # Sample narrative (from Phase 3)
    sample_narrative = {
        "executive_summary": "The Amazon Rainforest experienced accelerating deforestation from 2019 to 2023, with forest loss increasing from -1.2% to -3.1% annually. Urban expansion remained consistently high, averaging 4.9% growth per year, while water bodies showed minimal fluctuation.",
        "forest_analysis": "Forest cover declined every year during the analysis period, with the rate of loss accelerating over time. The most drastic forest loss occurred in 2023 at -3.1%, representing a 158% increase in annual loss rate compared to 2019. This acceleration suggests intensifying deforestation pressures, potentially linked to agricultural expansion and infrastructure development.",
        "urban_analysis": "Urban areas experienced sustained expansion throughout the period, peaking at 6.1% growth in 2022. The consistently high growth rates (ranging from 3.5% to 6.1%) indicate ongoing urbanization and development pressure in the region. The slight decrease in 2023 to 5.8% may suggest approaching spatial constraints or policy interventions.",
        "water_analysis": "Water body coverage remained relatively stable with minor fluctuations between -0.2% and +0.3%. These small variations likely reflect seasonal changes in precipitation and water management rather than significant hydrological trends. The overall stability suggests that water resources have not been dramatically affected during this period.",
    }

    # Sample filmstrip path
    sample_filmstrip = "/tmp/filmstrip_amazon_rainforest.png"

    # Check if filmstrip exists (from Phase 2 test)
    if not os.path.exists(sample_filmstrip):
        print("\n⚠️  Filmstrip from Phase 2 not found. Creating placeholder...")
        # Create a simple placeholder for testing
        from PIL import Image, ImageDraw

        placeholder = Image.new("RGB", (400, 900), (100, 100, 100))
        draw = ImageDraw.Draw(placeholder)
        draw.text((150, 450), "Placeholder", fill="white")
        placeholder.save(sample_filmstrip)
        print(f"   ✓ Placeholder created: {sample_filmstrip}")

    # Test report assembly
    print("\n1. Assembling HTML report...")
    try:
        report_path = assemble_report(
            data_df=sample_data,
            narrative_json=sample_narrative,
            area_name="Amazon Rainforest",
            output_path="/outputs/reports/test_report.html",
            time_period={"start": "2019", "end": "2023"},
            filmstrip_path=sample_filmstrip,
            embed_images=True,
        )

        print(f"   ✓ Report created: {report_path}")

        # Check file size
        file_size = os.path.getsize(report_path) / 1024  # KB
        print(f"   ✓ File size: {file_size:.2f} KB")

        # Check if file contains expected content
        with open(report_path, "r") as f:
            content = f.read()
            checks = [
                ("Title" in content or "Amazon Rainforest" in content, "Area name"),
                (
                    "executive_summary" in content or "Executive Summary" in content,
                    "Executive summary",
                ),
                ("table" in content or "Year" in content, "Data table"),
                ("Forest" in content, "Forest analysis"),
                ("Urban" in content, "Urban analysis"),
            ]

            print("\n2. Content validation:")
            for check, desc in checks:
                status = "✓" if check else "✗"
                print(f"   {status} {desc}")

    except ReportAssemblerError as e:
        print(f"   ✗ Error: {e}")

    # Test LangGraph node
    print("\n3. LangGraph Node Test:")
    test_state = {
        "data_df": sample_data,
        "narrative_json": sample_narrative,
        "area_name": "Amazon Rainforest",
        "time_period": {"start": "2019", "end": "2023"},
        "filmstrip_path": sample_filmstrip,
    }

    result_state = node_4_report_assembler(test_state)

    if result_state.get("error"):
        print(f"   ✗ ERROR: {result_state['error']}")
    else:
        print("   ✓ SUCCESS! Report generated:")
        print(f"   ✓ Path: {result_state['html_path']}")
        print(f"   ✓ File exists: {os.path.exists(result_state['html_path'])}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - Test Report: /tmp/test_report.html")
    print(f"  - Node Output: {result_state.get('html_path', 'N/A')}")
    print("\nOpen the HTML file in a browser to view the report!")
