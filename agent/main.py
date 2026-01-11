#!/usr/bin/env python3
"""
Satellite Analysis Agent - Main Entry Point
Easy-to-use interface for running the complete analysis pipeline

Author: Hackathon Team
Usage: python main.py --data data.csv --gif satellite.gif --area "Amazon Basin"
"""

import argparse
import os
import sys

import pandas as pd
from agent.pipeline import SatelliteAnalysisPipeline


def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from CSV or JSON file.

    Args:
        data_path: Path to data file (.csv or .json)

    Returns:
        DataFrame with YoY data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    file_ext = os.path.splitext(data_path)[1].lower()

    if file_ext == ".csv":
        df = pd.read_csv(data_path)
    elif file_ext == ".json":
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .csv or .json")

    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Validates that the DataFrame has the expected structure.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If validation fails
    """
    if "year" not in df.columns:
        raise ValueError("DataFrame must have a 'year' column")

    pct_columns = [col for col in df.columns if "_pct" in col or "pct_change" in col]
    if not pct_columns:
        raise ValueError(
            "DataFrame must have at least one percentage change column "
            "(e.g., 'forest_pct', 'urban_pct_change')"
        )


def print_banner():
    """Prints the application banner."""
    print("\n" + "=" * 70)
    print(" " * 15 + "🛰️  SATELLITE ANALYSIS AGENT 🛰️")
    print(" " * 10 + "Land Cover Change Detection & Reporting")
    print("=" * 70 + "\n")


def print_summary(result: dict):
    """
    Prints a summary of the pipeline results.

    Args:
        result: Final state dictionary from pipeline
    """
    print("\n" + "=" * 70)
    print("📋 EXECUTION SUMMARY")
    print("=" * 70)

    if result.get("error"):
        print("\n❌ Pipeline Failed")
        print(f"   Error: {result['error']}")
        return

    print("\n✅ Pipeline Completed Successfully")
    print(f"\n📍 Area: {result.get('area_name', 'N/A')}")

    if result.get("time_period"):
        period = result["time_period"]
        print(f"📅 Period: {period.get('start', '?')} - {period.get('end', '?')}")

    print("\n📁 Output Files:")

    # HTML Report
    html_path = result.get("html_path")
    if html_path and os.path.exists(html_path):
        size_kb = os.path.getsize(html_path) / 1024
        print(f"   ✓ HTML Report: {html_path} ({size_kb:.2f} KB)")
    else:
        print("   ✗ HTML Report: Not generated")

    # PDF Report
    pdf_path = result.get("pdf_path")
    if pdf_path and os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"   ✓ PDF Report:  {pdf_path} ({size_kb:.2f} KB)")
    elif result.get("pdf_error"):
        print(f"   ℹ️  PDF Report:  {result['pdf_error']}")
    else:
        print("   - PDF Report:  Not requested")

    # Filmstrip
    filmstrip_path = result.get("filmstrip_path")
    if filmstrip_path and os.path.exists(filmstrip_path):
        print(f"   ✓ Filmstrip:   {filmstrip_path}")

    # Analysis Sections
    narrative = result.get("narrative_json", {})
    if narrative:
        print("\n📊 Analysis Sections Generated:")
        for key in narrative.keys():
            section_name = key.replace("_", " ").title()
            word_count = len(narrative[key].split())
            print(f"   • {section_name} ({word_count} words)")

    print("\n" + "=" * 70 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Satellite Analysis Agent - Automated land cover change analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --data changes.csv --gif timelapse.gif --area "Amazon Basin"
  
  # With time period
  python main.py --data changes.csv --gif timelapse.gif --area "Amazon" \\
                 --start 2019 --end 2023
  
  # With custom API key
  python main.py --data changes.csv --gif timelapse.gif --area "Amazon" \\
                 --api-key YOUR_GEMINI_KEY

Data Format:
  Your CSV/JSON must have columns like:
  - year (required)
  - forest_pct_change or forest_pct (required)
  - urban_pct_change or urban_pct (required)
  - water_pct_change or water_pct (optional)
        """,
    )

    # Required arguments
    parser.add_argument(
        "--data",
        "-d",
        required=True,
        help="Path to data file (.csv or .json) with YoY changes",
    )
    parser.add_argument(
        "--gif", "-g", required=True, help="Path to satellite timelapse GIF"
    )
    parser.add_argument(
        "--area", "-a", required=True, help="Name of the geographic area"
    )

    # Optional arguments
    parser.add_argument("--start", "-s", type=str, help="Start year of analysis period")
    parser.add_argument("--end", "-e", type=str, help="End year of analysis period")
    parser.add_argument(
        "--api-key",
        "-k",
        type=str,
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="/tmp",
        help="Output directory for reports (default: /tmp)",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation")

    args = parser.parse_args()

    # Print banner
    print_banner()

    try:
        # Load and validate data
        print("📂 Loading data...")
        df = load_data(args.data)
        print(f"   ✓ Loaded {len(df)} rows from {args.data}")

        validate_data(df)
        print("   ✓ Data validated")

        # Validate GIF
        if not os.path.exists(args.gif):
            raise FileNotFoundError(f"GIF file not found: {args.gif}")
        print(f"   ✓ GIF found: {args.gif}")

        # Prepare time period
        time_period = None
        if args.start and args.end:
            time_period = {"start": args.start, "end": args.end}

        # Get API key
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("\n⚠️  Warning: No Gemini API key provided")
            print("   Set GEMINI_API_KEY env var or use --api-key flag")
            print("   Pipeline will fail at the narrative generation step")

        # Create pipeline
        print("\n🔧 Initializing pipeline...")
        pipeline = SatelliteAnalysisPipeline(gemini_api_key=api_key)

        # Run pipeline
        print("\n🚀 Running analysis pipeline...\n")
        print("-" * 70)

        result = pipeline.run_sequential(
            data_df=df, area_name=args.area, gif_path=args.gif, time_period=time_period
        )

        print("-" * 70)

        # Print summary
        print_summary(result)

        # Exit code based on success
        if result.get("error"):
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ Fatal Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
