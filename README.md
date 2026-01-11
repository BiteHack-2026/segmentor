# Land Cover Change Tracker

Interactive web application for analyzing historical land cover changes using MODIS satellite data and AI-powered insights.

## Features

- Interactive map interface for location selection
- 20+ years of MODIS land cover data (2001-2022)
- Satellite imagery visualization (Sentinel-2 & Landsat 8)
- Year-over-year change analysis
- AI-generated reports using Google Gemini
- Export to CSV, GIF, HTML, and PDF

## Requirements

- Python 3.12 or higher
- Google Earth Engine account
- Google Gemini API key (for report generation)

## Installation

Install dependencies using uv:

```bash
uv sync
```

Or using pip:

```bash
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```
GEE_PROJECT=your-gee-project-id
GEMINI_API_KEY=your-gemini-api-key
```

Authenticate with Google Earth Engine (first time only):

```bash
earthengine authenticate
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Or with uv:

```bash
uv run streamlit run app.py
```

The application will be available at `http://localhost:8501`

## How to Use

1. Select a location on the map or choose from hotspot locations
2. Choose years to analyze (minimum 2 years)
3. Click "Analyze Historical Changes"
4. View visualizations and statistics
5. Export data or generate AI-powered reports

## Data Sources

- **Land Cover**: MODIS MCD12Q1 (500m resolution, IGBP classification)
- **Satellite Imagery**: Sentinel-2 SR (10m) and Landsat 8 (30m)
- **Analysis**: Google Earth Engine

## Technology Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly, Folium, Matplotlib
- **Data Processing**: NumPy, Pandas
- **AI**: Google Gemini (via google-genai)
- **Workflow**: LangGraph
- **Satellite Data**: Google Earth Engine

## License

MIT
