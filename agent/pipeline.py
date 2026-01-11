"""
Pipeline Module for Satellite Analysis Agent
Phase 5: LangGraph orchestration connecting all nodes

Author: Hackathon Team
Timeline: H 9-12
"""

import os
from typing import Any, Dict, Optional, TypedDict

import pandas as pd

# Try to import LangGraph, but provide fallback
try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print(
        "⚠️  LangGraph not installed. Install with: pip install langgraph --break-system-packages"
    )

# Import our custom nodes
from agent.data_formatter import node_1_data_formatter
from agent.filmstrip_generator import node_2_filmstrip_generator
from agent.narrative_agent import node_3_narrative_agent
from agent.pdf_publisher import node_5_publisher
from agent.report_assembler import node_4_report_assembler


class AgentState(TypedDict):
    """
    State schema for the Satellite Analysis Agent pipeline.
    This matches the schema defined in the implementation plan.
    """

    # --- INPUTS ---
    data_df: pd.DataFrame  # Pre-calculated YoY DataFrame
    area_name: str  # Geographic area name
    time_period: Dict[str, str]  # {'start': 'YYYY', 'end': 'YYYY'}
    gif_path: str  # Path to satellite GIF

    # --- INTERNAL STATE ---
    data_context_str: str  # Stringified data for LLM
    filmstrip_path: str  # Path to generated filmstrip PNG

    # --- GENERATION (LLM) ---
    narrative_json: Dict[str, str]  # AI-generated analysis

    # --- OUTPUTS ---
    html_path: str  # Path to HTML report
    pdf_path: Optional[str]  # Path to PDF report (optional)
    error: Optional[str]  # Error message if any
    pdf_error: Optional[str]  # PDF-specific error


class SatelliteAnalysisPipeline:
    """
    Main pipeline class that orchestrates the satellite analysis workflow.
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Initialize the pipeline.

        Args:
            gemini_api_key: Optional Gemini API key (uses env var if not provided)
        """
        self.gemini_api_key = gemini_api_key
        self.graph = None

        if LANGGRAPH_AVAILABLE:
            self._build_graph()

    def _build_graph(self):
        """
        Builds the LangGraph workflow connecting all 5 nodes.
        """
        # Create workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("data_formatter", node_1_data_formatter)
        workflow.add_node("filmstrip_generator", node_2_filmstrip_generator)
        workflow.add_node("narrative_agent", self._narrative_agent_wrapper)
        workflow.add_node("report_assembler", node_4_report_assembler)
        workflow.add_node("publisher", node_5_publisher)

        # Define edges (linear pipeline)
        workflow.set_entry_point("data_formatter")
        workflow.add_edge("data_formatter", "filmstrip_generator")
        workflow.add_edge("filmstrip_generator", "narrative_agent")
        workflow.add_edge("narrative_agent", "report_assembler")
        workflow.add_edge("report_assembler", "publisher")
        workflow.add_edge("publisher", END)

        # Compile graph
        self.graph = workflow.compile()

    def _narrative_agent_wrapper(self, state: dict) -> dict:
        """
        Wrapper for narrative agent to pass API key.
        """
        return node_3_narrative_agent(state, api_key=self.gemini_api_key)

    def run(
        self,
        data_df: pd.DataFrame,
        area_name: str,
        gif_path: str,
        time_period: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the complete pipeline.

        Args:
            data_df: Pre-calculated YoY DataFrame
            area_name: Name of the geographic area
            gif_path: Path to satellite timelapse GIF
            time_period: Optional dict with 'start' and 'end' years

        Returns:
            Final state dictionary with all outputs

        Example:
            >>> pipeline = SatelliteAnalysisPipeline()
            >>> result = pipeline.run(
            ...     data_df=df,
            ...     area_name="Amazon Basin",
            ...     gif_path="satellite.gif",
            ...     time_period={'start': '2019', 'end': '2023'}
            ... )
            >>> print(f"HTML: {result['html_path']}")
            >>> print(f"PDF: {result['pdf_path']}")
        """
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError(
                "LangGraph is not installed. Cannot run pipeline.\n"
                "Install with: pip install langgraph --break-system-packages"
            )

        # Initialize state
        initial_state: AgentState = {
            "data_df": data_df,
            "area_name": area_name,
            "gif_path": gif_path,
            "time_period": time_period or {},
            "data_context_str": "",
            "filmstrip_path": "",
            "narrative_json": {},
            "html_path": "",
            "pdf_path": None,
            "error": None,
            "pdf_error": None,
        }

        # Execute pipeline
        final_state = self.graph.invoke(initial_state)

        return final_state

    def run_sequential(
        self,
        data_df: pd.DataFrame,
        area_name: str,
        gif_path: str,
        time_period: Optional[Dict[str, str]] = None,
        generate_pdf: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes the pipeline sequentially without LangGraph (fallback mode).
        Useful when LangGraph is not available.

        Args:
            data_df: Pre-calculated YoY DataFrame
            area_name: Name of the geographic area
            gif_path: Path to satellite timelapse GIF
            time_period: Optional dict with 'start' and 'end' years
            generate_pdf: Whether to generate PDF report


        Returns:
            Final state dictionary with all outputs
        """
        # Initialize state
        state: AgentState = {
            "data_df": data_df,
            "area_name": area_name,
            "gif_path": gif_path,
            "time_period": time_period or {},
            "data_context_str": "",
            "filmstrip_path": "",
            "narrative_json": {},
            "html_path": "",
            "pdf_path": None,
            "error": None,
            "pdf_error": None,
        }

        print("🚀 Starting Sequential Pipeline...\n")

        # Node 1: Data Formatter
        print("📊 [1/5] Running Data Formatter...")
        state = node_1_data_formatter(state)
        if state.get("error"):
            print(f"   ✗ Error: {state['error']}")
            return state
        print(f"   ✓ Data formatted ({len(state['data_context_str'])} chars)")

        # Node 2: Filmstrip Generator
        print("\n🎬 [2/5] Running Filmstrip Generator...")
        state = node_2_filmstrip_generator(state)
        if state.get("error"):
            print(f"   ✗ Error: {state['error']}")
            return state
        print(f"   ✓ Filmstrip created: {state['filmstrip_path']}")

        # Node 3: Narrative Agent
        print("\n🤖 [3/5] Running Narrative Agent (Gemini)...")
        state = node_3_narrative_agent(state, api_key=self.gemini_api_key)
        if state.get("error"):
            print(f"   ✗ Error: {state['error']}")
            return state
        print(f"   ✓ Narrative generated ({len(state['narrative_json'])} sections)")

        # Node 4: Report Assembler
        print("\n📄 [4/5] Running Report Assembler...")
        state = node_4_report_assembler(state)
        if state.get("error"):
            print(f"   ✗ Error: {state['error']}")
            return state
        file_size = os.path.getsize(state["html_path"]) / 1024
        print(f"   ✓ HTML report created: {state['html_path']} ({file_size:.2f} KB)")

        # Node 5: Publisher
        print("\n📦 [5/5] Running Publisher...")
        state = node_5_publisher(state, generate_pdf_output=generate_pdf)
        if state.get("error"):
            print(f"   ✗ Error: {state['error']}")
            return state
        if state.get("pdf_path"):
            pdf_size = os.path.getsize(state["pdf_path"]) / 1024
            print(f"   ✓ PDF report created: {state['pdf_path']} ({pdf_size:.2f} KB)")
        elif state.get("pdf_error"):
            print(f"   ⚠️  PDF generation skipped: {state['pdf_error']}")
        else:
            print("   ℹ️  PDF generation not attempted")

        print("\n✅ Pipeline Complete!")
        return state


def run_pipeline_simple(
    data_df: pd.DataFrame,
    area_name: str,
    gif_path: str,
    time_period: Optional[Dict[str, str]] = None,
    gemini_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Simple function to run the pipeline without instantiating the class.

    Args:
        data_df: Pre-calculated YoY DataFrame
        area_name: Name of the geographic area
        gif_path: Path to satellite timelapse GIF
        time_period: Optional dict with 'start' and 'end' years
        gemini_api_key: Optional Gemini API key

    Returns:
        Final state dictionary with outputs
    """
    pipeline = SatelliteAnalysisPipeline(gemini_api_key=gemini_api_key)

    if LANGGRAPH_AVAILABLE:
        return pipeline.run(data_df, area_name, gif_path, time_period)
    else:
        return pipeline.run_sequential(data_df, area_name, gif_path, time_period)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("SATELLITE ANALYSIS PIPELINE TEST")
    print("=" * 60)

    # Sample data
    sample_data = pd.DataFrame(
        {
            "year": [2019, 2020, 2021, 2022, 2023],
            "forest_pct_change": [-1.2, -2.1, -1.5, -2.8, -3.1],
            "urban_pct_change": [3.5, 5.0, 4.2, 6.1, 5.8],
            "water_pct_change": [0.1, -0.2, 0.3, -0.1, 0.2],
        }
    )

    # Check if test GIF exists
    test_gif = "/tmp/test_satellite.gif"

    if not os.path.exists(test_gif):
        print("\n⚠️  Creating test GIF...")
        from PIL import Image, ImageDraw, ImageFont

        frames = []
        colors = [
            (50, 100, 50),
            (80, 130, 80),
            (110, 160, 110),
            (140, 140, 140),
            (170, 170, 170),
        ]

        for i, color in enumerate(colors):
            frame = Image.new("RGB", (400, 300), color)
            draw = ImageDraw.Draw(frame)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40
                )
            except:
                font = ImageFont.load_default()
            draw.text((150, 130), f"Year {2019 + i}", fill="white", font=font)
            frames.append(frame)

        frames[0].save(
            test_gif, save_all=True, append_images=frames[1:], duration=500, loop=0
        )
        print(f"   ✓ Test GIF created: {test_gif}")

    # Run pipeline
    print("\n" + "=" * 60)
    print("RUNNING FULL PIPELINE (Sequential Mode)")
    print("=" * 60 + "\n")

    pipeline = SatelliteAnalysisPipeline()

    result = pipeline.run_sequential(
        data_df=sample_data,
        area_name="Amazon Rainforest",
        gif_path=test_gif,
        time_period={"start": "2019", "end": "2023"},
    )

    # Display results
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)

    if result.get("error"):
        print("\n❌ Pipeline failed with error:")
        print(f"   {result['error']}")
    else:
        print("\n✅ Pipeline completed successfully!")
        print("\n📁 Output Files:")
        print(f"   • HTML Report: {result.get('html_path', 'N/A')}")
        if result.get("pdf_path"):
            print(f"   • PDF Report:  {result['pdf_path']}")

        print("\n📊 Generated Sections:")
        if result.get("narrative_json"):
            for key in result["narrative_json"].keys():
                print(f"   • {key.replace('_', ' ').title()}")

        if result.get("pdf_error"):
            print(f"\n⚠️  PDF Note: {result['pdf_error']}")

    print("\n" + "=" * 60)
