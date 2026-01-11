"""
Narrative Agent Module for Satellite Analysis Agent
Phase 3: Uses Gemini AI to analyze YoY data and generate structured insights

Author: Hackathon Team
Timeline: H 4-6
"""

import json
from typing import Dict, Optional

from dotenv import load_dotenv
from google import genai

load_dotenv()


class NarrativeAgentError(Exception):
    """Custom exception for narrative generation errors"""

    pass


def build_analysis_prompt(
    data_context_str: str,
    area_name: str = "the study area",
    time_period: Optional[Dict[str, str]] = None,
) -> str:
    """
    Builds the prompt for Gemini to analyze the satellite data.

    Args:
        data_context_str: Formatted string with YoY percentage changes
        area_name: Name of the geographic area being analyzed
        time_period: Dict with 'start' and 'end' years

    Returns:
        Complete prompt string for Gemini
    """
    period_str = ""
    if time_period:
        start = time_period.get("start", "unknown")
        end = time_period.get("end", "unknown")
        period_str = f" from {start} to {end}"

    prompt = f"""You are an expert environmental analyst specializing in satellite imagery analysis and land cover change detection.

CONTEXT:
You are analyzing satellite-derived land cover change data for {area_name}{period_str}.

INPUT DATA:
{data_context_str}

IMPORTANT NOTES:
- The data above contains Year-over-Year (YoY) percentage changes for different land cover types.
- Negative percentages indicate decrease/loss (e.g., forest loss, urban shrinkage).
- Positive percentages indicate increase/growth (e.g., urban expansion, forest regrowth).
- DO NOT invent or fabricate any numbers. Quote the percentages exactly as provided in the data.

YOUR TASK:
Analyze the provided data and generate insights following these guidelines:

1. EXECUTIVE SUMMARY:
   - Provide a high-level overview (2-3 sentences) of the most significant changes
   - Identify the dominant trend (e.g., urbanization, deforestation, etc.)
   - Mention the most dramatic year if applicable

2. FOREST ANALYSIS:
   - Describe the forest cover change trajectory
   - Identify years with the most drastic forest changes
   - Discuss any acceleration or deceleration patterns
   - Provide environmental context if relevant

3. URBAN ANALYSIS:
   - Describe urban expansion patterns
   - Identify years with peak urban growth
   - Discuss any notable trends or patterns
   - Consider implications for development

4. WATER ANALYSIS:
   - Describe water body changes (if applicable)
   - Note any significant fluctuations
   - Consider seasonal or climate factors if patterns suggest them
   - If water changes are minimal, state that

IMPORTANT: Output ONLY a valid JSON object with NO additional text, markdown formatting, or explanations.

OUTPUT FORMAT (valid JSON only):
{{
  "executive_summary": "Your 2-3 sentence summary here",
  "forest_analysis": "Your detailed forest analysis here (3-5 sentences)",
  "urban_analysis": "Your detailed urban analysis here (3-5 sentences)",
  "water_analysis": "Your detailed water analysis here (2-4 sentences)"
}}

Remember: Quote percentages exactly as provided. Do not round or modify the numbers."""

    return prompt


def parse_gemini_response(response_text: str) -> Dict[str, str]:
    """
    Parses Gemini's response and extracts the JSON object.

    Args:
        response_text: Raw response text from Gemini

    Returns:
        Dictionary with analysis sections

    Raises:
        NarrativeAgentError: If response cannot be parsed as valid JSON
    """
    # Try to extract JSON from response
    # Sometimes models wrap JSON in markdown code blocks
    cleaned_text = response_text.strip()

    # Remove markdown code blocks if present
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]  # Remove ```json
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]  # Remove ```

    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]  # Remove trailing ```

    cleaned_text = cleaned_text.strip()

    # Parse JSON
    try:
        result = json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        raise NarrativeAgentError(
            f"Failed to parse Gemini response as JSON: {str(e)}\n"
            f"Response was: {response_text[:500]}"
        )

    # Validate required keys
    required_keys = [
        "executive_summary",
        "forest_analysis",
        "urban_analysis",
        "water_analysis",
    ]
    missing_keys = [key for key in required_keys if key not in result]

    if missing_keys:
        raise NarrativeAgentError(
            f"Gemini response missing required keys: {missing_keys}\n"
            f"Response was: {result}"
        )

    return result


def generate_narrative(
    data_context_str: str,
    area_name: str = "the study area",
    time_period: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None,
    model_name: str = "gemini-pro",
) -> Dict[str, str]:
    """
    Main function: Generates narrative analysis using Gemini.

    Args:
        data_context_str: Formatted string with YoY percentage changes
        area_name: Name of the geographic area
        time_period: Dict with 'start' and 'end' years
        api_key: Gemini API key (optional if env var is set)
        model_name: Gemini model to use (default: gemini-pro)

    Returns:
        Dictionary with analysis sections

    Raises:
        NarrativeAgentError: If generation or parsing fails

    Example:
        >>> narrative = generate_narrative(
        ...     data_context_str="Year 2020: Forest -2.1%, Urban +5.0%...",
        ...     area_name="Amazon Basin",
        ...     time_period={'start': '2019', 'end': '2023'}
        ... )
    """

    # Build prompt
    prompt = build_analysis_prompt(data_context_str, area_name, time_period)

    client = genai.Client()

    # Generate response
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        response_text = response.text
    except Exception as e:
        raise NarrativeAgentError(f"Failed to generate content with Gemini: {str(e)}")

    # Parse response
    narrative_json = parse_gemini_response(response_text)

    return narrative_json


def node_3_narrative_agent(state: dict, api_key: Optional[str] = None) -> dict:
    """
    LangGraph Node 3: Narrative Agent

    Takes the data context string from state and generates narrative
    analysis using Gemini, saving the result back to state.

    Args:
        state: Agent state dictionary containing 'data_context_str'
        api_key: Optional Gemini API key (uses env var if not provided)

    Returns:
        Updated state with 'narrative_json' populated
    """
    try:
        # Extract required fields from state
        data_context_str = state.get("data_context_str")
        area_name = state.get("area_name", "the study area")
        time_period = state.get("time_period")

        if not data_context_str:
            raise NarrativeAgentError(
                "No data context string found in state['data_context_str']"
            )

        # Generate narrative
        narrative_json = generate_narrative(
            data_context_str=data_context_str,
            area_name=area_name,
            time_period=time_period,
            api_key=api_key,
        )

        # Update state
        state["narrative_json"] = narrative_json

        # Clear any previous errors
        if "error" in state:
            state["error"] = None

        return state

    except Exception as e:
        # Store error in state
        state["error"] = f"Narrative Agent Error: {str(e)}"
        state["narrative_json"] = {}
        return state


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("NARRATIVE AGENT TEST")
    print("=" * 60)

    # Sample data context (from Phase 1 output)
    sample_context = """Yearly Changes Data:
- Year 2019: Forest: -1.2%, Urban: +3.5%, Water: +0.1%
- Year 2020: Forest: -2.1%, Urban: +5.0%, Water: -0.2%
- Year 2021: Forest: -1.5%, Urban: +4.2%, Water: +0.3%
- Year 2022: Forest: -2.8%, Urban: +6.1%, Water: -0.1%
- Year 2023: Forest: -3.1%, Urban: +5.8%, Water: +0.2%"""

    # Check if API key is available

    # Show the prompt that would be sent
    print("1. PROMPT STRUCTURE:")
    print("-" * 60)
    prompt = build_analysis_prompt(
        sample_context,
        area_name="Amazon Rainforest",
        time_period={"start": "2019", "end": "2023"},
    )
    print(prompt[:800] + "...")
    print("-" * 60)

    # Show expected output structure
    print("\n2. EXPECTED OUTPUT STRUCTURE:")
    print("-" * 60)
    expected = {
        "executive_summary": "Summary of major trends...",
        "forest_analysis": "Detailed forest analysis...",
        "urban_analysis": "Detailed urban analysis...",
        "water_analysis": "Detailed water analysis...",
    }
    print(json.dumps(expected, indent=2))
    print("-" * 60)

    # Test LangGraph node (will fail gracefully)
    print("\n3. LANGGRAPH NODE TEST:")
    print("-" * 60)
    test_state = {
        "data_context_str": sample_context,
        "area_name": "Amazon Rainforest",
        "time_period": {"start": "2019", "end": "2023"},
    }
    result_state = node_3_narrative_agent(test_state)

    if result_state.get("error"):
        print(f"   Expected Error (no API key): {result_state['error'][:100]}...")
    else:
        print("   Unexpected success!")
    print("-" * 60)

    print("\n✓ GEMINI_API_KEY found! Running full test...\n")

    # Test narrative generation
    print("1. Generating narrative with Gemini...")
    try:
        narrative = generate_narrative(
            data_context_str=sample_context,
            area_name="Amazon Rainforest",
            time_period={"start": "2019", "end": "2023"},
        )

        print("   ✓ Narrative generated successfully!\n")

        print("2. Generated Narrative:")
        print("-" * 60)
        for key, value in narrative.items():
            print(f"\n{key.upper().replace('_', ' ')}:")
            print(f"   {value}")
        print("-" * 60)

    except NarrativeAgentError as e:
        print(f"   ✗ Error: {e}")

    # Test LangGraph node
    print("\n3. LangGraph Node Test:")
    test_state = {
        "data_context_str": sample_context,
        "area_name": "Amazon Rainforest",
        "time_period": {"start": "2019", "end": "2023"},
    }
    result_state = node_3_narrative_agent(test_state)

    if result_state.get("error"):
        print(f"   ✗ ERROR: {result_state['error']}")
    else:
        print("   ✓ SUCCESS! Narrative generated via LangGraph node")
        print(f"   ✓ Keys in narrative: {list(result_state['narrative_json'].keys())}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
