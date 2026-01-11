"""
Data Formatter Module for Satellite Analysis Agent
Phase 1: Converts pre-calculated YoY DataFrame into LLM-ready string context
"""

import pandas as pd
from typing import Optional, List


class DataFormatterError(Exception):
    """Custom exception for data formatting errors"""
    pass


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes column names.
    
    Args:
        df: Input DataFrame with potentially messy column names
        
    Returns:
        DataFrame with cleaned column names
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Convert to lowercase and replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    return df_clean


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validates that the DataFrame has the required structure.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        DataFormatterError: If validation fails
    """
    if df is None or df.empty:
        raise DataFormatterError("DataFrame is None or empty")
    
    # Check for 'year' column
    if 'year' not in df.columns:
        raise DataFormatterError(
            f"Missing 'year' column. Available columns: {list(df.columns)}"
        )
    
    # Check for at least one percentage change column
    pct_columns = [col for col in df.columns if '_pct' in col or 'pct_change' in col]
    if not pct_columns:
        raise DataFormatterError(
            "No percentage change columns found. Expected columns like 'forest_pct', 'urban_pct', etc."
        )


def format_data_for_llm(
    df: pd.DataFrame,
    include_header: bool = True,
    round_decimals: int = 2
) -> str:
    """
    Converts the DataFrame (Year + % Change) into a string context for Gemini.
    
    This function takes pre-calculated YoY percentage changes and formats them
    into a readable string that the LLM can analyze for trends and insights.
    
    Args:
        df: DataFrame with columns like 'year', 'forest_pct', 'urban_pct', 'water_pct'
        include_header: Whether to include a header in the output
        round_decimals: Number of decimal places for percentage values
        
    Returns:
        Formatted string with yearly changes data
        
    Raises:
        DataFormatterError: If DataFrame validation fails
        
    Example:
        >>> df = pd.DataFrame({
        ...     'year': [2020, 2021],
        ...     'forest_pct': [-2.1, -1.5],
        ...     'urban_pct': [5.0, 4.2]
        ... })
        >>> print(format_data_for_llm(df))
        Yearly Changes Data:
        - Year 2020: Forest -2.1%, Urban 5.0%
        - Year 2021: Forest -1.5%, Urban 4.2%
    """
    # Clean column names
    df = clean_column_names(df)
    
    # Validate DataFrame structure
    validate_dataframe(df)
    
    # Identify percentage change columns (excluding 'year')
    pct_columns = [col for col in df.columns if col != 'year' and ('_pct' in col or 'pct_change' in col)]
    
    # Sort columns for consistent output
    pct_columns.sort()
    
    # Build the context string
    context = ""
    if include_header:
        context = "Yearly Changes Data:\n"
    
    # Iterate through each year
    for index, row in df.iterrows():
        year = row['year']
        context += f"- Year {year}: "
        
        # Add each percentage change
        change_parts = []
        for col in pct_columns:
            # Extract the land cover type from column name
            # e.g., 'forest_pct' -> 'Forest', 'urban_pct_change' -> 'Urban'
            land_type = col.replace('_pct', '').replace('_change', '').replace('_', ' ').title()
            
            # Get the percentage value and format it
            pct_value = row[col]
            
            # Handle NaN values
            if pd.isna(pct_value):
                change_parts.append(f"{land_type}: N/A")
            else:
                # Round to specified decimals
                pct_value = round(pct_value, round_decimals)
                
                # Format with sign
                if pct_value > 0:
                    change_parts.append(f"{land_type}: +{pct_value}%")
                else:
                    change_parts.append(f"{land_type}: {pct_value}%")
        
        # Join all changes for this year
        context += ", ".join(change_parts) + "\n"
    
    return context


def format_data_summary(df: pd.DataFrame) -> str:
    """
    Creates a summary of the data range and available metrics.
    
    Args:
        df: DataFrame with yearly data
        
    Returns:
        Summary string describing the dataset
    """
    df = clean_column_names(df)
    validate_dataframe(df)
    
    years = sorted(df['year'].unique())
    pct_columns = [col for col in df.columns if col != 'year' and ('_pct' in col or 'pct_change' in col)]
    
    # Extract land cover types
    land_types = [
        col.replace('_pct', '').replace('_change', '').replace('_', ' ').title()
        for col in pct_columns
    ]
    
    summary = f"Data Range: {years[0]} to {years[-1]} ({len(years)} years)\n"
    summary += f"Metrics Tracked: {', '.join(land_types)}\n"
    summary += f"Total Data Points: {len(df)} records"
    
    return summary


def node_1_data_formatter(state: dict) -> dict:
    """
    LangGraph Node 1: Data Formatter
    
    Takes the pre-calculated DataFrame from state and converts it to
    an LLM-ready string context.
    
    Args:
        state: Agent state dictionary containing 'data_df'
        
    Returns:
        Updated state with 'data_context_str' populated
    """
    try:
        df = state.get('data_df')
        
        if df is None:
            raise DataFormatterError("No DataFrame found in state['data_df']")
        
        # Format the data for LLM
        data_context_str = format_data_for_llm(df)
        
        # Update state
        state['data_context_str'] = data_context_str
        
        # Clear any previous errors
        if 'error' in state:
            state['error'] = None
            
        return state
        
    except Exception as e:
        # Store error in state
        state['error'] = f"Data Formatter Error: {str(e)}"
        state['data_context_str'] = ""
        return state


# Example usage and testing
if __name__ == "__main__":
    # Create sample DataFrame for testing
    sample_data = pd.DataFrame({
        'year': [2019, 2020, 2021, 2022, 2023],
        'forest_pct_change': [-1.2, -2.1, -1.5, -2.8, -3.1],
        'urban_pct_change': [3.5, 5.0, 4.2, 6.1, 5.8],
        'water_pct_change': [0.1, -0.2, 0.3, -0.1, 0.2]
    })
    
    print("=" * 60)
    print("DATA FORMATTER TEST")
    print("=" * 60)
    
    # Test data summary
    print("\n1. Data Summary:")
    print(format_data_summary(sample_data))
    
    # Test LLM formatting
    print("\n2. LLM Context String:")
    print(format_data_for_llm(sample_data))
    
    # Test LangGraph node
    print("\n3. LangGraph Node Test:")
    test_state = {'data_df': sample_data}
    result_state = node_1_data_formatter(test_state)
    
    if result_state.get('error'):
        print(f"ERROR: {result_state['error']}")
    else:
        print("SUCCESS! Context generated:")
        print(result_state['data_context_str'])
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)