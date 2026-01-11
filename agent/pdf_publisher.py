"""
PDF Publisher Module for Satellite Analysis Agent
Phase 5: Converts HTML reports to PDF format using WeasyPrint

Author: Hackathon Team
Timeline: H 9-12
"""

import os
from typing import Optional

# Try to import WeasyPrint, but don't fail if not available
# On Windows, WeasyPrint may be installed but GTK libraries missing
try:
    from weasyprint import CSS, HTML

    WEASYPRINT_AVAILABLE = True
    WEASYPRINT_ERROR = None
except ImportError:
    WEASYPRINT_AVAILABLE = False
    WEASYPRINT_ERROR = "WeasyPrint not installed"
except OSError as e:
    # This catches missing GTK/Pango libraries on Windows
    WEASYPRINT_AVAILABLE = False
    WEASYPRINT_ERROR = f"WeasyPrint GTK libraries missing: {str(e)[:100]}"


class PDFPublisherError(Exception):
    """Custom exception for PDF publishing errors"""

    pass


def check_weasyprint() -> bool:
    """
    Checks if WeasyPrint is available and properly installed.

    Returns:
        True if WeasyPrint is available, False otherwise
    """
    return WEASYPRINT_AVAILABLE


def generate_pdf(
    html_path: str, output_path: str, custom_css: Optional[str] = None
) -> str:
    """
    Converts an HTML file to PDF using WeasyPrint.

    Args:
        html_path: Path to the input HTML file
        output_path: Path to save the output PDF
        custom_css: Optional custom CSS for PDF styling

    Returns:
        Path to the generated PDF file

    Raises:
        PDFPublisherError: If PDF generation fails

    Example:
        >>> pdf_path = generate_pdf(
        ...     html_path="report.html",
        ...     output_path="report.pdf"
        ... )
    """
    if not WEASYPRINT_AVAILABLE:
        raise PDFPublisherError(
            "WeasyPrint is not installed. Install with: "
            "pip install weasyprint --break-system-packages\n"
            "Note: WeasyPrint also requires system dependencies. On Ubuntu: "
            "apt-get install python3-cffi python3-brotli libpango-1.0-0 libpangoft2-1.0-0"
        )

    # Validate input file
    if not os.path.exists(html_path):
        raise PDFPublisherError(f"HTML file not found: {html_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Load HTML
        html = HTML(filename=html_path)

        # Generate PDF
        if custom_css:
            css = CSS(string=custom_css)
            html.write_pdf(output_path, stylesheets=[css])
        else:
            html.write_pdf(output_path)

        return output_path

    except Exception as e:
        raise PDFPublisherError(f"Failed to generate PDF: {str(e)}")


def create_pdf_with_fallback(
    html_path: str, output_path: str, custom_css: Optional[str] = None
) -> Optional[str]:
    """
    Attempts to create PDF, returns None if WeasyPrint is unavailable.
    This is a graceful fallback version that doesn't raise exceptions.

    Args:
        html_path: Path to the input HTML file
        output_path: Path to save the output PDF
        custom_css: Optional custom CSS for PDF styling

    Returns:
        Path to PDF if successful, None if WeasyPrint unavailable
    """
    if not WEASYPRINT_AVAILABLE:
        return None

    try:
        return generate_pdf(html_path, output_path, custom_css)
    except PDFPublisherError:
        return None


def node_5_publisher(state: dict, generate_pdf_output: bool = True) -> dict:
    """
    LangGraph Node 5: Publisher

    Takes the HTML report from state and optionally generates a PDF version.

    Args:
        state: Agent state dictionary containing 'html_path'
        generate_pdf_output: Whether to attempt PDF generation

    Returns:
        Updated state with 'pdf_path' populated (if successful)
    """
    try:
        html_path = state.get("html_path")

        if not html_path:
            raise PDFPublisherError("No HTML path found in state['html_path']")

        if not os.path.exists(html_path):
            raise PDFPublisherError(f"HTML file not found: {html_path}")

        # Initialize pdf_path as None
        pdf_path = None

        # Attempt PDF generation if requested
        if generate_pdf_output:
            if WEASYPRINT_AVAILABLE:
                # Generate PDF output path
                pdf_path = html_path.replace(".html", ".pdf")

                try:
                    pdf_path = generate_pdf(html_path, pdf_path)
                except PDFPublisherError as e:
                    # Log error but don't fail the node
                    state["pdf_error"] = str(e)
                    pdf_path = None
            else:
                # WeasyPrint not available - skip PDF generation
                state["pdf_error"] = WEASYPRINT_ERROR or "WeasyPrint not available - PDF generation skipped"
                pdf_path = None

        # Update state
        state["pdf_path"] = pdf_path

        # Clear any previous errors if we made it this far
        if "error" in state and not state.get("pdf_error"):
            state["error"] = None

        return state

    except Exception as e:
        # Store error in state
        state["error"] = f"Publisher Error: {str(e)}"
        state["pdf_path"] = None
        return state


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("PDF PUBLISHER TEST")
    print("=" * 60)

    # Check WeasyPrint availability
    print("\n1. Checking WeasyPrint availability...")
    if WEASYPRINT_AVAILABLE:
        print("   ✓ WeasyPrint is installed and available")
    else:
        print("   ✗ WeasyPrint is NOT installed")
        print("   Install with: pip install weasyprint --break-system-packages")
        print("   System deps (Ubuntu): apt-get install python3-cffi python3-brotli \\")
        print("                         libpango-1.0-0 libpangoft2-1.0-0")

    # Check if test HTML exists (from Phase 4)
    test_html = "/tmp/test_report.html"

    if not os.path.exists(test_html):
        print(f"\n⚠️  Test HTML not found: {test_html}")
        print("   Run Phase 4 tests first to generate the HTML report")
    else:
        print(f"\n2. Test HTML found: {test_html}")

        if WEASYPRINT_AVAILABLE:
            print("\n3. Generating PDF...")
            try:
                pdf_path = generate_pdf(
                    html_path=test_html, output_path="/tmp/test_report.pdf"
                )

                print(f"   ✓ PDF created: {pdf_path}")

                # Check file size
                file_size = os.path.getsize(pdf_path) / 1024  # KB
                print(f"   ✓ File size: {file_size:.2f} KB")

            except PDFPublisherError as e:
                print(f"   ✗ Error: {e}")
        else:
            print("\n3. PDF generation skipped (WeasyPrint not available)")

        # Test LangGraph node
        print("\n4. LangGraph Node Test:")
        test_state = {"html_path": test_html}

        result_state = node_5_publisher(test_state, generate_pdf_output=True)

        if result_state.get("error"):
            print(f"   ✗ ERROR: {result_state['error']}")
        elif result_state.get("pdf_error"):
            print(f"   ⚠️  PDF Warning: {result_state['pdf_error']}")
            print(f"   ✓ HTML path: {result_state.get('html_path')}")
        else:
            print("   ✓ SUCCESS! Publisher completed:")
            print(f"   ✓ HTML path: {result_state.get('html_path')}")
            if result_state.get("pdf_path"):
                print(f"   ✓ PDF path: {result_state['pdf_path']}")
                print(f"   ✓ PDF exists: {os.path.exists(result_state['pdf_path'])}")
            else:
                print("   ℹ️  PDF not generated (WeasyPrint unavailable)")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    if WEASYPRINT_AVAILABLE and os.path.exists(test_html):
        print("\nGenerated files:")
        print(f"  - Test HTML: {test_html}")
        print("  - Test PDF: /tmp/test_report.pdf")
    else:
        print("\nNote: PDF generation requires WeasyPrint installation")
        print("The HTML report is fully functional without PDF conversion")
