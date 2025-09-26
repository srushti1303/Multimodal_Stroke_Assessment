"""
Multimodal Stroke Assessment System
Main execution script for Streamlit dashboard
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

def main():
    """Main function to run the Streamlit app"""
    
    # Set page config
    st.set_page_config(
        page_title="Stroke Assessment System",
        page_icon="🧠",
        layout="wide"
    )
    
    # Import and run the main dashboard
    try:
        from stroke_dashboard import main as dashboard_main
        dashboard_main()
        
    except ImportError as e:
        st.error(f"Failed to import dashboard: {e}")
        st.info("Make sure all dependencies are installed: pip install -r requirements.txt")
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check the logs for more information.")

if __name__ == "__main__":
    main()
