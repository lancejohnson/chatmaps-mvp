"""
UI components for the ChatMaps MVP application.
"""

from typing import List, Dict, Any

import streamlit as st

from src.database.schema import get_llm_schema_context
from src.services.map_service import MapService


def display_property_list(features: List[Dict[str, Any]]):
    """Display a clickable list of properties with APN and acreage."""
    st.subheader("📋 Found Properties")

    for i, feature in enumerate(features, 1):
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        # Extract property details
        display_name = properties.get("display_name", f"Property {i}")
        apn = properties.get("apn", "N/A")  # Use dedicated APN field

        # Get acreage from properties (calculated by database)
        acres = properties.get("acres", 0.0)
        acreage = f"{acres:.2f}" if acres > 0 else "N/A"

        # Create button for each property
        button_label = f"{i}. APN: {apn} | Acres: {acreage}"

        if st.button(
            button_label, key=f"property_{i}", help=f"Click to zoom to {display_name}"
        ):
            # Extract coordinates for map centering
            if geometry and geometry.get("coordinates"):
                coords = MapService.extract_center_coordinates(geometry)
                st.session_state.selected_property = {
                    "coordinates": coords,
                    "feature": feature,
                    "index": i,
                }
                st.rerun()


def show_schema_info():
    """Display schema information in the sidebar for debugging/development."""
    with st.sidebar:
        if st.checkbox("Show Schema Info (Debug)"):
            st.subheader("Parcels Table Schema")
            schema_context = get_llm_schema_context("parcels")
            st.text(schema_context)


def show_example_questions():
    """Display example questions in an expandable section."""
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **🗺️ Property Searches** (will query database and highlight results on map):
        - "Find parcels on Main Street"
        - "Show me properties in San Jose"
        - "What parcels are near 123 First Street?"
        - "Show me large parcels over 10,000 square feet"
        - "Find properties in ZIP code 95110"
        - "Find parcel 123-45-678" (APN number)
        
        **💬 General Chat** (just conversation):
        - "How does this system work?"
        - "What data do you have available?"
        - "Tell me about Santa Clara County"
        - "What can I search for?"
        - "How accurate is this property data?"
        
        The assistant will automatically decide whether to search the database or just chat based on your question!
        """)


def show_clear_map_button(query_results):
    """Show the clear map highlights button if there are query results."""
    if query_results is not None:
        if st.button("🗺️ Clear Map Highlights"):
            st.session_state.query_results = None
            st.rerun()


def display_chat_interface_header():
    """Display the chat interface header and description."""
    st.markdown("### 💬 Chat with Santa Clara County")
    st.markdown(
        "*Ask questions about parcels and properties, or just chat! I'll automatically decide whether to search the database or have a conversation. Property search results will be highlighted on the map above.*"
    )
