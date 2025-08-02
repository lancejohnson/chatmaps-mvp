import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ChatMap MVP",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def load_santa_clara_boundary():
    """Load Santa Clara County boundary from GeoJSON file once per session"""
    try:
        # Use pathlib for cross-platform file paths
        script_dir = Path(__file__).parent
        boundary_file = (
            script_dir / "data" / "boundaries" / "santa_clara_county.geojson"
        )

        # Verify file exists
        if not boundary_file.exists():
            st.warning(f"Boundary file not found at: {boundary_file}")
            return None, None

        # Load the GeoJSON file
        boundary_gdf = gpd.read_file(boundary_file)

        # Convert to GeoJSON format for Folium
        boundary_geojson = json.loads(boundary_gdf.to_json())

        # Extract properties for display
        properties = boundary_gdf.iloc[0].drop("geometry").to_dict()

        return boundary_geojson, properties
    except Exception as e:
        st.error(f"Error loading boundary file: {e}")
        return None, None


def create_santa_clara_map():
    """Create a Folium map with actual Santa Clara County boundary"""
    # Santa Clara County center coordinates
    santa_clara_center = [37.3382, -121.8863]

    # Create the map
    m = folium.Map(location=santa_clara_center, zoom_start=10, tiles="OpenStreetMap")

    # Load boundary from GeoJSON file
    boundary_geojson, properties = load_santa_clara_boundary()

    if boundary_geojson:
        # Add the actual county boundary
        folium.GeoJson(
            boundary_geojson,
            style_function=lambda feature: {
                "fillColor": "transparent",
                "color": "red",
                "weight": 2,
                "fillOpacity": 0.1,
                "opacity": 0.8,
            },
            popup=folium.Popup(
                f"""
                <b>Santa Clara County</b><br>
                Area: {float(properties.get('acres', 0)):,.0f} acres<br>
                Size: {float(properties.get('sq_miles', 0)):,.1f} sq miles
                """,
                parse_html=True,
            ),
            tooltip="Santa Clara County Boundary",
        ).add_to(m)
    else:
        # Fallback to approximate boundary if file loading fails
        st.warning("Using approximate boundary - GeoJSON file not available")
        santa_clara_bounds = [
            [37.1, -122.2],  # Southwest corner
            [37.1, -121.2],  # Southeast corner
            [37.6, -121.2],  # Northeast corner
            [37.6, -122.2],  # Northwest corner
            [37.1, -122.2],  # Close the polygon
        ]

        folium.Polygon(
            locations=santa_clara_bounds,
            color="red",
            weight=2,
            fill=False,
            popup="Santa Clara County (Approximate Boundary)",
        ).add_to(m)

    return m


def main():
    st.title("🗺️ ChatMap MVP")
    st.markdown("*Explore Santa Clara County parcels through natural language queries*")

    # Create two columns for layout
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### Map View")

        # Create and display the map
        map_obj = create_santa_clara_map()

        # Display the map using streamlit-folium
        map_data = st_folium(
            map_obj, width=800, height=500, returned_objects=["last_object_clicked"]
        )

    with col2:
        st.markdown("### Map Info")

        # Load boundary info for display
        boundary_geojson, properties = load_santa_clara_boundary()

        if properties:
            st.info(f"""
            📍 **Santa Clara County, CA**
            
            **Boundary Details:**
            - Area: {float(properties.get('acres', 0)):,.0f} acres
            - Size: {float(properties.get('sq_miles', 0)):,.1f} sq miles
            - Data Source: Official county boundary
            """)
        else:
            st.info("📍 Santa Clara County, CA")

        st.markdown("""
        **Current View:**
        - Official county boundary in red
        - OpenStreetMap tiles
        - Ready for parcel data integration
        """)

        if map_data["last_object_clicked"]:
            st.markdown("**Last Clicked:**")
            st.json(map_data["last_object_clicked"])

    # Chat interface section
    st.markdown("---")
    st.markdown("### 💬 Chat with Your Data")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about parcels in Santa Clara County..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response (placeholder for now)
        with st.chat_message("assistant"):
            response = f"You asked: '{prompt}'\n\nI'm ready to help you explore parcel data! Currently, this is a placeholder response. Soon I'll be able to query the database and show results on the map above."
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
