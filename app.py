import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import json
from pathlib import Path
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd
from schema import (
    get_primary_key,
    get_geometry_column,
    get_display_column,
    get_llm_schema_context,
)

# Load environment variables
load_dotenv()

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


@st.cache_resource
def get_db_engine():
    """Create and cache database engine for PostGIS connection"""
    try:
        # Get database URL from environment variable
        database_url = os.getenv("DATABASE_URL")

        if not database_url:
            st.error("DATABASE_URL environment variable not found")
            return None

        # Create engine
        engine = create_engine(database_url)

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return engine
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


@st.cache_data
def load_parcels_from_db(limit=1000, simplify_tolerance=0.0001):
    """Load parcel data from PostGIS database with performance optimizations"""
    engine = get_db_engine()
    if not engine:
        return None

    try:
        # Get schema-defined column names
        table_name = "parcels"
        id_column = get_primary_key(table_name)
        geometry_column = get_geometry_column(table_name)
        display_column = get_display_column(table_name)

        # Query to get parcels with their geometries as GeoJSON
        # Using ST_Simplify for performance and ST_AsGeoJSON for format
        query = f"""
        SELECT 
            {id_column},
            {display_column},
            situs_street_name,
            situs_city_name,
            situs_house_number,
            ST_AsGeoJSON(
                CASE 
                    WHEN {simplify_tolerance} > 0 THEN ST_Simplify({geometry_column}, {simplify_tolerance})
                    ELSE {geometry_column} 
                END
            ) as geometry_json
        FROM {table_name} 
        WHERE {geometry_column} IS NOT NULL
        ORDER BY ST_Area({geometry_column}) DESC  -- Show larger parcels first
        LIMIT {limit};
        """

        # Execute query and get results
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            st.warning("No parcels found in database")
            return None

        # Create GeoJSON structure manually for better performance
        features = []
        for _, row in df.iterrows():
            if row["geometry_json"]:
                geometry = json.loads(row["geometry_json"])

                # Build properties from available columns
                properties = {
                    "id": row[id_column],
                    "display_name": row[display_column]
                    if pd.notna(row[display_column])
                    else f"Parcel {row[id_column]}",
                }

                # Add address information if available
                if pd.notna(row.get("situs_house_number")) or pd.notna(
                    row.get("situs_street_name")
                ):
                    address_parts = []
                    if pd.notna(row.get("situs_house_number")):
                        address_parts.append(str(row["situs_house_number"]))
                    if pd.notna(row.get("situs_street_name")):
                        address_parts.append(str(row["situs_street_name"]))
                    properties["address"] = " ".join(address_parts)

                if pd.notna(row.get("situs_city_name")):
                    properties["city"] = str(row["situs_city_name"])

                feature = {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": geometry,
                }
                features.append(feature)

        parcels_geojson = {"type": "FeatureCollection", "features": features}

        return parcels_geojson

    except Exception as e:
        st.error(f"Error loading parcels: {e}")
        return None


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
            highlight=False,  # Disable the blue click highlight box
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

    # Load and add parcels to the map
    with st.spinner("Loading parcels from database..."):
        parcels_geojson = load_parcels_from_db()

    if parcels_geojson:
        # Add parcels layer to map
        folium.GeoJson(
            parcels_geojson,
            style_function=lambda feature: {
                "fillColor": "blue",
                "color": "darkblue",
                "weight": 1,
                "fillOpacity": 0.3,
                "opacity": 0.8,
            },
            popup=folium.GeoJsonPopup(
                fields=["display_name", "address", "city"],
                aliases=["Parcel:", "Address:", "City:"],
                localize=True,
                labels=True,
                style="background-color: yellow;",
            ),
            tooltip=folium.GeoJsonTooltip(
                fields=["display_name", "address"],
                aliases=["Parcel:", "Address:"],
                localize=True,
                sticky=True,
                labels=True,
                style="""
                    background-color: #F0EFEF;
                    border: 2px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """,
                max_width=800,
            ),
        ).add_to(m)

        st.success(
            f"Loaded {len(parcels_geojson.get('features', []))} parcels from database"
        )
    else:
        st.info("No parcels loaded - check database connection and data")

    return m


def show_schema_info():
    """Display schema information in the sidebar for debugging/development"""
    with st.sidebar:
        if st.checkbox("Show Schema Info (Debug)"):
            st.subheader("Parcels Table Schema")
            schema_context = get_llm_schema_context("parcels")
            st.text(schema_context)


def main():
    st.title("🗺️ ChatMap MVP")

    # Show schema info in sidebar for development
    show_schema_info()

    # Create and display the map
    map_obj = create_santa_clara_map()

    # Display the map using streamlit-folium
    st_folium(map_obj, width=800, height=500)

    # Chat interface section
    st.markdown("### 💬 Chat with Santa Clara County")

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
