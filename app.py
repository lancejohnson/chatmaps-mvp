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
from src.llm.query_generator import QueryGenerator
from src.database.query_executor import QueryExecutor

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


@st.cache_resource
def get_query_generator():
    """Initialize and cache the LLM query generator"""
    try:
        return QueryGenerator()
    except Exception as e:
        st.error(f"Failed to initialize query generator: {e}")
        return None


@st.cache_resource
def get_query_executor():
    """Initialize and cache the query executor"""
    return QueryExecutor(max_results=500)


def process_user_query(user_prompt: str):
    """
    Process a user query through the complete LLM pipeline.

    Returns a dictionary with:
    - success: bool
    - response_message: str (formatted for display)
    - sql_query: str (if successful)
    - result_data: dict (GeoJSON if successful)
    - error_details: str (if failed)
    """
    # Initialize components
    query_generator = get_query_generator()
    query_executor = get_query_executor()
    db_engine = get_db_engine()

    if not query_generator:
        return {
            "success": False,
            "response_message": "❌ **Query Generator Error**\n\nThe AI query generator is not available. Please check your OpenAI API key configuration.",
            "error_details": "QueryGenerator initialization failed",
        }

    if not db_engine:
        return {
            "success": False,
            "response_message": "❌ **Database Connection Error**\n\nCannot connect to the database. Please check your database configuration.",
            "error_details": "Database engine not available",
        }

    try:
        # Step 1: Generate SQL query from natural language
        schema_context = get_llm_schema_context("parcels")
        success, sql_or_error, validation_message = (
            query_generator.generate_and_validate(user_prompt, schema_context)
        )

        if not success:
            return {
                "success": False,
                "response_message": f"❌ **Query Generation Failed**\n\nI couldn't generate a valid SQL query for your request.\n\n**Error:** {sql_or_error}\n\n**Details:** {validation_message}",
                "error_details": f"SQL generation failed: {sql_or_error}",
            }

        sql_query = sql_or_error

        # Step 2: Execute the SQL query
        execution_result = query_executor.execute_query(sql_query, db_engine)

        if not execution_result["success"]:
            return {
                "success": False,
                "response_message": f"❌ **Query Execution Failed**\n\n{execution_result['message']}\n\n**SQL Query:**\n```sql\n{sql_query}\n```",
                "sql_query": sql_query,
                "error_details": execution_result["message"],
            }

        # Step 3: Format successful response
        row_count = execution_result["row_count"]
        result_data = execution_result["data"]

        if row_count == 0:
            response_message = f"✅ **Query Executed Successfully**\n\nYour query executed without errors, but **no parcels matched** your criteria.\n\n**SQL Query:**\n```sql\n{sql_query}\n```\n\n💡 **Try:**\n- Broadening your search criteria\n- Checking spelling of street names or cities\n- Using partial matches instead of exact names"
        else:
            response_message = f"✅ **Found {row_count} parcel{'s' if row_count != 1 else ''}**\n\nI found **{row_count} parcel{'s' if row_count != 1 else ''}** matching your query. The results are now displayed on the map above with blue highlighting.\n\n**SQL Query:**\n```sql\n{sql_query}\n```\n\n🗺️ **Map Updated:** Click on any highlighted parcel for details."

        return {
            "success": True,
            "response_message": response_message,
            "sql_query": sql_query,
            "result_data": result_data,
            "row_count": row_count,
        }

    except Exception as e:
        return {
            "success": False,
            "response_message": f"❌ **Unexpected Error**\n\nAn unexpected error occurred while processing your query.\n\n**Error:** {str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists.",
            "error_details": f"Unexpected error: {str(e)}",
        }


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

                # Add address information if available, otherwise set to empty string
                address_parts = []
                if pd.notna(row.get("situs_house_number")):
                    address_parts.append(str(row["situs_house_number"]))
                if pd.notna(row.get("situs_street_name")):
                    address_parts.append(str(row["situs_street_name"]))

                # Always include address field, even if empty
                properties["address"] = (
                    " ".join(address_parts) if address_parts else "No address available"
                )

                # Always include city field
                properties["city"] = (
                    str(row["situs_city_name"])
                    if pd.notna(row.get("situs_city_name"))
                    else "Unknown city"
                )

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


def create_santa_clara_map(query_results=None):
    """Create a Folium map with Santa Clara County boundary and optional query results"""
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

    # If we have query results, display them prominently
    if query_results and query_results.get("features"):
        # Add query results layer to map
        folium.GeoJson(
            query_results,
            style_function=lambda feature: {
                "fillColor": "orange",
                "color": "darkorange",
                "weight": 2,
                "fillOpacity": 0.6,
                "opacity": 1.0,
            },
            popup=folium.GeoJsonPopup(
                fields=list(query_results["features"][0]["properties"].keys()),
                localize=True,
                labels=True,
                style="background-color: orange; color: black;",
            ),
            tooltip=folium.GeoJsonTooltip(
                fields=list(query_results["features"][0]["properties"].keys())[:3],
                localize=True,
                sticky=True,
                labels=True,
                style="""
                    background-color: orange;
                    color: black;
                    border: 2px solid darkorange;
                    border-radius: 3px;
                    box-shadow: 3px;
                """,
                max_width=800,
            ),
        ).add_to(m)

        st.success(
            f"🎯 Showing {len(query_results['features'])} query result{'s' if len(query_results['features']) != 1 else ''} highlighted in orange"
        )
    else:
        # Load and add default parcels to the map (sample)
        with st.spinner("Loading sample parcels from database..."):
            parcels_geojson = load_parcels_from_db(
                limit=200
            )  # Reduced limit for performance

        if parcels_geojson:
            # Add parcels layer to map
            folium.GeoJson(
                parcels_geojson,
                style_function=lambda feature: {
                    "fillColor": "lightblue",
                    "color": "blue",
                    "weight": 1,
                    "fillOpacity": 0.2,
                    "opacity": 0.6,
                },
                popup=folium.GeoJsonPopup(
                    fields=["display_name", "address", "city"],
                    aliases=["Parcel:", "Address:", "City:"],
                    localize=True,
                    labels=True,
                    style="background-color: lightblue;",
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

            st.info(
                f"📍 Showing {len(parcels_geojson.get('features', []))} sample parcels (use chat to search for specific parcels)"
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

    # Initialize session state for query results
    if "query_results" not in st.session_state:
        st.session_state.query_results = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Create and display the map (with current query results if any)
    map_obj = create_santa_clara_map(st.session_state.query_results)

    # Display the map using streamlit-folium
    st_folium(map_obj, width=800, height=500)

    # Chat interface section
    st.markdown("### 💬 Chat with Santa Clara County")
    st.markdown(
        "*Ask questions about parcels, addresses, or areas in Santa Clara County. I'll search the database and highlight results on the map above.*"
    )

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input with enhanced processing
    if prompt := st.chat_input("Ask about parcels in Santa Clara County..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the user query with loading state
        with st.chat_message("assistant"):
            # Show loading spinner while processing
            with st.spinner("🤖 Processing your query..."):
                with st.empty():
                    st.write("🧠 Generating SQL query...")

                # Process the query through the LLM pipeline
                result = process_user_query(prompt)

            # Display the result
            st.markdown(result["response_message"])

            # Update map with new query results if successful
            if result["success"] and result.get("result_data"):
                st.session_state.query_results = result["result_data"]
                # Trigger rerun to update the map
                st.rerun()
            elif result["success"] and result.get("row_count", 0) == 0:
                # Clear previous results if no matches found
                st.session_state.query_results = None

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": result["response_message"]}
        )

    # Add helpful examples
    with st.expander("💡 Example Questions"):
        st.markdown("""
        Try asking questions like:
        
        **Address & Location Searches:**
        - "Find parcels on Main Street"
        - "Show me properties in San Jose"
        - "What parcels are near 123 First Street?"
        
        **Area & Size Queries:**
        - "Show me large parcels over 10,000 square feet"
        - "Find small lots under 5,000 square feet"
        
        **City & ZIP Code Searches:**
        - "Show all parcels in Palo Alto"
        - "Find properties in ZIP code 95110"
        - "What's in Mountain View?"
        
        **Specific Parcel Lookups:**
        - "Find parcel 123-45-678" (APN number)
        - "Show me parcel with APN starting with 123"
        """)

    # Add clear results button
    if st.session_state.query_results is not None:
        if st.button("🗺️ Clear Map Highlights"):
            st.session_state.query_results = None
            st.rerun()


if __name__ == "__main__":
    main()
