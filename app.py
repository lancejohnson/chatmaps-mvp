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
from openai import OpenAI
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
        # Clear cache key when prompts change - update this comment to force cache refresh
        # Cache version: v2.0 - Updated limit handling
        return QueryGenerator()
    except Exception as e:
        st.error(f"Failed to initialize query generator: {e}")
        return None


@st.cache_resource
def get_query_executor():
    """Initialize and cache the query executor"""
    return QueryExecutor(max_results=500)


@st.cache_resource
def get_openai_client():
    """Initialize and cache the OpenAI client"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OPENAI_API_KEY environment variable not found")
            return None
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return None


# Define the search properties tool for OpenAI function calling
SEARCH_PROPERTIES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_properties",
        "description": "Search for properties in Santa Clara County based on natural language queries about addresses, locations, areas, or property characteristics",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query about properties (e.g., 'properties on Main Street', 'homes in San Jose', 'large parcels over 10000 sq ft')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Extract this from user query if they specify a number (e.g., 'find 8 properties' = 8, 'show me 25 parcels' = 25, 'a few properties' = 10). Default: 100 if no number specified.",
                    "default": 100,
                },
            },
            "required": ["query"],
        },
    },
}


def execute_search_properties(query: str, max_results: int = 100):
    """
    Execute the property search using existing QueryGenerator and QueryExecutor classes.

    Args:
        query: Natural language query about properties
        max_results: Maximum number of results to return

    Returns:
        Dictionary with search results and metadata
    """
    try:
        # Get cached components
        query_generator = get_query_generator()
        db_engine = get_db_engine()

        if not query_generator:
            return {
                "success": False,
                "message": "Query generator not available. Please check your OpenAI API key configuration.",
                "data": None,
                "row_count": 0,
            }

        if not db_engine:
            return {
                "success": False,
                "message": "Database connection not available. Please check your database configuration.",
                "data": None,
                "row_count": 0,
            }

        # Use schema context for query generation
        schema_context = get_llm_schema_context("parcels")

        # Generate and validate SQL query
        success, sql_or_error, validation_message = (
            query_generator.generate_and_validate(query, schema_context)
        )

        if not success:
            return {
                "success": False,
                "message": f"Could not generate a valid SQL query: {sql_or_error}",
                "data": None,
                "row_count": 0,
                "sql_query": None,
            }

        sql_query = sql_or_error

        # Execute the query with the specified max_results
        executor_with_limit = QueryExecutor(max_results=max_results)
        execution_result = executor_with_limit.execute_query(sql_query, db_engine)

        return {
            "success": execution_result["success"],
            "message": execution_result["message"],
            "data": execution_result["data"],
            "row_count": execution_result["row_count"],
            "sql_query": execution_result["sql_executed"],
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error executing property search: {str(e)}",
            "data": None,
            "row_count": 0,
            "sql_query": None,
        }


def process_chat_with_function_calling(user_prompt: str, chat_history: list):
    """
    Process a chat message using OpenAI function calling to decide whether to search properties or just chat.

    Args:
        user_prompt: The user's message
        chat_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}]

    Returns:
        Dictionary with response and any search results
    """
    client = get_openai_client()
    if not client:
        return {
            "success": False,
            "response": "❌ Chat service unavailable. Please check your OpenAI API key configuration.",
            "search_results": None,
        }

    try:
        # Prepare messages for OpenAI, including chat history
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant for exploring Santa Clara County property data. 
                
You have access to a search_properties function that can query a database of property parcels in Santa Clara County. Use this function when users ask about:
- Finding properties at specific addresses or streets
- Searching for properties in specific cities or areas  
- Looking for properties with certain characteristics (size, etc.)
- Any question that requires actual property data

For general conversation, questions about how the system works, or non-property related topics, just respond normally without using the function.

When you do find properties, always mention that the results will be highlighted on the map above.""",
            }
        ]

        # Add chat history (limit to last 10 messages to avoid token limits)
        messages.extend(chat_history[-10:])

        # Add the current user message
        messages.append({"role": "user", "content": user_prompt})

        # Call OpenAI with function calling
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=[SEARCH_PROPERTIES_TOOL],
            tool_choice="auto",  # Let GPT decide when to use the tool
            temperature=0.1,
        )

        message = response.choices[0].message

        # Check if GPT wants to use the search function
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "search_properties":
                    # Extract arguments and execute the search
                    try:
                        args = json.loads(tool_call.function.arguments)
                        search_query = args["query"]
                        max_results = args.get("max_results", 100)

                        # Debug: Log what function calling extracted
                        print(
                            f"DEBUG - Function calling extracted: query='{search_query}', max_results={max_results}"
                        )

                        # Execute the property search
                        search_result = execute_search_properties(
                            search_query, max_results
                        )

                        # Create a follow-up message with the search results
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {
                                    "success": search_result["success"],
                                    "message": search_result["message"],
                                    "row_count": search_result["row_count"],
                                    "sql_query": search_result.get("sql_query", ""),
                                }
                            ),
                        }

                        # Get final response from GPT incorporating the search results
                        final_messages = messages + [message, tool_message]
                        final_response = client.chat.completions.create(
                            model="gpt-4.1", messages=final_messages, temperature=0.1
                        )

                        final_content = final_response.choices[0].message.content

                        # Format the response nicely
                        if search_result["success"] and search_result["row_count"] > 0:
                            formatted_response = f"""✅ **Found {search_result['row_count']} propert{'y' if search_result['row_count'] == 1 else 'ies'}**

{final_content}

**SQL Query Used:**
```sql
{search_result.get('sql_query', 'N/A')}
```

🗺️ **The results are now highlighted on the map above!**"""
                        elif (
                            search_result["success"] and search_result["row_count"] == 0
                        ):
                            formatted_response = f"""ℹ️ **No properties found**

{final_content}

**SQL Query Used:**
```sql
{search_result.get('sql_query', 'N/A')}
```

💡 Try broadening your search criteria or checking the spelling of street names."""
                        else:
                            formatted_response = f"""❌ **Search Error**

{final_content}

**Error:** {search_result['message']}"""

                        return {
                            "success": search_result["success"],
                            "response": formatted_response,
                            "search_results": search_result["data"]
                            if search_result["success"]
                            else None,
                            "row_count": search_result["row_count"],
                        }

                    except json.JSONDecodeError:
                        return {
                            "success": False,
                            "response": "❌ **Error:** Could not parse search parameters. Please try rephrasing your question.",
                            "search_results": None,
                        }

        # If no function was called, just return the regular chat response
        return {
            "success": True,
            "response": message.content,
            "search_results": None,
            "is_chat": True,  # Flag to indicate this was just conversation
        }

    except Exception as e:
        return {
            "success": False,
            "response": f"❌ **Chat Error:** {str(e)}\n\nPlease try again or contact support if the issue persists.",
            "search_results": None,
        }


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

        # Query to get parcels with their geometries as GeoJSON and area in acres
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
            ) as geometry_json,
            ROUND((ST_Area({geometry_column}) / 4047)::numeric, 2) as acres
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
                    "acres": row.get("acres", 0.0)
                    if pd.notna(row.get("acres"))
                    else 0.0,
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


def create_santa_clara_map(query_results=None, selected_property=None):
    """Create a Folium map with Santa Clara County boundary and optional query results"""
    # Default Santa Clara County center coordinates
    default_center = [37.3382, -121.8863]
    default_zoom = 10

    # If a property is selected, center on it with higher zoom
    if selected_property and selected_property.get("coordinates"):
        map_center = selected_property["coordinates"]
        map_zoom = 16  # Zoom in closer to the selected property
    else:
        map_center = default_center
        map_zoom = default_zoom

    # Create the map
    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="OpenStreetMap")

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
        for idx, feature in enumerate(query_results["features"]):
            # Check if this feature is the selected property
            is_selected = (
                selected_property and selected_property.get("index") == idx + 1
            )

            # Different styles for selected vs unselected properties
            if is_selected:
                style = {
                    "fillColor": "red",
                    "color": "darkred",
                    "weight": 3,
                    "fillOpacity": 0.8,
                    "opacity": 1.0,
                }
                popup_style = "background-color: red; color: white;"
                tooltip_style = """
                    background-color: red;
                    color: white;
                    border: 3px solid darkred;
                    border-radius: 3px;
                    box-shadow: 3px;
                """
            else:
                style = {
                    "fillColor": "orange",
                    "color": "darkorange",
                    "weight": 2,
                    "fillOpacity": 0.6,
                    "opacity": 1.0,
                }
                popup_style = "background-color: orange; color: black;"
                tooltip_style = """
                    background-color: orange;
                    color: black;
                    border: 2px solid darkorange;
                    border-radius: 3px;
                    box-shadow: 3px;
                """

            # Add individual feature to map
            folium.GeoJson(
                {
                    "type": "Feature",
                    "properties": feature["properties"],
                    "geometry": feature["geometry"],
                },
                style_function=lambda x, style=style: style,
                popup=folium.GeoJsonPopup(
                    fields=list(feature["properties"].keys()),
                    localize=True,
                    labels=True,
                    style=popup_style,
                ),
                tooltip=folium.GeoJsonTooltip(
                    fields=list(feature["properties"].keys())[:3],
                    localize=True,
                    sticky=True,
                    labels=True,
                    style=tooltip_style,
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


def display_property_list(features):
    """Display a clickable list of properties with APN and acreage"""
    st.subheader("📋 Found Properties")

    for i, feature in enumerate(features, 1):
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        # Extract property details
        display_name = properties.get("display_name", f"Property {i}")
        apn = properties.get("id", "N/A")  # Using id as APN

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
                coords = extract_center_coordinates(geometry)
                st.session_state.selected_property = {
                    "coordinates": coords,
                    "feature": feature,
                    "index": i,
                }
                st.rerun()


def extract_center_coordinates(geometry):
    """Extract center coordinates from geometry for map centering"""
    try:
        if geometry["type"] == "Polygon":
            # Get first ring of polygon
            coords = geometry["coordinates"][0]
            # Calculate centroid (simple average)
            lats = [coord[1] for coord in coords]
            lngs = [coord[0] for coord in coords]
            center_lat = sum(lats) / len(lats)
            center_lng = sum(lngs) / len(lngs)
            return [center_lat, center_lng]
        elif geometry["type"] == "Point":
            return [geometry["coordinates"][1], geometry["coordinates"][0]]
        elif geometry["type"] == "MultiPolygon":
            # Use first polygon
            coords = geometry["coordinates"][0][0]
            lats = [coord[1] for coord in coords]
            lngs = [coord[0] for coord in coords]
            center_lat = sum(lats) / len(lats)
            center_lng = sum(lngs) / len(lngs)
            return [center_lat, center_lng]
    except (KeyError, IndexError, TypeError):
        pass
    return None


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

    # Initialize session state for selected property
    if "selected_property" not in st.session_state:
        st.session_state.selected_property = None

    # Create columns layout - map on left, property list on right
    col_map, col_properties = st.columns([2, 1])

    with col_map:
        # Create and display the map (with current query results if any)
        map_obj = create_santa_clara_map(
            st.session_state.query_results, st.session_state.selected_property
        )
        # Display the map using streamlit-folium
        st_folium(map_obj, width=800, height=500)

    with col_properties:
        # Display property list if we have query results
        if st.session_state.query_results and st.session_state.query_results.get(
            "features"
        ):
            display_property_list(st.session_state.query_results["features"])

    # Chat interface section
    st.markdown("### 💬 Chat with Santa Clara County")
    st.markdown(
        "*Ask questions about parcels and properties, or just chat! I'll automatically decide whether to search the database or have a conversation. Property search results will be highlighted on the map above.*"
    )

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input with enhanced processing
    if prompt := st.chat_input("Ask about properties or just chat..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the user query with function calling
        with st.chat_message("assistant"):
            # Show loading spinner while processing
            with st.spinner("🤖 Thinking about your request..."):
                # Process through the new function calling pipeline
                result = process_chat_with_function_calling(
                    prompt, st.session_state.messages[:-1]
                )

            # Display the result
            st.markdown(result["response"])

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": result["response"]}
        )

        # Update map with new query results if we have search results
        if result.get("search_results"):
            st.session_state.query_results = result["search_results"]
            # Trigger rerun to update the map
            st.rerun()
        elif result.get("row_count", 0) == 0 and not result.get("is_chat"):
            # Clear previous results if search returned no matches
            st.session_state.query_results = None
        # If it's just chat (is_chat=True), don't clear the map

    # Add helpful examples
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

    # Add clear results button
    if st.session_state.query_results is not None:
        if st.button("🗺️ Clear Map Highlights"):
            st.session_state.query_results = None
            st.rerun()


if __name__ == "__main__":
    main()
