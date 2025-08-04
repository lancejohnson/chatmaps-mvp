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
from schema import get_llm_schema_context
from src.llm.property_search_generator import PropertySearchGenerator
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
        return PropertySearchGenerator()
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

# Define the analyze properties tool for aggregation/statistical queries
ANALYZE_PROPERTIES_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_properties",
        "description": "Analyze and summarize Santa Clara County property data with statistics like counts, averages, totals, minimums, maximums, and distributions. Use this for questions like 'how many properties', 'what's the average size', 'total area', etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query requesting statistical analysis (e.g., 'how many properties in San Jose', 'average property size', 'total acreage by city', 'count of large parcels')",
                },
            },
            "required": ["query"],
        },
    },
}

# Define the list values tool for getting distinct values from columns
LIST_VALUES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_values",
        "description": "Get distinct/unique values from database columns for data exploration. Use for questions like 'what cities are available?', 'list all zip codes', 'what street names exist?', 'available street types?'",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query asking for distinct values (e.g., 'what cities are available', 'list unique zip codes', 'available street names')",
                },
            },
            "required": ["query"],
        },
    },
}

# Define the find specific property tool for direct property lookups
FIND_SPECIFIC_PROPERTY_TOOL = {
    "type": "function",
    "function": {
        "name": "find_specific_property",
        "description": "Find a specific property by APN (parcel number), exact address, or coordinates. Use this when the user provides specific identifiers like '123-45-678', '123 Main St', or coordinates like '37.4419, -122.1430'",
        "parameters": {
            "type": "object",
            "properties": {
                "lookup_value": {
                    "type": "string",
                    "description": "The specific value to look up: APN (e.g. '123-45-678'), address (e.g. '123 Main Street'), or coordinates (e.g. '37.4419, -122.1430')",
                },
                "lookup_type": {
                    "type": "string",
                    "enum": ["apn", "address", "coordinates", "auto"],
                    "description": "Type of lookup: 'apn' for parcel numbers, 'address' for street addresses, 'coordinates' for lat/long, 'auto' to detect automatically",
                    "default": "auto",
                },
            },
            "required": ["lookup_value"],
        },
    },
}


def execute_search_properties(query: str, max_results: int = 100):
    """
    Execute the property search using existing PropertySearchGenerator and QueryExecutor classes.

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


def execute_analyze_properties(query: str):
    """
    Execute property analysis for aggregation/statistical queries.

    Args:
        query: Natural language query requesting statistical analysis

    Returns:
        Dictionary with analysis results and metadata
    """
    try:
        # Get cached components - we'll need a separate analyzer
        from src.llm.analysis_generator import AnalysisGenerator

        db_engine = get_db_engine()

        if not db_engine:
            return {
                "success": False,
                "message": "Database connection not available. Please check your database configuration.",
                "data": None,
                "stats": None,
            }

        # Initialize analysis generator
        analysis_generator = AnalysisGenerator()

        # Use schema context for query generation
        schema_context = get_llm_schema_context("parcels")

        # Generate and validate SQL query for analysis
        success, sql_or_error, validation_message = (
            analysis_generator.generate_and_validate(query, schema_context)
        )

        if not success:
            return {
                "success": False,
                "message": f"Could not generate a valid analysis query: {sql_or_error}",
                "data": None,
                "stats": None,
                "sql_query": None,
            }

        sql_query = sql_or_error

        # Execute the analysis query - no need for limits on aggregations
        with db_engine.connect() as conn:
            df = pd.read_sql(sql_query, conn)

        if df.empty:
            return {
                "success": True,
                "message": "Analysis query executed but returned no results",
                "data": None,
                "stats": {},
                "sql_query": sql_query,
            }

        # Convert DataFrame to simple statistics format
        stats_data = df.to_dict("records")[0] if len(df) == 1 else df.to_dict("records")

        return {
            "success": True,
            "message": "Analysis completed successfully",
            "data": stats_data,
            "stats": stats_data,
            "sql_query": sql_query,
            "row_count": len(df),
        }

    except ImportError:
        # Fallback if AnalysisGenerator is not available
        return {
            "success": False,
            "message": "Analysis functionality not available. Please ensure all dependencies are installed.",
            "data": None,
            "stats": None,
            "sql_query": None,
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error executing property analysis: {str(e)}",
            "data": None,
            "stats": None,
            "sql_query": None,
        }


def execute_list_values(query: str):
    """
    Execute the list values query using ListValuesGenerator and QueryExecutor classes.

    Args:
        query: Natural language query asking for distinct values

    Returns:
        Dictionary with success status, message, and list of values
    """
    try:
        from src.llm.list_values_generator import ListValuesGenerator
        from src.database.query_executor import QueryExecutor
        from schema import get_llm_schema_context

        # Get cached components
        db_engine = get_db_engine()

        if not db_engine:
            return {
                "success": False,
                "message": "Database connection not available. Please check your database configuration.",
                "values": [],
                "sql_query": None,
            }

        # Get schema context for the LLM
        schema_context = get_llm_schema_context("parcels")

        # Generate the SQL query
        list_generator = ListValuesGenerator()
        success, sql_query, validation_message = list_generator.generate_and_validate(
            query, schema_context
        )

        if not success:
            return {
                "success": False,
                "message": f"❌ List Query Error\n\n{sql_query}",
                "values": [],
                "sql_query": None,
            }

        # Execute the query with reasonable limit for list queries
        query_executor = QueryExecutor(max_results=500)
        execution_result = query_executor.execute_query(sql_query, db_engine)

        if not execution_result["success"]:
            return {
                "success": False,
                "message": f"❌ Database Error: {execution_result['message']}",
                "values": [],
                "sql_query": sql_query,
            }

        geojson_data = execution_result["data"]

        if geojson_data is None or execution_result["row_count"] == 0:
            return {
                "success": True,
                "message": "✅ Query executed successfully, but no values found.",
                "values": [],
                "sql_query": sql_query,
            }

        # Extract values from GeoJSON features
        # For list queries, the values are in the properties of each feature
        features = geojson_data.get("features", [])
        values_list = []

        for feature in features:
            properties = feature.get("properties", {})
            # Get the first property value (should be our column value)
            if properties:
                # Get the first value from properties (our distinct column value)
                first_key = next(iter(properties.keys()))
                value = properties[first_key]
                if value is not None:
                    values_list.append(value)

        # Remove duplicates and sort (though SQL DISTINCT should handle duplicates)
        values_list = list(set(values_list))

        # Sort the values for better presentation
        try:
            values_list.sort()
        except TypeError:
            # If values can't be sorted (mixed types), leave as is
            pass

        return {
            "success": True,
            "message": f"✅ Found {len(values_list)} distinct values",
            "values": values_list,
            "sql_query": sql_query,
            "row_count": len(values_list),
        }

    except ImportError:
        # Fallback if ListValuesGenerator is not available
        return {
            "success": False,
            "message": "List values functionality not available. Please ensure all dependencies are installed.",
            "values": [],
            "sql_query": None,
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error executing list values query: {str(e)}",
            "values": [],
            "sql_query": None,
        }


def execute_find_specific_property(lookup_value: str, lookup_type: str = "auto"):
    """
    Find a specific property by APN, address, or coordinates with optimized queries.

    Args:
        lookup_value: The value to search for (APN, address, or coordinates)
        lookup_type: Type of lookup ('apn', 'address', 'coordinates', 'auto')

    Returns:
        Dictionary with search results and metadata
    """
    import re

    try:
        db_engine = get_db_engine()
        if not db_engine:
            return {
                "success": False,
                "message": "Database connection not available. Please check your database configuration.",
                "data": None,
                "row_count": 0,
                "sql_query": None,
            }

        # Auto-detect lookup type if not specified
        if lookup_type == "auto":
            lookup_type = detect_lookup_type(lookup_value)

        # Generate optimized SQL based on lookup type
        sql_query = None

        if lookup_type == "apn":
            # Clean APN input (remove spaces, normalize dashes)
            apn_clean = re.sub(r"[^\d\-]", "", lookup_value)
            sql_query = f"""
            SELECT ogc_fid, apn, situs_house_number, situs_street_name, situs_city_name, 
                   situs_zip_code, shape_area, shape_length,
                   ST_AsGeoJSON(wkb_geometry) as geometry
            FROM parcels 
            WHERE apn = '{apn_clean}' OR apn LIKE '%{apn_clean}%'
            LIMIT 10
            """

        elif lookup_type == "address":
            # Use LLM to generate address query - handles parsing intelligently
            query_generator = get_query_generator()
            if not query_generator:
                return {
                    "success": False,
                    "message": "Query generator not available for address lookup.",
                    "data": None,
                    "row_count": 0,
                    "sql_query": None,
                }

            # Get schema context and let LLM handle the address parsing
            from schema import get_llm_schema_context

            schema_context = get_llm_schema_context("parcels")

            # Create a focused query for the specific address
            address_query = f"Find the exact property at address: {lookup_value}"
            success, sql_or_error, validation_message = (
                query_generator.generate_and_validate(address_query, schema_context)
            )

            if success:
                sql_query = sql_or_error
                print(f"DEBUG - Generated SQL: {sql_query}")
            else:
                return {
                    "success": False,
                    "message": f"Could not generate address lookup query: {sql_or_error}",
                    "data": None,
                    "row_count": 0,
                    "sql_query": None,
                }

        elif lookup_type == "coordinates":
            # Parse coordinates
            coords = parse_coordinates(lookup_value)
            if coords:
                lat, lon = coords
                # Find parcels within 100 meters of the point
                sql_query = f"""
                SELECT ogc_fid, apn, situs_house_number, situs_street_name, situs_city_name,
                       situs_zip_code, shape_area, shape_length,
                       ST_AsGeoJSON(wkb_geometry) as geometry,
                       ST_Distance(wkb_geometry, ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)) as distance
                FROM parcels 
                WHERE ST_DWithin(wkb_geometry, ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326), 0.001)
                ORDER BY distance
                LIMIT 10
                """

        if not sql_query:
            return {
                "success": False,
                "message": f"Could not generate a valid query for lookup type '{lookup_type}' with value '{lookup_value}'",
                "data": None,
                "row_count": 0,
                "sql_query": None,
            }

        # Execute the optimized query
        executor = QueryExecutor(max_results=10)
        execution_result = executor.execute_query(sql_query, db_engine)

        return {
            "success": execution_result["success"],
            "message": execution_result["message"],
            "data": execution_result["data"],
            "row_count": execution_result["row_count"],
            "sql_query": sql_query,
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"❌ Property Lookup Error\n\n{str(e)}",
            "data": None,
            "row_count": 0,
            "sql_query": None,
        }


def detect_lookup_type(value: str) -> str:
    """Detect the type of lookup based on the input value."""
    import re

    # APN patterns: 123-45-678, 12345678, etc.
    if re.match(r"^\d{3}-?\d{2}-?\d{3}$", value.replace(" ", "").replace("-", "")):
        return "apn"

    # Coordinate patterns: 37.4419, -122.1430 or (37.4419, -122.1430)
    coord_pattern = r"[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+"
    if re.search(coord_pattern, value):
        return "coordinates"

    # Address patterns: contains numbers and street indicators
    if re.search(
        r"\d+.*\b(st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|way|ln|lane|ct|court)\b",
        value.lower(),
    ):
        return "address"

    # Default to address for other text inputs
    return "address"


# parse_address_components removed - using LLM-based parsing for better accuracy


def parse_coordinates(coord_str: str) -> tuple:
    """Parse coordinate string into (lat, lon) tuple."""
    import re

    # Remove parentheses and extra spaces
    cleaned = re.sub(r"[()]", "", coord_str)

    # Extract two numbers separated by comma
    match = re.findall(r"[-+]?\d*\.?\d+", cleaned)
    if len(match) >= 2:
        try:
            lat, lon = float(match[0]), float(match[1])
            # Basic validation for reasonable lat/lon ranges
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)
        except ValueError:
            pass

    return None


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

You have access to four functions. CAREFULLY choose the right one:

**find_specific_property** - Use when users provide SPECIFIC identifiers for a single property:
   - "Find parcel 123-45-678" → Direct APN lookup
   - "Show me property at 123 Main Street" → Exact address lookup
   - "What's at coordinates 37.4419, -122.1430" → Coordinate lookup
   - Any time they give you a specific APN, exact address, or lat/long coordinates

**analyze_properties** - Use for questions asking for NUMBERS/STATISTICS (no map display):
   - "How many properties..." → COUNT 
   - "What's the average..." → AVERAGE
   - "What's the total..." → SUM
   - "How many properties over X acres?" → COUNT with condition
   - "Average property size in [city]?" → AVERAGE with filter
   - Any question where the answer is a NUMBER or STATISTIC

**search_properties** - Use when users want to SEE/FIND multiple properties on the map:
   - "Show me properties over X acres" → Display on map
   - "Find properties on Main Street" → Show locations  
   - "Properties in San Jose" → Map individual parcels
   - When they want to see specific properties highlighted (but not one exact property)

**list_values** - Use when users want to EXPLORE/LIST what options are available:
   - "What cities are available?" → List distinct city names
   - "What zip codes exist?" → List unique zip codes
   - "Available street names?" → List street names
   - "What street types are there?" → List street types
   - Any question asking for distinct/unique values from the database

KEY DISTINCTIONS: 
- "Find parcel X" or "property at X address" = find_specific_property (exact lookup)
- "How many X?" = analyze_properties (returns a number)
- "Show me properties..." = search_properties (returns multiple map markers)
- "What X are available?" = list_values (returns a list of options)

For general conversation, respond normally without using functions.""",
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
            tools=[
                SEARCH_PROPERTIES_TOOL,
                ANALYZE_PROPERTIES_TOOL,
                LIST_VALUES_TOOL,
                FIND_SPECIFIC_PROPERTY_TOOL,
            ],
            tool_choice="auto",  # Let GPT decide when to use the tool
            temperature=0.1,
        )

        message = response.choices[0].message

        # Check if GPT wants to use the search or analyze function
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

                elif tool_call.function.name == "analyze_properties":
                    # Extract arguments and execute the analysis
                    try:
                        args = json.loads(tool_call.function.arguments)
                        analysis_query = args["query"]

                        # Debug: Log what function calling extracted
                        print(
                            f"DEBUG - Function calling extracted for analysis: query='{analysis_query}'"
                        )

                        # Execute the property analysis
                        analysis_result = execute_analyze_properties(analysis_query)

                        # Create a follow-up message with the analysis results
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {
                                    "success": analysis_result["success"],
                                    "message": analysis_result["message"],
                                    "stats": analysis_result.get("stats", {}),
                                    "sql_query": analysis_result.get("sql_query", ""),
                                }
                            ),
                        }

                        # Get final response from GPT incorporating the analysis results
                        final_messages = messages + [message, tool_message]
                        final_response = client.chat.completions.create(
                            model="gpt-4.1", messages=final_messages, temperature=0.1
                        )

                        final_content = final_response.choices[0].message.content

                        # Format the response nicely for analysis results
                        if analysis_result["success"] and analysis_result.get("stats"):
                            formatted_response = f"""📊 **Analysis Results**

{final_content}

**SQL Query Used:**
```sql
{analysis_result.get('sql_query', 'N/A')}
```"""
                        else:
                            formatted_response = f"""❌ **Analysis Error**

{final_content}

**Error:** {analysis_result['message']}"""

                        return {
                            "success": analysis_result["success"],
                            "response": formatted_response,
                            "search_results": None,  # Analysis doesn't return map data
                            "analysis_results": analysis_result.get("stats"),
                            "is_analysis": True,
                        }

                    except json.JSONDecodeError:
                        return {
                            "success": False,
                            "response": "❌ **Error:** Could not parse analysis parameters. Please try rephrasing your question.",
                            "search_results": None,
                        }

                elif tool_call.function.name == "list_values":
                    # Extract arguments and execute the list values query
                    try:
                        args = json.loads(tool_call.function.arguments)
                        list_query = args["query"]

                        # Debug: Log what function calling extracted
                        print(
                            f"DEBUG - Function calling extracted for list values: query='{list_query}'"
                        )

                        # Execute the list values query
                        list_result = execute_list_values(list_query)

                        # Create a follow-up message with the list results
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {
                                    "success": list_result["success"],
                                    "message": list_result["message"],
                                    "values": list_result.get("values", []),
                                    "sql_query": list_result.get("sql_query", ""),
                                }
                            ),
                        }

                        # Get final response from GPT incorporating the list results
                        final_messages = messages + [message, tool_message]
                        final_response = client.chat.completions.create(
                            model="gpt-4.1", messages=final_messages, temperature=0.1
                        )

                        final_content = final_response.choices[0].message.content

                        # Format the response nicely for list results
                        if list_result["success"] and list_result.get("values"):
                            values_list = list_result["values"]
                            # Format the values nicely
                            if len(values_list) <= 20:
                                # If 20 or fewer values, show them all
                                values_display = ", ".join(str(v) for v in values_list)
                            else:
                                # If more than 20, show first 20 and indicate there are more
                                values_display = (
                                    ", ".join(str(v) for v in values_list[:20])
                                    + f"... ({len(values_list)} total)"
                                )

                            formatted_response = f"""📋 **Available Values**

{final_content}

**Found {len(values_list)} values:**
{values_display}

**SQL Query Used:**
```sql
{list_result.get('sql_query', 'N/A')}
```"""
                        else:
                            formatted_response = f"""❌ **List Values Error**

{final_content}

**Error:** {list_result['message']}"""

                        return {
                            "success": list_result["success"],
                            "response": formatted_response,
                            "search_results": None,  # List values doesn't return map data
                            "list_results": list_result.get("values"),
                            "is_list": True,
                        }

                    except json.JSONDecodeError:
                        return {
                            "success": False,
                            "response": "❌ **Error:** Could not parse list values parameters. Please try rephrasing your question.",
                            "search_results": None,
                        }

                elif tool_call.function.name == "find_specific_property":
                    # Extract arguments and execute the specific property lookup
                    try:
                        args = json.loads(tool_call.function.arguments)
                        lookup_value = args["lookup_value"]
                        lookup_type = args.get("lookup_type", "auto")

                        # Debug: Log what function calling extracted
                        print(
                            f"DEBUG - Function calling extracted for property lookup: value='{lookup_value}', type='{lookup_type}'"
                        )

                        # Execute the specific property lookup
                        lookup_result = execute_find_specific_property(
                            lookup_value, lookup_type
                        )

                        # Create a follow-up message with the lookup results
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {
                                    "success": lookup_result["success"],
                                    "message": lookup_result["message"],
                                    "row_count": lookup_result["row_count"],
                                    "sql_query": lookup_result.get("sql_query", ""),
                                }
                            ),
                        }

                        # Get final response from GPT incorporating the lookup results
                        final_messages = messages + [message, tool_message]
                        final_response = client.chat.completions.create(
                            model="gpt-4.1", messages=final_messages, temperature=0.1
                        )

                        final_content = final_response.choices[0].message.content

                        # Format the response nicely for property lookup
                        if lookup_result["success"] and lookup_result["row_count"] > 0:
                            detected_type = detect_lookup_type(lookup_value)
                            formatted_response = f"""🎯 **Found {lookup_result['row_count']} propert{'y' if lookup_result['row_count'] == 1 else 'ies'}** (detected as {detected_type} lookup)

{final_content}

**SQL Query Used:**
```sql
{lookup_result.get('sql_query', 'N/A')}
```

🗺️ **The property is now highlighted on the map above!**"""
                        elif (
                            lookup_result["success"] and lookup_result["row_count"] == 0
                        ):
                            detected_type = detect_lookup_type(lookup_value)
                            formatted_response = f"""ℹ️ **No property found** (searched as {detected_type})

{final_content}

**SQL Query Used:**
```sql
{lookup_result.get('sql_query', 'N/A')}
```

💡 Please check the format and try again:
- **APN**: Try format like '123-45-678'
- **Address**: Try '123 Main Street' or '123 Main St, San Jose'
- **Coordinates**: Try '37.4419, -122.1430'"""
                        else:
                            formatted_response = f"""❌ **Property Lookup Error**

{final_content}

**Error:** {lookup_result['message']}"""

                        return {
                            "success": lookup_result["success"],
                            "response": formatted_response,
                            "search_results": lookup_result["data"]
                            if lookup_result["success"]
                            else None,
                            "row_count": lookup_result["row_count"],
                            "is_specific_lookup": True,
                        }

                    except json.JSONDecodeError:
                        return {
                            "success": False,
                            "response": "❌ **Error:** Could not parse property lookup parameters. Please try rephrasing your question.",
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
            "error_details": "PropertySearchGenerator initialization failed",
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


def create_santa_clara_map(query_results=None, selected_property=None):
    """Create a Folium map with Santa Clara County boundary and optional query results"""
    # Default Santa Clara County center coordinates
    default_center = [37.3382, -121.8863]
    default_zoom = 10

    # If a property is selected, center on it with higher zoom
    if selected_property and selected_property.get("coordinates"):
        map_center = selected_property["coordinates"]
        map_zoom = 16  # Zoom in closer to the selected property
        m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="OpenStreetMap")
    elif query_results and query_results.get("features"):
        # Calculate bounds for all search results and auto-fit
        bounds = calculate_bounds(query_results)
        if bounds:
            # Create map without specific center/zoom - will be set by fit_bounds
            m = folium.Map(tiles="OpenStreetMap")
            # Fit map to show all search results
            m.fit_bounds(bounds)
        else:
            # Fallback to default if bounds calculation fails
            m = folium.Map(
                location=default_center, zoom_start=default_zoom, tiles="OpenStreetMap"
            )
    else:
        # Default map view when no search results
        m = folium.Map(
            location=default_center, zoom_start=default_zoom, tiles="OpenStreetMap"
        )

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
        # Show clean map without any default parcel data
        st.info(
            "💬 Use the chat below to search for specific parcels, properties, or areas"
        )

    return m


def display_property_list(features):
    """Display a clickable list of properties with APN and acreage"""
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


def extract_all_coordinates(geometry):
    """Extract all coordinates from geometry for bounds calculation"""
    coords = []
    try:
        if geometry["type"] == "Polygon":
            # Get all coordinates from the outer ring
            coords.extend(geometry["coordinates"][0])
        elif geometry["type"] == "Point":
            coords.append(geometry["coordinates"])
        elif geometry["type"] == "MultiPolygon":
            # Get coordinates from all polygons
            for polygon in geometry["coordinates"]:
                coords.extend(polygon[0])  # Outer ring of each polygon
    except (KeyError, IndexError, TypeError):
        pass
    return coords


def calculate_bounds(query_results):
    """Calculate bounding box for all search result geometries"""
    if not query_results or not query_results.get("features"):
        return None

    all_coords = []

    # Extract coordinates from all features
    for feature in query_results["features"]:
        geometry = feature.get("geometry")
        if geometry:
            coords = extract_all_coordinates(geometry)
            all_coords.extend(coords)

    if not all_coords:
        return None

    # Calculate bounds [south, west, north, east]
    lats = [coord[1] for coord in all_coords]
    lngs = [coord[0] for coord in all_coords]

    south = min(lats)
    north = max(lats)
    west = min(lngs)
    east = max(lngs)

    # Add small padding (about 5% of the span)
    lat_padding = (north - south) * 0.05
    lng_padding = (east - west) * 0.05

    return [
        [south - lat_padding, west - lng_padding],  # Southwest corner
        [north + lat_padding, east + lng_padding],  # Northeast corner
    ]


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
