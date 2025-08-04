"""
Property service for handling all property-related operations.
"""

import os
import re
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from sqlalchemy import create_engine, text
from openai import OpenAI
import streamlit as st

from src.llm.property_search_generator import PropertySearchGenerator
from src.llm.analysis_generator import AnalysisGenerator
from src.llm.list_values_generator import ListValuesGenerator
from src.database.query_executor import QueryExecutor
from src.database.schema import get_llm_schema_context


class PropertyService:
    """Service for handling property searches, analysis, and lookups."""

    def __init__(self):
        """Initialize the property service with cached resources."""
        self._db_engine = None
        self._query_generator = None
        self._openai_client = None

    @property
    def db_engine(self):
        """Get or create database engine."""
        if self._db_engine is None:
            self._db_engine = self._create_db_engine()
        return self._db_engine

    @property
    def query_generator(self):
        """Get or create query generator."""
        if self._query_generator is None:
            self._query_generator = self._create_query_generator()
        return self._query_generator

    @property
    def openai_client(self):
        """Get or create OpenAI client."""
        if self._openai_client is None:
            self._openai_client = self._create_openai_client()
        return self._openai_client

    def _create_db_engine(self):
        """Create and cache database engine for PostGIS connection."""
        try:
            database_url = os.getenv("DATABASE_URL")

            if not database_url:
                st.error("DATABASE_URL environment variable not found")
                return None

            engine = create_engine(database_url)

            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            return engine
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return None

    def _create_query_generator(self):
        """Initialize and cache the LLM query generator."""
        try:
            return PropertySearchGenerator()
        except Exception as e:
            st.error(f"Failed to initialize query generator: {e}")
            return None

    def _create_openai_client(self):
        """Initialize and cache the OpenAI client."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OPENAI_API_KEY environment variable not found")
                return None
            return OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")
            return None

    def search_properties(self, query: str, max_results: int = 100) -> Dict[str, Any]:
        """
        Execute the property search using PropertySearchGenerator and QueryExecutor.

        Args:
            query: Natural language query about properties
            max_results: Maximum number of results to return

        Returns:
            Dictionary with search results and metadata
        """
        try:
            if not self.query_generator:
                return {
                    "success": False,
                    "message": "Query generator not available. Please check your OpenAI API key configuration.",
                    "data": None,
                    "row_count": 0,
                }

            if not self.db_engine:
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
                self.query_generator.generate_and_validate(query, schema_context)
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
            execution_result = executor_with_limit.execute_query(
                sql_query, self.db_engine
            )

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

    def analyze_properties(self, query: str) -> Dict[str, Any]:
        """
        Execute property analysis for aggregation/statistical queries.

        Args:
            query: Natural language query requesting statistical analysis

        Returns:
            Dictionary with analysis results and metadata
        """
        try:
            if not self.db_engine:
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
            with self.db_engine.connect() as conn:
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
            stats_data = (
                df.to_dict("records")[0] if len(df) == 1 else df.to_dict("records")
            )

            return {
                "success": True,
                "message": "Analysis completed successfully",
                "data": stats_data,
                "stats": stats_data,
                "sql_query": sql_query,
                "row_count": len(df),
            }

        except ImportError:
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

    def list_values(self, query: str) -> Dict[str, Any]:
        """
        Execute the list values query using ListValuesGenerator and QueryExecutor.

        Args:
            query: Natural language query asking for distinct values

        Returns:
            Dictionary with success status, message, and list of values
        """
        try:
            if not self.db_engine:
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
            success, sql_query, validation_message = (
                list_generator.generate_and_validate(query, schema_context)
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
            execution_result = query_executor.execute_query(sql_query, self.db_engine)

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
            features = geojson_data.get("features", [])
            values_list = []

            for feature in features:
                properties = feature.get("properties", {})
                if properties:
                    # Get the first value from properties (our distinct column value)
                    first_key = next(iter(properties.keys()))
                    value = properties[first_key]
                    if value is not None:
                        values_list.append(value)

            # Remove duplicates and sort
            values_list = list(set(values_list))

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

    def find_specific_property(
        self, lookup_value: str, lookup_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Find a specific property by APN, address, or coordinates with optimized queries.

        Args:
            lookup_value: The value to search for (APN, address, or coordinates)
            lookup_type: Type of lookup ('apn', 'address', 'coordinates', 'auto')

        Returns:
            Dictionary with search results and metadata
        """
        try:
            if not self.db_engine:
                return {
                    "success": False,
                    "message": "Database connection not available. Please check your database configuration.",
                    "data": None,
                    "row_count": 0,
                    "sql_query": None,
                }

            # Auto-detect lookup type if not specified
            if lookup_type == "auto":
                lookup_type = self.detect_lookup_type(lookup_value)

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
                if not self.query_generator:
                    return {
                        "success": False,
                        "message": "Query generator not available for address lookup.",
                        "data": None,
                        "row_count": 0,
                        "sql_query": None,
                    }

                # Get schema context and let LLM handle the address parsing
                schema_context = get_llm_schema_context("parcels")

                # Create a focused query for the specific address
                address_query = f"Find the exact property at address: {lookup_value}"
                success, sql_or_error, validation_message = (
                    self.query_generator.generate_and_validate(
                        address_query, schema_context
                    )
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
                coords = self.parse_coordinates(lookup_value)
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
            execution_result = executor.execute_query(sql_query, self.db_engine)

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

    @staticmethod
    def detect_lookup_type(value: str) -> str:
        """Detect the type of lookup based on the input value."""
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

    @staticmethod
    def parse_coordinates(coord_str: str) -> Optional[Tuple[float, float]]:
        """Parse coordinate string into (lat, lon) tuple."""
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
