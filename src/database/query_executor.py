"""
Query executor for safe PostGIS query execution with GeoJSON conversion.
"""

import re
import pandas as pd
from typing import Dict, Any, Tuple
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
import json


def validate_sql_safety(sql_query: str) -> Tuple[bool, str]:
    """
    Validate that the SQL query is safe to execute.

    Only SELECT statements are allowed, and dangerous operations are blocked.

    Args:
        sql_query: SQL query string to validate

    Returns:
        Tuple of (is_safe: bool, message: str)
    """
    if not sql_query or not sql_query.strip():
        return False, "Empty query provided"

    # Normalize the query
    sql_upper = sql_query.upper().strip()

    # Must start with SELECT
    if not sql_upper.startswith("SELECT"):
        return False, "Only SELECT queries are allowed"

    # Check for dangerous operations
    dangerous_keywords = [
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "CREATE",
        "ALTER",
        "TRUNCATE",
        "EXEC",
        "EXECUTE",
        "DECLARE",
        "GRANT",
        "REVOKE",
    ]

    for keyword in dangerous_keywords:
        # Use word boundaries to avoid false positives
        if re.search(rf"\b{keyword}\b", sql_upper):
            return False, f"Operation {keyword} is not allowed"

    # Check for semicolons (which could indicate multiple statements)
    semicolon_count = sql_query.count(";")
    if semicolon_count > 1 or (
        semicolon_count == 1 and not sql_query.strip().endswith(";")
    ):
        return False, "Multiple statements or unexpected semicolons detected"

    # Check for common SQL injection patterns
    injection_patterns = [
        r"--",  # SQL comments
        r"/\*.*?\*/",  # Block comments
        r"xp_",  # Extended stored procedures
        r"sp_",  # System stored procedures
    ]

    for pattern in injection_patterns:
        if re.search(pattern, sql_query, re.IGNORECASE):
            return False, f"Potentially unsafe pattern detected: {pattern}"

    return True, "Query is safe"


class QueryExecutor:
    """
    Executes validated SQL queries against PostGIS and converts results to GeoJSON.
    """

    def __init__(self, max_results: int = 1000):
        """
        Initialize the query executor.

        Args:
            max_results: Maximum number of results to return (default: 1000)
        """
        self.max_results = max_results

    def execute_query(self, sql_query: str, engine: Engine) -> Dict[str, Any]:
        """
        Execute a validated SQL query and return results with metadata.

        Args:
            sql_query: SQL query to execute (must be pre-validated)
            engine: SQLAlchemy database engine

        Returns:
            Dictionary containing:
            - success: bool
            - data: GeoJSON FeatureCollection or None
            - row_count: int
            - message: str
            - sql_executed: str

        Raises:
            ValueError: If query validation fails
        """
        # Validate query safety first
        is_safe, safety_message = validate_sql_safety(sql_query)
        if not is_safe:
            return {
                "success": False,
                "data": None,
                "row_count": 0,
                "message": f"Query validation failed: {safety_message}",
                "sql_executed": sql_query,
            }

        try:
            # Add LIMIT if not present and query doesn't already have one
            limited_query = self._ensure_query_limit(sql_query)

            # Execute the query
            with engine.connect() as conn:
                df = pd.read_sql(limited_query, conn)

            if df.empty:
                return {
                    "success": True,
                    "data": self._create_empty_geojson(),
                    "row_count": 0,
                    "message": "Query executed successfully but returned no results",
                    "sql_executed": limited_query,
                }

            # Convert to GeoJSON
            geojson_data = self._dataframe_to_geojson(df)

            return {
                "success": True,
                "data": geojson_data,
                "row_count": len(df),
                "message": f"Query executed successfully, returned {len(df)} rows",
                "sql_executed": limited_query,
            }

        except SQLAlchemyError as e:
            return {
                "success": False,
                "data": None,
                "row_count": 0,
                "message": f"Database error: {str(e)}",
                "sql_executed": sql_query,
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "row_count": 0,
                "message": f"Unexpected error: {str(e)}",
                "sql_executed": sql_query,
            }

    def _ensure_query_limit(self, sql_query: str) -> str:
        """Add or replace LIMIT clause to respect max_results setting."""
        import re

        print(
            f"DEBUG - QueryExecutor: Input query='{sql_query}', max_results={self.max_results}"
        )

        sql_upper = sql_query.upper()
        query_stripped = sql_query.strip()
        if query_stripped.endswith(";"):
            query_stripped = query_stripped[:-1]

        # If query already has LIMIT, extract the existing limit and use the minimum
        if "LIMIT" in sql_upper:
            # Use regex to find and replace the LIMIT clause
            limit_pattern = r"\bLIMIT\s+(\d+)\b"
            match = re.search(limit_pattern, sql_query, re.IGNORECASE)
            if match:
                existing_limit = int(match.group(1))
                # Use the smaller of existing limit and max_results
                final_limit = min(existing_limit, self.max_results)
                print(
                    f"DEBUG - Replacing LIMIT {existing_limit} with LIMIT {final_limit}"
                )
                # Replace the existing LIMIT with the final limit
                new_query = re.sub(
                    limit_pattern,
                    f"LIMIT {final_limit}",
                    sql_query,
                    flags=re.IGNORECASE,
                )
                result = new_query.strip()
                print(f"DEBUG - Final query: '{result}'")
                return result

        # Add LIMIT clause if not present
        result = f"{query_stripped} LIMIT {self.max_results};"
        print(f"DEBUG - Added LIMIT: '{result}'")
        return result

    def _create_empty_geojson(self) -> Dict[str, Any]:
        """Create an empty GeoJSON FeatureCollection."""
        return {"type": "FeatureCollection", "features": []}

    def _dataframe_to_geojson(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Convert a pandas DataFrame to GeoJSON FeatureCollection.

        Expects PostGIS ST_AsGeoJSON() to have already converted geometry to JSON strings.
        Looks for columns containing 'geometry' in the name.
        """
        features = []

        # Find geometry column(s) - could be 'geometry', 'geometry_json', etc.
        geometry_cols = [col for col in df.columns if "geometry" in col.lower()]
        geometry_col = geometry_cols[0] if geometry_cols else None

        for _, row in df.iterrows():
            try:
                # Parse geometry if available
                geometry = None
                if geometry_col and not pd.isna(row[geometry_col]):
                    geometry_str = row[geometry_col]
                    if isinstance(geometry_str, str) and geometry_str.strip():
                        geometry = json.loads(geometry_str)

                # Create properties (all columns except geometry)
                properties = {}
                for col in df.columns:
                    if col != geometry_col:
                        properties[col] = self._serialize_value(row[col])

                feature = {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": properties,
                }

                features.append(feature)

            except (json.JSONDecodeError, Exception):
                # If geometry parsing fails, create feature without geometry
                properties = {}
                for col in df.columns:
                    if col != geometry_col:
                        properties[col] = self._serialize_value(row[col])

                feature = {
                    "type": "Feature",
                    "geometry": None,
                    "properties": properties,
                }
                features.append(feature)

        return {"type": "FeatureCollection", "features": features}

    def _serialize_value(self, value) -> Any:
        """Convert pandas/numpy values to JSON-serializable types."""
        if pd.isna(value):
            return None
        elif hasattr(value, "item"):  # numpy scalar
            return value.item()
        elif isinstance(value, (list, dict)):
            return value
        else:
            return str(value)
