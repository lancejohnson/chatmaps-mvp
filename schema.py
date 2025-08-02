"""
Database schema definitions with semantic information for LLM query generation.

This module defines the structure and meaning of database tables to enable
both programmatic access and LLM-assisted query generation.
"""

from typing import Dict, List, Any

# Rich schema dictionary for parcels table
PARCELS_SCHEMA = {
    "table_name": "parcels",
    "table_description": "Property parcels in Santa Clara County with address and geometric information",
    "primary_key": "ogc_fid",
    "geometry_column": "wkb_geometry",
    "display_column": "apn",  # Best column for showing to users
    "search_columns": [
        "apn",
        "situs_street_name",
        "situs_house_number",
        "situs_city_name",
    ],
    "columns": {
        "ogc_fid": {
            "type": "integer",
            "description": "Unique database ID for each parcel record",
            "primary_key": True,
            "nullable": False,
            "good_for": ["joins", "internal_reference", "sorting"],
            "bad_for": ["user_display", "search"],
            "example": "12345",
        },
        "apn": {
            "type": "string",
            "description": "Assessor Parcel Number - the official property identifier used by the county",
            "nullable": True,
            "good_for": ["search", "display", "user_queries", "parcel_lookup"],
            "bad_for": ["calculations"],
            "example": "123-45-678",
            "search_tips": "Users often search by APN when they know the specific parcel number",
        },
        "objectid": {
            "type": "string",
            "description": "Alternative object identifier, may be from original data source",
            "nullable": True,
            "good_for": ["reference", "data_lineage"],
            "bad_for": ["user_display"],
            "example": "OBJ_12345",
        },
        "situs_house_number": {
            "type": "string",
            "description": "House number of the property address",
            "nullable": True,
            "good_for": ["address_search", "display", "address_queries"],
            "bad_for": ["calculations"],
            "example": "123",
            "search_tips": "Combine with street name for full address searches",
        },
        "situs_house_number_suffix": {
            "type": "string",
            "description": "House number suffix (like A, B, 1/2)",
            "nullable": True,
            "good_for": ["address_search", "display"],
            "example": "A",
        },
        "situs_street_name": {
            "type": "string",
            "description": "Street name of the property address",
            "nullable": True,
            "good_for": ["search", "display", "address_queries", "area_analysis"],
            "bad_for": ["calculations"],
            "example": "Main Street",
            "search_tips": "Popular for finding all properties on a street",
        },
        "situs_street_type": {
            "type": "string",
            "description": "Street type (St, Ave, Blvd, etc.)",
            "nullable": True,
            "good_for": ["address_display", "address_search"],
            "example": "St",
        },
        "situs_street_direction": {
            "type": "string",
            "description": "Street direction (N, S, E, W)",
            "nullable": True,
            "good_for": ["address_display", "address_search"],
            "example": "N",
        },
        "situs_unit_number": {
            "type": "string",
            "description": "Unit or apartment number",
            "nullable": True,
            "good_for": ["address_display", "multi_unit_properties"],
            "example": "Unit 5",
        },
        "situs_city_name": {
            "type": "string",
            "description": "City name of the property",
            "nullable": True,
            "good_for": ["search", "display", "geographic_filtering", "city_analysis"],
            "bad_for": ["calculations"],
            "example": "San Jose",
            "search_tips": "Great for filtering by municipality or analyzing city-wide data",
        },
        "situs_state_code": {
            "type": "string",
            "description": "State code (usually CA for California)",
            "nullable": True,
            "good_for": ["address_display"],
            "example": "CA",
        },
        "situs_zip_code": {
            "type": "string",
            "description": "ZIP code of the property",
            "nullable": True,
            "good_for": ["search", "geographic_filtering", "area_analysis"],
            "bad_for": ["calculations"],
            "example": "95110",
            "search_tips": "Useful for neighborhood-level analysis",
        },
        "number_of_situs_address": {
            "type": "string",
            "description": "Count of addresses associated with this parcel",
            "nullable": True,
            "good_for": ["multi_address_analysis"],
            "example": "1",
        },
        "tax_rate_area": {
            "type": "string",
            "description": "Tax rate area code for property tax calculations",
            "nullable": True,
            "good_for": ["tax_analysis", "administrative_boundaries"],
            "example": "TRA001",
        },
        "ap_lp": {
            "type": "string",
            "description": "Unknown field - possibly administrative code",
            "nullable": True,
            "good_for": ["administrative_reference"],
            "bad_for": ["user_queries"],
        },
        "shape_area": {
            "type": "string",
            "description": "Calculated area of the parcel (stored as string)",
            "nullable": True,
            "good_for": ["size_analysis", "property_comparison"],
            "bad_for": ["precise_calculations"],
            "example": "7500.25",
            "search_tips": "Note: stored as string, may need conversion for calculations",
        },
        "shape_length": {
            "type": "string",
            "description": "Calculated perimeter of the parcel (stored as string)",
            "nullable": True,
            "good_for": ["perimeter_analysis"],
            "bad_for": ["precise_calculations"],
            "example": "350.75",
        },
        "reserved1": {
            "type": "string",
            "description": "Reserved field - purpose unknown",
            "nullable": True,
            "good_for": [],
            "bad_for": ["user_queries", "analysis"],
        },
        "reserved2": {
            "type": "string",
            "description": "Reserved field - purpose unknown",
            "nullable": True,
            "good_for": [],
            "bad_for": ["user_queries", "analysis"],
        },
        "reserved3": {
            "type": "string",
            "description": "Reserved field - purpose unknown",
            "nullable": True,
            "good_for": [],
            "bad_for": ["user_queries", "analysis"],
        },
        "wkb_geometry": {
            "type": "geometry(MultiPolygon,4326)",
            "description": "PostGIS geometry column containing the parcel boundaries",
            "nullable": True,
            "good_for": [
                "spatial_queries",
                "mapping",
                "area_calculations",
                "geometric_analysis",
            ],
            "bad_for": ["text_search", "simple_display"],
            "example": "MULTIPOLYGON(...)",
            "search_tips": "Use PostGIS functions like ST_Contains, ST_Intersects for spatial queries",
        },
    },
}

# Main schema registry
DATABASE_SCHEMA = {"parcels": PARCELS_SCHEMA}


# Helper functions for schema access
def get_table_schema(table_name: str) -> Dict[str, Any]:
    """Get schema information for a specific table."""
    return DATABASE_SCHEMA.get(table_name, {})


def get_column_info(table_name: str, column_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific column."""
    table_schema = get_table_schema(table_name)
    return table_schema.get("columns", {}).get(column_name, {})


def get_primary_key(table_name: str) -> str:
    """Get the primary key column name for a table."""
    table_schema = get_table_schema(table_name)
    return table_schema.get("primary_key", "")


def get_geometry_column(table_name: str) -> str:
    """Get the geometry column name for a table."""
    table_schema = get_table_schema(table_name)
    return table_schema.get("geometry_column", "")


def get_display_column(table_name: str) -> str:
    """Get the best column for displaying to users."""
    table_schema = get_table_schema(table_name)
    return table_schema.get("display_column", "")


def get_search_columns(table_name: str) -> List[str]:
    """Get columns that are good for user searches."""
    table_schema = get_table_schema(table_name)
    return table_schema.get("search_columns", [])


def get_llm_schema_context(table_name: str) -> str:
    """Generate LLM-friendly schema description for query generation."""
    table_schema = get_table_schema(table_name)
    if not table_schema:
        return f"No schema found for table: {table_name}"

    context = f"""
Table: {table_schema['table_name']}
Description: {table_schema['table_description']}
Primary Key: {table_schema.get('primary_key', 'Unknown')}
Geometry Column: {table_schema.get('geometry_column', 'None')}

Key Columns for Different Purposes:

SEARCH & DISPLAY:
"""

    # Add search columns
    search_cols = table_schema.get("search_columns", [])
    for col in search_cols:
        col_info = table_schema["columns"].get(col, {})
        context += f"- {col}: {col_info.get('description', 'No description')}\n"

    context += "\nGEOMETRY & SPATIAL:\n"
    geom_col = table_schema.get("geometry_column")
    if geom_col:
        geom_info = table_schema["columns"].get(geom_col, {})
        context += f"- {geom_col}: {geom_info.get('description', 'Geometry column')}\n"

    context += "\nOTHER USEFUL COLUMNS:\n"
    for col_name, col_info in table_schema["columns"].items():
        if col_name not in search_cols and col_name != geom_col:
            if "user_queries" in col_info.get(
                "good_for", []
            ) or "analysis" in col_info.get("good_for", []):
                context += (
                    f"- {col_name}: {col_info.get('description', 'No description')}\n"
                )

    return context


def validate_columns(table_name: str, columns: List[str]) -> Dict[str, bool]:
    """Validate that columns exist in the table schema."""
    table_schema = get_table_schema(table_name)
    table_columns = table_schema.get("columns", {})

    return {col: col in table_columns for col in columns}
