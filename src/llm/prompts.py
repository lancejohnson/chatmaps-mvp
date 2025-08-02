"""
Prompt templates and examples for LLM query generation.
"""

SYSTEM_PROMPT_TEMPLATE = """You are a PostGIS SQL expert for Santa Clara County parcel data.

RULES:
- Generate ONLY SELECT statements
- Use PostGIS spatial functions when appropriate  
- Always include geometry column for map display (use ST_AsGeoJSON(wkb_geometry) as geometry)
- Limit results to < 1000 unless requested (use LIMIT clause)
- Use provided schema context for accurate column names
- Return valid SQL only, no explanations or markdown formatting

SCHEMA: {schema_context}

SPATIAL QUERY PATTERNS:
- Large parcels: ST_Area(wkb_geometry) > 4047 (1 acre = 4047 sq meters)
- Parcels in city: situs_city_name ILIKE '%city_name%'
- Parcels near point: ST_DWithin(wkb_geometry, ST_Point(lon, lat, 4326), distance_meters)
- Parcels containing point: ST_Contains(wkb_geometry, ST_Point(lon, lat, 4326))
- Parcels intersecting area: ST_Intersects(wkb_geometry, other_geometry)

REQUIRED COLUMNS:
- Always include: ogc_fid, apn, situs_street_name, situs_city_name
- Always include: ST_AsGeoJSON(wkb_geometry) as geometry
- Optional: situs_house_number, situs_zip_code

Return valid SQL only."""

FEW_SHOT_EXAMPLES = [
    {
        "user": "Find large parcels over 1 acre",
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels WHERE ST_Area(wkb_geometry) > 4047 LIMIT 500"
    },
    {
        "user": "Show me parcels in San Jose", 
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels WHERE situs_city_name ILIKE '%san jose%' LIMIT 500"
    },
    {
        "user": "Find parcels on Main Street",
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, situs_house_number, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels WHERE situs_street_name ILIKE '%main%' LIMIT 500"
    },
    {
        "user": "Show parcels in Palo Alto larger than half an acre",
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels WHERE situs_city_name ILIKE '%palo alto%' AND ST_Area(wkb_geometry) > 2023 LIMIT 500"
    },
    {
        "user": "Find all parcels in zip code 95110",
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, situs_zip_code, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels WHERE situs_zip_code = '95110' LIMIT 500"
    }
]


def create_chat_messages(user_prompt: str, schema_context: str) -> list:
    """Create OpenAI chat messages with system prompt and few-shot examples."""
    
    messages = [
        {
            "role": "system", 
            "content": SYSTEM_PROMPT_TEMPLATE.format(schema_context=schema_context)
        }
    ]
    
    # Add few-shot examples
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["sql"]})
    
    # Add the actual user query
    messages.append({"role": "user", "content": user_prompt})
    
    return messages