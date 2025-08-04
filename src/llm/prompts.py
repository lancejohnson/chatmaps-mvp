"""
Prompt templates and examples for LLM query generation.
"""

SYSTEM_PROMPT_TEMPLATE = """You are a PostGIS SQL expert for Santa Clara County parcel data.

RULES:
- **STEP 1: EXTRACT THE LIMIT NUMBER FROM USER QUERY BEFORE ANYTHING ELSE!**
- Find numbers like: "find 3", "show me 10", "first 25", "limit to 100"
- "properties" and "parcels" mean the same thing - both refer to database records
- Parse phrases: "a few"=10, "several"=20, "many"=100  
- **THE NUMBER IN USER QUERY = THE LIMIT VALUE - USE IT EXACTLY!**
- If no limit specified, use LIMIT 100
- Generate ONLY SELECT statements
- Use PostGIS spatial functions when appropriate  
- Always include geometry column for map display (use ST_AsGeoJSON(wkb_geometry) as geometry)

- Use provided schema context for accurate column names
- Return valid SQL only, no explanations or markdown formatting

SCHEMA: {schema_context}


SPATIAL QUERY PATTERNS:
- Large parcels: ST_Area(wkb_geometry::geography) > 4047 (1 acre = 4047 sq meters)
- Parcels in city: situs_city_name ILIKE '%city_name%'
- Parcels near point: ST_DWithin(wkb_geometry, ST_Point(lon, lat, 4326), distance_meters)
- Parcels containing point: ST_Contains(wkb_geometry, ST_Point(lon, lat, 4326))
- Parcels intersecting area: ST_Intersects(wkb_geometry, other_geometry)

REQUIRED COLUMNS:
- Always include: ogc_fid, apn, situs_street_name, situs_city_name
- Always include: ST_AsGeoJSON(wkb_geometry) as geometry
- Optional: situs_house_number, situs_zip_code

**FINAL REMINDER: EXTRACT THE EXACT NUMBER FROM USER QUERY FOR LIMIT CLAUSE!**
Examples: "find 2" = LIMIT 2, "show 3" = LIMIT 3, "find two" = LIMIT 2, no number mentioned = LIMIT 100

Return valid SQL only."""

FEW_SHOT_EXAMPLES = [
    {
        "user": "Find 3 properties over 3 acres",
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels WHERE ST_Area(wkb_geometry::geography) > 12141 LIMIT 3",
    },
    {
        "user": "Show me 10 parcels over 10 acres",
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels WHERE ST_Area(wkb_geometry::geography) > 40470 LIMIT 10",
    },
    {
        "user": "Find the first 10 large parcels over 1 acre",
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels WHERE ST_Area(wkb_geometry::geography) > 4047 LIMIT 10",
    },
    {
        "user": "Show me 25 parcels in San Jose",
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels WHERE situs_city_name ILIKE '%san jose%' LIMIT 25",
    },
    {
        "user": "Find a few parcels on Main Street",
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, situs_house_number, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels WHERE situs_street_name ILIKE '%main%' LIMIT 10",
    },
    {
        "user": "Show parcels in Palo Alto larger than half an acre",
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels WHERE situs_city_name ILIKE '%palo alto%' AND ST_Area(wkb_geometry::geography) > 2023 LIMIT 100",
    },
    {
        "user": "Find several parcels in zip code 95110",
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, situs_zip_code, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels WHERE situs_zip_code = '95110' LIMIT 20",
    },
    {
        "user": "Show me exactly 100 parcels with the largest areas",
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels ORDER BY ST_Area(wkb_geometry::geography) DESC LIMIT 100",
    },
]


def create_chat_messages(user_prompt: str, schema_context: str) -> list:
    """Create OpenAI chat messages with system prompt and few-shot examples."""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TEMPLATE.format(schema_context=schema_context),
        }
    ]

    # Add few-shot examples
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["sql"]})

    # Add the actual user query
    messages.append({"role": "user", "content": user_prompt})

    return messages
