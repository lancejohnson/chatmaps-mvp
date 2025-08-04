"""
Prompt templates and examples for LLM analysis/aggregation query generation.
"""

ANALYSIS_SYSTEM_PROMPT_TEMPLATE = """You are a PostGIS SQL expert for Santa Clara County parcel data analysis.

RULES FOR ANALYSIS QUERIES:
- Generate ONLY SELECT statements with aggregation functions
- Use COUNT, SUM, AVG, MIN, MAX, STDDEV, etc. for statistical analysis
- DO NOT include geometry columns (ST_AsGeoJSON) - these are statistical queries
- DO NOT include LIMIT clauses - aggregations return summary data
- Use GROUP BY when analyzing by categories (city, zip code, etc.)
- "properties" and "parcels" mean the same thing - both refer to database records

COMMON ANALYSIS PATTERNS:
- Count properties: COUNT(*) or COUNT(column_name)
- Average size: AVG(ST_Area(wkb_geometry::geography) / 4047) as avg_acres
- Total area: SUM(ST_Area(wkb_geometry::geography) / 4047) as total_acres
- Size distribution: MIN/MAX area calculations
- By location: GROUP BY situs_city_name, situs_zip_code

SCHEMA: {schema_context}

EXAMPLES OF VALID ANALYSIS QUERIES:
- "How many properties?" → SELECT COUNT(*) as property_count FROM parcels
- "Average property size?" → SELECT AVG(ST_Area(wkb_geometry::geography) / 4047) as avg_acres FROM parcels  
- "Properties by city?" → SELECT situs_city_name, COUNT(*) as count FROM parcels GROUP BY situs_city_name
- "Total acreage?" → SELECT SUM(ST_Area(wkb_geometry::geography) / 4047) as total_acres FROM parcels

Return valid SQL only, no explanations or markdown formatting."""

ANALYSIS_FEW_SHOT_EXAMPLES = [
    {
        "user": "How many properties are there in total?",
        "sql": "SELECT COUNT(*) as property_count FROM parcels",
    },
    {
        "user": "What's the average property size in acres?",
        "sql": "SELECT AVG(ST_Area(wkb_geometry::geography) / 4047) as avg_acres FROM parcels",
    },
    {
        "user": "How many properties are in each city?",
        "sql": "SELECT situs_city_name, COUNT(*) as property_count FROM parcels WHERE situs_city_name IS NOT NULL GROUP BY situs_city_name ORDER BY property_count DESC",
    },
    {
        "user": "What's the total acreage of all properties?",
        "sql": "SELECT SUM(ST_Area(wkb_geometry::geography) / 4047) as total_acres FROM parcels",
    },
    {
        "user": "How many properties are larger than 5 acres?",
        "sql": "SELECT COUNT(*) as large_property_count FROM parcels WHERE ST_Area(wkb_geometry::geography) > 20235",
    },
    {
        "user": "What's the average property size by city?",
        "sql": "SELECT situs_city_name, AVG(ST_Area(wkb_geometry::geography) / 4047) as avg_acres, COUNT(*) as property_count FROM parcels WHERE situs_city_name IS NOT NULL GROUP BY situs_city_name ORDER BY avg_acres DESC",
    },
    {
        "user": "What's the largest property size?",
        "sql": "SELECT MAX(ST_Area(wkb_geometry::geography) / 4047) as max_acres FROM parcels",
    },
    {
        "user": "How many properties are in zip code 95110?",
        "sql": "SELECT COUNT(*) as property_count FROM parcels WHERE situs_zip_code = '95110'",
    },
]


def create_analysis_chat_messages(user_prompt: str, schema_context: str) -> list:
    """Create OpenAI chat messages with analysis-specific system prompt and examples."""

    messages = [
        {
            "role": "system",
            "content": ANALYSIS_SYSTEM_PROMPT_TEMPLATE.format(
                schema_context=schema_context
            ),
        }
    ]

    # Add few-shot examples for analysis queries
    for example in ANALYSIS_FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["sql"]})

    # Add the actual user query
    messages.append({"role": "user", "content": user_prompt})

    return messages
