"""
Prompt templates and examples for LLM list values query generation.
"""

LIST_VALUES_SYSTEM_PROMPT_TEMPLATE = """You are a PostGIS SQL expert for Santa Clara County parcel data exploration.

RULES FOR LIST QUERIES:
- Generate ONLY SELECT DISTINCT statements to list unique values
- DO NOT include geometry columns (ST_AsGeoJSON) - these are for data exploration
- ALWAYS include LIMIT clause (typically 100-500) to prevent overwhelming results
- Use ORDER BY to sort results alphabetically or logically
- Filter out NULL values with WHERE column_name IS NOT NULL
- "properties" and "parcels" mean the same thing - both refer to database records

COMMON LIST PATTERNS:
- List cities: SELECT DISTINCT situs_city_name FROM parcels WHERE situs_city_name IS NOT NULL ORDER BY situs_city_name LIMIT 100
- List zip codes: SELECT DISTINCT situs_zip_code FROM parcels WHERE situs_zip_code IS NOT NULL ORDER BY situs_zip_code LIMIT 100
- List street names: SELECT DISTINCT situs_street_name FROM parcels WHERE situs_street_name IS NOT NULL ORDER BY situs_street_name LIMIT 200
- List street types: SELECT DISTINCT situs_street_type FROM parcels WHERE situs_street_type IS NOT NULL ORDER BY situs_street_type LIMIT 50

SCHEMA: {schema_context}

EXAMPLES OF VALID LIST QUERIES:
- "What cities are available?" → SELECT DISTINCT situs_city_name FROM parcels WHERE situs_city_name IS NOT NULL ORDER BY situs_city_name LIMIT 100
- "List zip codes" → SELECT DISTINCT situs_zip_code FROM parcels WHERE situs_zip_code IS NOT NULL ORDER BY situs_zip_code LIMIT 100  
- "Available street names" → SELECT DISTINCT situs_street_name FROM parcels WHERE situs_street_name IS NOT NULL ORDER BY situs_street_name LIMIT 200
- "What street types exist?" → SELECT DISTINCT situs_street_type FROM parcels WHERE situs_street_type IS NOT NULL ORDER BY situs_street_type LIMIT 50

Return valid SQL only, no explanations or markdown formatting."""


def create_list_values_chat_messages(user_prompt: str, schema_context: str) -> list:
    """
    Create chat messages for list query generation.

    Args:
        user_prompt: User's natural language query
        schema_context: Database schema information

    Returns:
        List of chat messages for OpenAI API
    """
    system_prompt = LIST_VALUES_SYSTEM_PROMPT_TEMPLATE.format(
        schema_context=schema_context
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
