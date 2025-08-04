"""
OpenAI function calling tool definitions for ChatMap MVP.
"""

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

# List of all available tools for easy importing
ALL_TOOLS = [
    SEARCH_PROPERTIES_TOOL,
    ANALYZE_PROPERTIES_TOOL,
    LIST_VALUES_TOOL,
    FIND_SPECIFIC_PROPERTY_TOOL,
]
