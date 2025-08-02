# LLM Query Integration Plan

## Goal

Integrate GPT-4o to convert natural language chat input into PostGIS SQL queries, execute them, and display results on the map.

## Current State ✅

- PostGIS database with Santa Clara County parcel data
- Streamlit chat interface (placeholder responses)
- Database connection (`get_db_engine()`)
- Rich schema context (`get_llm_schema_context()`)
- Map visualization with Folium

## What to Add

### 1. LLM Query Generator (`src/llm/query_generator.py`)

```python
class QueryGenerator:
    def generate_sql_query(user_prompt, schema_context):
        # System prompt + few-shot examples + schema context
        # GPT-4.1 API call
        # Return SQL query string
    
    def validate_query(sql_query):
        # SELECT-only validation
        # SQL injection prevention
```

### 2. Query Executor (`src/database/query_executor.py`)

```python
class QueryExecutor:
    def execute_query(sql_query, engine):
        # Execute against PostGIS
        # Convert to GeoJSON for map
        # Error handling
```

### 3. Enhanced Chat Handler (modify `app.py` lines 307-313)

```python
def handle_chat_query(prompt):
    # 1. Get schema context
    # 2. Generate SQL via LLM
    # 3. Validate & execute query  
    # 4. Update map with results
    # 5. Display summary + SQL
```

## Implementation Phases

### Phase 1: Core LLM Integration ✅

- [x] Add `openai>=1.0.0` to requirements.txt
- [x] Create `src/llm/query_generator.py`
- [x] Add `OPENAI_API_KEY` environment variable
- [x] Implement basic prompt → SQL generation
- [x] Run a few tests to see if they make sense

### Phase 2: Query Execution ✅

- [x] Create `src/database/query_executor.py`
- [x] Add SQL safety validation (SELECT-only)
- [x] Implement query execution with PostGIS

### Phase 3: UI Integration ✅ COMPLETED

- [x] Modify chat handler in `app.py`
- [x] Add query results to map visualization
- [x] Display query summary and generated SQL
- [x] Add loading states and error handling

### Phase 4: Polish

- [ ] Query result caching
- [ ] Export functionality
- [ ] Advanced spatial query patterns

## Key Technical Components

### System Prompt Template

```text
You are a PostGIS SQL expert for Santa Clara County parcel data.
RULES:
- Generate ONLY SELECT statements
- Use PostGIS spatial functions when appropriate  
- Always include geometry column for map display
- Limit results to < 1000 unless requested
- Use provided schema context for accurate column names

SCHEMA: {schema_context}

EXAMPLES:
- "large parcels" → ST_Area(wkb_geometry) > 4047
- "parcels in San Jose" → situs_city_name ILIKE '%san jose%'
- "parcels near [address]" → ST_DWithin(wkb_geometry, point, distance)

Return valid SQL only.
```

### Few-Shot Examples

```python
EXAMPLES = [
    {
        "user": "Find large parcels over 1 acre",
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels WHERE ST_Area(wkb_geometry) > 4047 LIMIT 500"
    },
    {
        "user": "Show me parcels in San Jose", 
        "sql": "SELECT ogc_fid, apn, situs_street_name, situs_city_name, ST_AsGeoJSON(wkb_geometry) as geometry FROM parcels WHERE situs_city_name ILIKE '%san jose%' LIMIT 500"
    }
]
```

### Safety Validation

```python
def validate_sql_safety(sql_query):
    dangerous = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER']
    sql_upper = sql_query.upper()
    
    if not sql_upper.strip().startswith('SELECT'):
        return False, "Only SELECT queries allowed"
    
    for keyword in dangerous:
        if keyword in sql_upper:
            return False, f"Operation {keyword} not allowed"
    
    return True, "Query is safe"
```

## Environment Variables

```bash
OPENAI_API_KEY=your_gpt4o_key_here
LLM_MODEL=gpt-4o
MAX_QUERY_RESULTS=1000
```

## New Dependencies

```bash
pip install openai>=1.0.0
pip install sqlparse
```

## Expected User Flow

1. User: *"Show me large parcels in Palo Alto"*
2. LLM generates: `SELECT ... WHERE situs_city_name ILIKE '%palo alto%' AND ST_Area(wkb_geometry) > 2000`
3. Query executes against PostGIS
4. Results highlight on map
5. Summary: "Found 45 parcels in Palo Alto over 2000 sq ft"
6. Show generated SQL for transparency

## File Structure After Implementation

```text
chatmap-mvp/
├── app.py (modified)
├── schema.py (existing)
├── src/
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── query_generator.py
│   │   └── prompts.py
│   └── database/
│       ├── __init__.py
│       └── query_executor.py
```

## Next Steps

1. Start with Phase 1: Set up OpenAI integration
2. Test basic query generation with simple examples
3. Add query execution and map integration
4. Iterate on prompt engineering for better results
