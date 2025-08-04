"""
LLM-powered SQL list query generator for getting distinct values from database columns.
"""

import os
import re
from typing import Tuple, Optional
from openai import OpenAI


class ListGenerator:
    """Generates PostGIS SQL queries for listing distinct values from natural language using OpenAI GPT models."""

    def __init__(self, model: str = "gpt-4.1", api_key: Optional[str] = None):
        """
        Initialize the list query generator.

        Args:
            model: OpenAI model to use (default: gpt-4.1)
            api_key: OpenAI API key (default: reads from OPENAI_API_KEY env var)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        if not self.client.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )

    def generate_sql_query(self, user_prompt: str, schema_context: str) -> str:
        """
        Generate a PostGIS SQL query for listing distinct values from a natural language prompt.

        Args:
            user_prompt: Natural language description of what distinct values to list
            schema_context: Database schema information for the LLM

        Returns:
            Generated SQL query string for listing values

        Raises:
            Exception: If API call fails or query generation fails
        """
        try:
            # Create chat messages with list-specific system prompt and examples
            messages = self._create_list_chat_messages(user_prompt, schema_context)

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent SQL generation
                max_tokens=500,  # Sufficient for most list queries
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            # Extract the generated SQL
            sql_query = response.choices[0].message.content.strip()

            # Clean up the response (remove any markdown formatting)
            sql_query = self._clean_sql_response(sql_query)

            return sql_query

        except Exception as e:
            raise Exception(f"Failed to generate list SQL query: {str(e)}")

    def validate_query(self, sql_query: str) -> Tuple[bool, str]:
        """
        Validate that the generated list SQL query is safe to execute.

        Args:
            sql_query: SQL query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Remove comments and normalize whitespace
        normalized_query = re.sub(r"--.*?\n", "", sql_query)
        normalized_query = re.sub(r"/\*.*?\*/", "", normalized_query, flags=re.DOTALL)
        normalized_query = " ".join(normalized_query.split())

        # Convert to uppercase for checking
        query_upper = normalized_query.upper()

        # Must start with SELECT
        if not query_upper.strip().startswith("SELECT"):
            return False, "Query must start with SELECT statement"

        # Check for dangerous operations
        dangerous_keywords = [
            "DROP",
            "DELETE",
            "UPDATE",
            "INSERT",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "MERGE",
            "REPLACE",
            "CALL",
            "EXEC",
            "EXECUTE",
        ]

        for keyword in dangerous_keywords:
            if f" {keyword} " in f" {query_upper} " or query_upper.startswith(
                f"{keyword} "
            ):
                return False, f"Operation {keyword} not allowed"

        # Check if this looks like a list/distinct query
        if "DISTINCT" not in query_upper:
            return False, "Query must include DISTINCT to list unique values"

        # For list queries, we should have reasonable limits to prevent overwhelming results
        if "LIMIT" not in query_upper:
            return False, "Query must include LIMIT clause to prevent large result sets"

        return True, "List query is valid"

    def _clean_sql_response(self, sql_response: str) -> str:
        """
        Clean up the LLM response to extract pure SQL.

        Args:
            sql_response: Raw response from LLM

        Returns:
            Cleaned SQL query
        """
        # Remove markdown code blocks
        sql_response = re.sub(r"```sql\s*", "", sql_response)
        sql_response = re.sub(r"```\s*", "", sql_response)

        # Remove any leading/trailing explanations
        lines = sql_response.strip().split("\n")
        sql_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines and lines that look like explanations
            if (
                line
                and not line.startswith("#")
                and not line.startswith("//")
                and not line.lower().startswith("this query")
                and not line.lower().startswith("the query")
            ):
                sql_lines.append(line)

        return " ".join(sql_lines)

    def _create_list_chat_messages(self, user_prompt: str, schema_context: str) -> list:
        """
        Create chat messages for list query generation.

        Args:
            user_prompt: User's natural language query
            schema_context: Database schema information

        Returns:
            List of chat messages for OpenAI API
        """
        system_prompt = f"""You are a PostGIS SQL expert for Santa Clara County parcel data exploration.

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

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_and_validate(
        self, user_prompt: str, schema_context: str
    ) -> Tuple[bool, str, str]:
        """
        Generate list SQL query and validate it in one step.

        Args:
            user_prompt: Natural language list query
            schema_context: Database schema information

        Returns:
            Tuple of (success, sql_query_or_error, validation_message)
        """
        try:
            # Generate the query
            sql_query = self.generate_sql_query(user_prompt, schema_context)

            # Validate the query
            is_valid, validation_message = self.validate_query(sql_query)

            if is_valid:
                return True, sql_query, validation_message
            else:
                return False, validation_message, f"Generated query: {sql_query}"

        except Exception as e:
            return False, str(e), "List query generation failed"
