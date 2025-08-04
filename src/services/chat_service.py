"""
Chat service for handling OpenAI function calling and chat interactions.
"""

import json
import os
from typing import Dict, List, Any

from openai import OpenAI
import streamlit as st

from src.config.tools import ALL_TOOLS
from src.services.property_service import PropertyService


class ChatService:
    """Service for handling chat interactions with OpenAI function calling."""

    def __init__(self):
        """Initialize the chat service with property service and OpenAI client."""
        self.property_service = PropertyService()
        self._openai_client = None

    @property
    def openai_client(self):
        """Get or create OpenAI client."""
        if self._openai_client is None:
            self._openai_client = self._create_openai_client()
        return self._openai_client

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

    def process_chat_with_function_calling(
        self, user_prompt: str, chat_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Process a chat message using OpenAI function calling to decide whether to search properties or just chat.

        Args:
            user_prompt: The user's message
            chat_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}]

        Returns:
            Dictionary with response and any search results
        """
        if not self.openai_client:
            return {
                "success": False,
                "response": "❌ Chat service unavailable. Please check your OpenAI API key configuration.",
                "search_results": None,
            }

        try:
            # Prepare messages for OpenAI, including chat history
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful assistant for exploring Santa Clara County property data. 

You have access to four functions. CAREFULLY choose the right one:

**find_specific_property** - Use when users provide SPECIFIC identifiers for a single property:
   - "Find parcel 123-45-678" → Direct APN lookup
   - "Show me property at 123 Main Street" → Exact address lookup
   - "What's at coordinates 37.4419, -122.1430" → Coordinate lookup
   - Any time they give you a specific APN, exact address, or lat/long coordinates

**analyze_properties** - Use for questions asking for NUMBERS/STATISTICS (no map display):
   - "How many properties..." → COUNT 
   - "What's the average..." → AVERAGE
   - "What's the total..." → SUM
   - "How many properties over X acres?" → COUNT with condition
   - "Average property size in [city]?" → AVERAGE with filter
   - Any question where the answer is a NUMBER or STATISTIC

**search_properties** - Use when users want to SEE/FIND multiple properties on the map:
   - "Show me properties over X acres" → Display on map
   - "Find properties on Main Street" → Show locations  
   - "Properties in San Jose" → Map individual parcels
   - When they want to see specific properties highlighted (but not one exact property)

**list_values** - Use when users want to EXPLORE/LIST what options are available:
   - "What cities are available?" → List distinct city names
   - "What zip codes exist?" → List unique zip codes
   - "Available street names?" → List street names
   - "What street types are there?" → List street types
   - Any question asking for distinct/unique values from the database

KEY DISTINCTIONS: 
- "Find parcel X" or "property at X address" = find_specific_property (exact lookup)
- "How many X?" = analyze_properties (returns a number)
- "Show me properties..." = search_properties (returns multiple map markers)
- "What X are available?" = list_values (returns a list of options)

For general conversation, respond normally without using functions.""",
                }
            ]

            # Add chat history (limit to last 10 messages to avoid token limits)
            messages.extend(chat_history[-10:])

            # Add the current user message
            messages.append({"role": "user", "content": user_prompt})

            # Call OpenAI with function calling
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                tools=ALL_TOOLS,
                tool_choice="auto",  # Let GPT decide when to use the tool
                temperature=0.1,
            )

            message = response.choices[0].message

            # Check if GPT wants to use any of the functions
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "search_properties":
                        return self._handle_search_properties(
                            tool_call, messages, message
                        )

                    elif tool_call.function.name == "analyze_properties":
                        return self._handle_analyze_properties(
                            tool_call, messages, message
                        )

                    elif tool_call.function.name == "list_values":
                        return self._handle_list_values(tool_call, messages, message)

                    elif tool_call.function.name == "find_specific_property":
                        return self._handle_find_specific_property(
                            tool_call, messages, message
                        )

            # If no function was called, just return the regular chat response
            return {
                "success": True,
                "response": message.content,
                "search_results": None,
                "is_chat": True,  # Flag to indicate this was just conversation
            }

        except Exception as e:
            return {
                "success": False,
                "response": f"❌ **Chat Error:** {str(e)}\n\nPlease try again or contact support if the issue persists.",
                "search_results": None,
            }

    def _handle_search_properties(self, tool_call, messages, message) -> Dict[str, Any]:
        """Handle search_properties function call."""
        try:
            args = json.loads(tool_call.function.arguments)
            search_query = args["query"]
            max_results = args.get("max_results", 100)

            # Debug: Log what function calling extracted
            print(
                f"DEBUG - Function calling extracted: query='{search_query}', max_results={max_results}"
            )

            # Execute the property search
            search_result = self.property_service.search_properties(
                search_query, max_results
            )

            # Create a follow-up message with the search results
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    {
                        "success": search_result["success"],
                        "message": search_result["message"],
                        "row_count": search_result["row_count"],
                        "sql_query": search_result.get("sql_query", ""),
                    }
                ),
            }

            # Get final response from GPT incorporating the search results
            final_messages = messages + [message, tool_message]
            final_response = self.openai_client.chat.completions.create(
                model="gpt-4.1", messages=final_messages, temperature=0.1
            )

            final_content = final_response.choices[0].message.content

            # Format the response nicely
            if search_result["success"] and search_result["row_count"] > 0:
                formatted_response = f"""✅ **Found {search_result['row_count']} propert{'y' if search_result['row_count'] == 1 else 'ies'}**

{final_content}

**SQL Query Used:**
```sql
{search_result.get('sql_query', 'N/A')}
```

🗺️ **The results are now highlighted on the map above!**"""
            elif search_result["success"] and search_result["row_count"] == 0:
                formatted_response = f"""ℹ️ **No properties found**

{final_content}

**SQL Query Used:**
```sql
{search_result.get('sql_query', 'N/A')}
```

💡 Try broadening your search criteria or checking the spelling of street names."""
            else:
                formatted_response = f"""❌ **Search Error**

{final_content}

**Error:** {search_result['message']}"""

            return {
                "success": search_result["success"],
                "response": formatted_response,
                "search_results": search_result["data"]
                if search_result["success"]
                else None,
                "row_count": search_result["row_count"],
            }

        except json.JSONDecodeError:
            return {
                "success": False,
                "response": "❌ **Error:** Could not parse search parameters. Please try rephrasing your question.",
                "search_results": None,
            }

    def _handle_analyze_properties(
        self, tool_call, messages, message
    ) -> Dict[str, Any]:
        """Handle analyze_properties function call."""
        try:
            args = json.loads(tool_call.function.arguments)
            analysis_query = args["query"]

            # Debug: Log what function calling extracted
            print(
                f"DEBUG - Function calling extracted for analysis: query='{analysis_query}'"
            )

            # Execute the property analysis
            analysis_result = self.property_service.analyze_properties(analysis_query)

            # Create a follow-up message with the analysis results
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    {
                        "success": analysis_result["success"],
                        "message": analysis_result["message"],
                        "stats": analysis_result.get("stats", {}),
                        "sql_query": analysis_result.get("sql_query", ""),
                    }
                ),
            }

            # Get final response from GPT incorporating the analysis results
            final_messages = messages + [message, tool_message]
            final_response = self.openai_client.chat.completions.create(
                model="gpt-4.1", messages=final_messages, temperature=0.1
            )

            final_content = final_response.choices[0].message.content

            # Format the response nicely for analysis results
            if analysis_result["success"] and analysis_result.get("stats"):
                formatted_response = f"""📊 **Analysis Results**

{final_content}

**SQL Query Used:**
```sql
{analysis_result.get('sql_query', 'N/A')}
```"""
            else:
                formatted_response = f"""❌ **Analysis Error**

{final_content}

**Error:** {analysis_result['message']}"""

            return {
                "success": analysis_result["success"],
                "response": formatted_response,
                "search_results": None,  # Analysis doesn't return map data
                "analysis_results": analysis_result.get("stats"),
                "is_analysis": True,
            }

        except json.JSONDecodeError:
            return {
                "success": False,
                "response": "❌ **Error:** Could not parse analysis parameters. Please try rephrasing your question.",
                "search_results": None,
            }

    def _handle_list_values(self, tool_call, messages, message) -> Dict[str, Any]:
        """Handle list_values function call."""
        try:
            args = json.loads(tool_call.function.arguments)
            list_query = args["query"]

            # Debug: Log what function calling extracted
            print(
                f"DEBUG - Function calling extracted for list values: query='{list_query}'"
            )

            # Execute the list values query
            list_result = self.property_service.list_values(list_query)

            # Create a follow-up message with the list results
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    {
                        "success": list_result["success"],
                        "message": list_result["message"],
                        "values": list_result.get("values", []),
                        "sql_query": list_result.get("sql_query", ""),
                    }
                ),
            }

            # Get final response from GPT incorporating the list results
            final_messages = messages + [message, tool_message]
            final_response = self.openai_client.chat.completions.create(
                model="gpt-4.1", messages=final_messages, temperature=0.1
            )

            final_content = final_response.choices[0].message.content

            # Format the response nicely for list results
            if list_result["success"] and list_result.get("values"):
                values_list = list_result["values"]
                # Format the values nicely
                if len(values_list) <= 20:
                    # If 20 or fewer values, show them all
                    values_display = ", ".join(str(v) for v in values_list)
                else:
                    # If more than 20, show first 20 and indicate there are more
                    values_display = (
                        ", ".join(str(v) for v in values_list[:20])
                        + f"... ({len(values_list)} total)"
                    )

                formatted_response = f"""📋 **Available Values**

{final_content}

**Found {len(values_list)} values:**
{values_display}

**SQL Query Used:**
```sql
{list_result.get('sql_query', 'N/A')}
```"""
            else:
                formatted_response = f"""❌ **List Values Error**

{final_content}

**Error:** {list_result['message']}"""

            return {
                "success": list_result["success"],
                "response": formatted_response,
                "search_results": None,  # List values doesn't return map data
                "list_results": list_result.get("values"),
                "is_list": True,
            }

        except json.JSONDecodeError:
            return {
                "success": False,
                "response": "❌ **Error:** Could not parse list values parameters. Please try rephrasing your question.",
                "search_results": None,
            }

    def _handle_find_specific_property(
        self, tool_call, messages, message
    ) -> Dict[str, Any]:
        """Handle find_specific_property function call."""
        try:
            args = json.loads(tool_call.function.arguments)
            lookup_value = args["lookup_value"]
            lookup_type = args.get("lookup_type", "auto")

            # Debug: Log what function calling extracted
            print(
                f"DEBUG - Function calling extracted for property lookup: value='{lookup_value}', type='{lookup_type}'"
            )

            # Execute the specific property lookup
            lookup_result = self.property_service.find_specific_property(
                lookup_value, lookup_type
            )

            # Create a follow-up message with the lookup results
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    {
                        "success": lookup_result["success"],
                        "message": lookup_result["message"],
                        "row_count": lookup_result["row_count"],
                        "sql_query": lookup_result.get("sql_query", ""),
                    }
                ),
            }

            # Get final response from GPT incorporating the lookup results
            final_messages = messages + [message, tool_message]
            final_response = self.openai_client.chat.completions.create(
                model="gpt-4.1", messages=final_messages, temperature=0.1
            )

            final_content = final_response.choices[0].message.content

            # Format the response nicely for property lookup
            if lookup_result["success"] and lookup_result["row_count"] > 0:
                detected_type = PropertyService.detect_lookup_type(lookup_value)
                formatted_response = f"""🎯 **Found {lookup_result['row_count']} propert{'y' if lookup_result['row_count'] == 1 else 'ies'}** (detected as {detected_type} lookup)

{final_content}

**SQL Query Used:**
```sql
{lookup_result.get('sql_query', 'N/A')}
```

🗺️ **The property is now highlighted on the map above!**"""
            elif lookup_result["success"] and lookup_result["row_count"] == 0:
                detected_type = PropertyService.detect_lookup_type(lookup_value)
                formatted_response = f"""ℹ️ **No property found** (searched as {detected_type})

{final_content}

**SQL Query Used:**
```sql
{lookup_result.get('sql_query', 'N/A')}
```

💡 Please check the format and try again:
- **APN**: Try format like '123-45-678'
- **Address**: Try '123 Main Street' or '123 Main St, San Jose'
- **Coordinates**: Try '37.4419, -122.1430'"""
            else:
                formatted_response = f"""❌ **Property Lookup Error**

{final_content}

**Error:** {lookup_result['message']}"""

            return {
                "success": lookup_result["success"],
                "response": formatted_response,
                "search_results": lookup_result["data"]
                if lookup_result["success"]
                else None,
                "row_count": lookup_result["row_count"],
                "is_specific_lookup": True,
            }

        except json.JSONDecodeError:
            return {
                "success": False,
                "response": "❌ **Error:** Could not parse property lookup parameters. Please try rephrasing your question.",
                "search_results": None,
            }
