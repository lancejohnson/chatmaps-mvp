import streamlit as st
from streamlit_folium import st_folium
from dotenv import load_dotenv
import os

from src.services.chat_service import ChatService
from src.services.map_service import MapService
from src.ui.components import (
    display_property_list,
    show_schema_info,
    show_example_questions,
    show_clear_map_button,
    display_chat_interface_header,
)

# Load environment variables
load_dotenv()

# Check for required environment variables early
required_env_vars = ["DATABASE_URL", "OPENAI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    st.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
    st.error("Please check your environment configuration.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="ChatMaps MVP",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def get_chat_service():
    """Initialize and cache the chat service."""
    try:
        return ChatService()
    except Exception as e:
        st.error(f"❌ Failed to initialize chat service: {str(e)}")
        return None


@st.cache_resource
def get_map_service():
    """Initialize and cache the map service."""
    try:
        return MapService()
    except Exception as e:
        st.error(f"❌ Failed to initialize map service: {str(e)}")
        return None


def main():
    st.title("🗺️ ChatMaps MVP")

    # Show schema info in sidebar for development
    show_schema_info()

    # Initialize session state for query results
    if "query_results" not in st.session_state:
        st.session_state.query_results = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize session state for selected property
    if "selected_property" not in st.session_state:
        st.session_state.selected_property = None

    # Get services
    chat_service = get_chat_service()
    map_service = get_map_service()
    
    # Check if services initialized successfully
    if not chat_service or not map_service:
        st.error("❌ Application failed to initialize properly. Please check the deployment logs.")
        st.stop()

    # Create columns layout - map on left, property list on right
    col_map, col_properties = st.columns([2, 1])

    with col_map:
        # Create and display the map (with current query results if any)
        map_obj = map_service.create_santa_clara_map(
            st.session_state.query_results, st.session_state.selected_property
        )
        # Display the map using streamlit-folium
        st_folium(map_obj, width=800, height=500)

    with col_properties:
        # Display property list if we have query results
        if st.session_state.query_results and st.session_state.query_results.get(
            "features"
        ):
            display_property_list(st.session_state.query_results["features"])

    # Chat interface section
    display_chat_interface_header()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input with enhanced processing
    if prompt := st.chat_input("Ask about properties or just chat..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the user query with function calling
        with st.chat_message("assistant"):
            # Show loading spinner while processing
            with st.spinner("🤖 Thinking about your request..."):
                # Process through the chat service
                result = chat_service.process_chat_with_function_calling(
                    prompt, st.session_state.messages[:-1]
                )

            # Display the result
            st.markdown(result["response"])

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": result["response"]}
        )

        # Update map with new query results if we have search results
        if result.get("search_results"):
            st.session_state.query_results = result["search_results"]
            # Trigger rerun to update the map
            st.rerun()
        elif result.get("row_count", 0) == 0 and not result.get("is_chat"):
            # Clear previous results if search returned no matches
            st.session_state.query_results = None
        # If it's just chat (is_chat=True), don't clear the map

    # Add helpful examples
    show_example_questions()

    # Add clear results button
    show_clear_map_button(st.session_state.query_results)


if __name__ == "__main__":
    main()
