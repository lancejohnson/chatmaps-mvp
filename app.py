import streamlit as st
import folium
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(
    page_title="ChatMap MVP",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def create_santa_clara_map():
    """Create a Folium map centered on Santa Clara County"""
    # Santa Clara County approximate center coordinates
    santa_clara_center = [37.3382, -121.8863]

    # Create the map
    m = folium.Map(location=santa_clara_center, zoom_start=10, tiles="OpenStreetMap")

    # Add Santa Clara County approximate boundary
    # This is a rough boundary - in production you'd want the actual county shapefile
    santa_clara_bounds = [
        [37.1, -122.2],  # Southwest corner
        [37.1, -121.2],  # Southeast corner
        [37.6, -121.2],  # Northeast corner
        [37.6, -122.2],  # Northwest corner
        [37.1, -122.2],  # Close the polygon
    ]

    # Add county boundary
    folium.Polygon(
        locations=santa_clara_bounds,
        color="red",
        weight=2,
        fill=False,
        popup="Santa Clara County (Approximate Boundary)",
    ).add_to(m)

    return m


def main():
    st.title("🗺️ ChatMap MVP")
    st.markdown("*Explore Santa Clara County parcels through natural language queries*")

    # Create two columns for layout
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### Map View")

        # Create and display the map
        map_obj = create_santa_clara_map()

        # Display the map using streamlit-folium
        map_data = st_folium(
            map_obj, width=800, height=500, returned_objects=["last_object_clicked"]
        )

    with col2:
        st.markdown("### Map Info")
        st.info("📍 Santa Clara County, CA")
        st.markdown("""
        **Current View:**
        - County boundary shown in red
        - OpenStreetMap tiles
        - Ready for parcel data integration
        """)

        if map_data["last_object_clicked"]:
            st.markdown("**Last Clicked:**")
            st.json(map_data["last_object_clicked"])

    # Chat interface section
    st.markdown("---")
    st.markdown("### 💬 Chat with Your Data")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about parcels in Santa Clara County..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response (placeholder for now)
        with st.chat_message("assistant"):
            response = f"You asked: '{prompt}'\n\nI'm ready to help you explore parcel data! Currently, this is a placeholder response. Soon I'll be able to query the database and show results on the map above."
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
