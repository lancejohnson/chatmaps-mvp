import streamlit as st
import os

print("🔄 Simple app starting...")

st.title("Hello Railway!")
st.write(f"Running on port: {os.getenv('PORT', 'NOT SET')}")
st.write("If you can see this, Streamlit is working!")

print("✅ Simple app loaded successfully")
