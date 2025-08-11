#!/usr/bin/env python3

import os
import sys

print("=== Environment Debug Info ===")
print(f"Python version: {sys.version}")
print("All environment variables related to PORT:")

# Check all environment variables for any PORT-related ones
for key, value in sorted(os.environ.items()):
    if "PORT" in key.upper():
        print(f"  {key} = {value}")

print("\nChecking specific variables:")
print(f"  PORT = {os.getenv('PORT', 'NOT SET')}")
print(f"  OPENAI_API_KEY = {os.getenv('OPENAI_API_KEY', 'NOT SET')}")
print(f"  DATABASE_URL = {os.getenv('DATABASE_URL', 'NOT SET')}")
print(f"  RAILWAY_PUBLIC_DOMAIN = {os.getenv('RAILWAY_PUBLIC_DOMAIN', 'NOT SET')}")
print(f"  RAILWAY_STATIC_URL = {os.getenv('RAILWAY_STATIC_URL', 'NOT SET')}")

print("\nAll RAILWAY_ variables:")
for key, value in sorted(os.environ.items()):
    if key.startswith("RAILWAY_"):
        print(f"  {key} = {value}")

print("\nTesting database connection:")
try:
    from sqlalchemy import create_engine, text

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        print(f"  DATABASE_URL found: {database_url[:50]}...")

        # Fix postgres:// to postgresql:// if needed (Railway compatibility)
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
            print("  🔧 Fixed postgres:// to postgresql://")

        engine = create_engine(database_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("  ✅ Database connection successful")
    else:
        print("  ❌ DATABASE_URL not found")
except Exception as e:
    print(f"  ❌ Database connection failed: {e}")

print("\nTesting boundary file access:")
try:
    from pathlib import Path

    # Use same path logic as MapService
    script_dir = Path(__file__).parent  # debug_env.py is in root
    boundary_file = script_dir / "data" / "boundaries" / "santa_clara_county.geojson"

    print(f"  Looking for boundary file at: {boundary_file}")

    if boundary_file.exists():
        file_size = boundary_file.stat().st_size
        print(f"  ✅ Boundary file found ({file_size} bytes)")
    else:
        print("  ❌ Boundary file not found")

        # Check if data directory exists
        data_dir = script_dir / "data"
        if data_dir.exists():
            print("  📁 Data directory exists")
            boundaries_dir = data_dir / "boundaries"
            if boundaries_dir.exists():
                print("  📁 Boundaries directory exists")
                # List files in boundaries directory
                files = list(boundaries_dir.iterdir())
                print(f"  📋 Files in boundaries: {[f.name for f in files]}")
            else:
                print("  ❌ Boundaries directory missing")
        else:
            print("  ❌ Data directory missing")

except Exception as e:
    print(f"  ❌ Boundary file check failed: {e}")

print("\nTesting OpenAI API connection:")
try:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"  OPENAI_API_KEY found: {api_key[:20]}...")
        client = OpenAI(api_key=api_key)
        # Don't actually call the API to avoid costs, just test client creation
        print("  ✅ OpenAI client created successfully")
    else:
        print("  ❌ OPENAI_API_KEY not found")
except Exception as e:
    print(f"  ❌ OpenAI client creation failed: {e}")

print("\nTesting service imports:")
try:
    from src.services.chat_service import ChatService

    print("  ✅ ChatService import successful")

    from src.services.map_service import MapService

    print("  ✅ MapService import successful")

    from src.services.property_service import PropertyService

    print("  ✅ PropertyService import successful")

    # Test basic service initialization (without full functionality)
    print("  🔄 Testing basic service creation...")

except Exception as e:
    print(f"  ❌ Service import failed: {e}")

print("\nTesting Streamlit imports:")
try:
    import streamlit as st

    print("  ✅ Streamlit import successful")

    from streamlit_folium import st_folium

    print("  ✅ Streamlit-folium import successful")

    import folium

    print("  ✅ Folium import successful")

except Exception as e:
    print(f"  ❌ Streamlit import failed: {e}")

print("\nTesting port binding:")
try:
    import socket

    port = int(os.getenv("PORT", "8080"))
    print(f"  Checking if port {port} is available...")

    # Test if we can bind to the port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    result = sock.bind_ex(("0.0.0.0", port))
    sock.close()

    if result == 0:
        print(f"  ✅ Port {port} is available for binding")
    else:
        print(f"  ❌ Port {port} is already in use or unavailable")

except Exception as e:
    print(f"  ❌ Port binding test failed: {e}")

print("=== End Debug Info ===")
print("🚀 About to start Streamlit server...")

# Add a background health check test
import subprocess
import threading
import time


def test_health_check():
    """Test if we can connect to the Streamlit server after it starts"""
    time.sleep(15)  # Wait for Streamlit to start
    try:
        import urllib.request
        import urllib.error

        port = os.getenv("PORT", "8080")
        url = f"http://localhost:{port}/"
        print(f"🔍 Testing connection to {url}")

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.getcode()
            print(f"🌐 Health check SUCCESS: Status {status}")

    except urllib.error.URLError as e:
        print(f"❌ Health check FAILED: {e}")
    except Exception as e:
        print(f"❌ Health check ERROR: {e}")


# Start health check in background
threading.Thread(target=test_health_check, daemon=True).start()


def test_railway_routing():
    """Test Railway's routing after more time"""
    time.sleep(30)  # Wait longer for Railway routing
    try:
        import urllib.request
        import urllib.error

        # Test Railway's public domain
        public_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "")
        if public_domain:
            url = f"https://{public_domain}/"
            print(f"🔍 Testing Railway public domain: {url}")

            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as response:
                status = response.getcode()
                print(f"🌐 Railway routing SUCCESS: Status {status}")
        else:
            print("❌ No Railway public domain found")

    except urllib.error.URLError as e:
        print(f"❌ Railway routing FAILED: {e}")
    except Exception as e:
        print(f"❌ Railway routing ERROR: {e}")


# Start Railway routing test
threading.Thread(target=test_railway_routing, daemon=True).start()
