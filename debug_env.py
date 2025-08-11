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

print("=== End Debug Info ===")
