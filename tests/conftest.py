"""
Pytest configuration for the chatmap-mvp project.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
env_file = project_root / ".env"
if not env_file.exists():
    raise FileNotFoundError(
        f"Required .env file not found at {env_file}. Please create this file with your DATABASE_URL and OPENAI_API_KEY."
    )

load_dotenv(env_file)

# Verify required environment variables are loaded
required_vars = ["DATABASE_URL", "OPENAI_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(
        f"Missing required environment variables: {missing_vars}. Please check your .env file."
    )

print("✅ Environment variables loaded successfully")
