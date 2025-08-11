#!/usr/bin/env python3

import os
import sys

print("=== Environment Debug Info ===")
print(f"Python version: {sys.version}")
print(f"All environment variables related to PORT:")

# Check all environment variables for any PORT-related ones
for key, value in sorted(os.environ.items()):
    if 'PORT' in key.upper():
        print(f"  {key} = {value}")

print(f"\nChecking specific variables:")
print(f"  PORT = {os.getenv('PORT', 'NOT SET')}")
print(f"  RAILWAY_PUBLIC_DOMAIN = {os.getenv('RAILWAY_PUBLIC_DOMAIN', 'NOT SET')}")
print(f"  RAILWAY_STATIC_URL = {os.getenv('RAILWAY_STATIC_URL', 'NOT SET')}")

print(f"\nAll RAILWAY_ variables:")
for key, value in sorted(os.environ.items()):
    if key.startswith('RAILWAY_'):
        print(f"  {key} = {value}")

print("=== End Debug Info ===")
