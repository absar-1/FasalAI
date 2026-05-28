"""
Fasal-AI — Entry Point
Loads environment variables and validates all required API keys.
"""
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

REQUIRED_KEYS = [
    "OPENAI_API_KEY",
    "PLANT_ID_API_KEY",
]

missing_keys = [key for key in REQUIRED_KEYS if not os.getenv(key)]

if not missing_keys:
    print("Setup complete. All keys loaded.")
else:
    print(f"ERROR: Missing API keys: {', '.join(missing_keys)}")
