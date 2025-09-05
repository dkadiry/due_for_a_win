import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("THE_ODDS_API_KEY")
print(f"The API key is loaded: {API_KEY}")