import os
from dotenv import load_dotenv
from google import genai

# Load variables from .env
load_dotenv()

# Read Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Send a simple test prompt
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Explain diabetes in one simple sentence."
)

print("\nGemini response:")
print(response.text)