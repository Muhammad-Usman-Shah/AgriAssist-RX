
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# STEP 1: Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent

# STEP 2: Load the API key from .env
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "Gemini API key not found. Check your .env file."
    )

print("API key loaded successfully.")
print("Connecting to Google Gemini...")


# STEP 3: Connect to Gemini
client = genai.Client(api_key=api_key)

try:
    # STEP 4: Send our first agricultural question
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=(
            "What is wheat rust disease? "
            "Answer in one simple sentence."
        )
    )

    # STEP 5: Display the response
    print("\n" + "=" * 50)
    print("AGR IASSIST-RX: GEMINI TEST")
    print("=" * 50)

    if response.text:
        print("\nQuestion: What is wheat rust disease?")
        print("\nGemini Answer:")
        print(response.text)
        print("\nConnection test completed successfully!")
    else:
        print("Gemini returned no text.")

except Exception as error:
    print("\nGemini API request failed.")
    print("Error:", error)

finally:
    client.close()
