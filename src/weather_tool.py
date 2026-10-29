
"""
AgriAssist-RX
Phase 4: Open-Meteo Weather Tool

Features:
- Converts a city name into coordinates
- Retrieves current weather information
- Displays temperature, humidity, wind, and time
- Handles location and connection errors

No Gemini API key is needed.
"""

import requests


# ==========================================
# STEP 1: API endpoints
# ==========================================

GEOCODING_URL = (
    "https://geocoding-api.open-meteo.com/v1/search"
)

WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


# ==========================================
# STEP 2: Find city coordinates
# ==========================================

def find_location(city, country_code=None):

    parameters = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    if country_code:
        parameters["countryCode"] = country_code

    response = requests.get(
        GEOCODING_URL,
        params=parameters,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    locations = data.get("results", [])

    if not locations:
        raise ValueError(
            f"Location not found: {city}"
        )

    location = locations[0]

    return {
        "name": location["name"],
        "country": location.get("country", ""),
        "region": location.get("admin1", ""),
        "latitude": location["latitude"],
        "longitude": location["longitude"]
    }


# ==========================================
# STEP 3: Retrieve current weather
# ==========================================

def get_current_weather(city, country_code=None):

    location = find_location(
        city,
        country_code=country_code
    )

    parameters = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "precipitation"
        ),
        "timezone": "auto"
    }

    response = requests.get(
        WEATHER_URL,
        params=parameters,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    current = data.get("current", {})

    if not current or "temperature_2m" not in current:
        raise ValueError(
            "Weather information is unavailable."
        )

    return {
        "city": location["name"],
        "region": location["region"],
        "country": location["country"],
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "time": current.get("time"),
        "temperature_c": current.get("temperature_2m"),
        "humidity_percent": current.get(
            "relative_humidity_2m"
        ),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "precipitation_mm": current.get("precipitation"),
        "source": "Open-Meteo",
        "source_url": WEATHER_URL
    }


# ==========================================
# STEP 4: Format the weather response
# ==========================================

def format_weather(weather):

    return (
        f"Location: {weather['city']}, "
        f"{weather['region']}, {weather['country']}\n"
        f"Weather time (local): {weather['time']}\n"
        f"Temperature: {weather['temperature_c']} °C\n"
        f"Relative humidity: "
        f"{weather['humidity_percent']}%\n"
        f"Wind speed: {weather['wind_speed_kmh']} km/h\n"
        f"Current precipitation: "
        f"{weather['precipitation_mm']} mm\n"
        f"Source: {weather['source']}"
    )


# ==========================================
# STEP 5: Interactive test
# ==========================================

def main():

    print("=" * 55)
    print("AGRIASSIST-RX: LIVE WEATHER TOOL")
    print("=" * 55)

    print("\nEnter a city name to check current weather.")
    print("Type 'exit' to stop.\n")

    while True:

        city = input("Enter city: ").strip()

        if city.lower() in ("exit", "quit"):
            print("\nGoodbye!")
            break

        if not city:
            continue

        try:

            # During this first test, use cities
            # in Pakistan to avoid ambiguous names.
            weather = get_current_weather(
                city,
                country_code="PK"
            )

            print("\n" + "-" * 55)
            print(format_weather(weather))
            print("-" * 55 + "\n")

        except requests.RequestException as error:

            print(
                "\nUnable to contact Open-Meteo:",
                error
            )

        except ValueError as error:

            print("\nError:", error)


if __name__ == "__main__":
    main()
