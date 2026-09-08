from fastapi import FastAPI
from pydantic import BaseModel
from strands import tool, Agent
from strands.models.openai import OpenAIModel
from datetime import datetime
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import pytz
import requests
import os

model = OpenAIModel(
    client_args={
        "api_key": "xxxxxxxxxxxx",
        "base_url": os.getenv("MODEL_ENDPOINT")
    },
    model_id=os.getenv("MODEL_ID"),
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    }
)

app = FastAPI(title="Time and Weather Agent")

geolocator = Nominatim(user_agent="time-weather-agent", timeout=10)
tf = TimezoneFinder()

@tool
def current_time(location: str) -> str:
    """Get the current time for a location.
    
    Args:
        location: The city or location name to get the time for
    """
    try:
        location_data = geolocator.geocode(location)
        if location_data:
            timezone = tf.timezone_at(lng=location_data.longitude, lat=location_data.latitude) or "UTC"
            dt = datetime.now(pytz.timezone(timezone))
            return dt.strftime("%Y-%m-%d %I:%M %p %Z")
    except Exception as e:
        print(f"Time error: {e}")
    return datetime.now().strftime("%Y-%m-%d %I:%M %p")

@tool
def current_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city or location name to get the weather for
    """
    try:
        location_data = geolocator.geocode(location)
        if not location_data:
            return "Location not found"

        lat, lon = location_data.latitude, location_data.longitude
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code&timezone=auto"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            temp = data.get("current", {}).get("temperature_2m", "N/A")
            code = data.get("current", {}).get("weather_code", "N/A")
            
            weather_codes = {
                0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle",
                53: "Moderate drizzle", 55: "Dense drizzle", 61: "Light rain",
                63: "Moderate rain", 65: "Heavy rain", 71: "Light snow",
                73: "Moderate snow", 75: "Heavy snow", 95: "Thunderstorm"
            }
            weather_desc = weather_codes.get(code, f"Unknown (code {code})")
            
            return f"{weather_desc}, {temp}°C"
        
        return "Weather data currently unavailable"
    except Exception as e:
        print(f"Weather error: {e}")
        return "Weather service is currently unavailable"

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/agent")
async def agent_endpoint(request: QueryRequest):
    """Endpoint that uses time and weather agent"""
    agent = Agent(model=model, tools=[current_time, current_weather])
    response = agent(request.query)
    
    return {
    "status": "success",
    "response": response
    }