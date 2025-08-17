import os
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    """Weather data structure"""
    location: str
    coordinates: Dict[str, float]
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    wind_direction: float
    precipitation: float
    visibility: float
    cloud_cover: float
    weather_condition: str
    timestamp: datetime
    forecast_data: List[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'location': self.location,
            'coordinates': self.coordinates,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'pressure': self.pressure,
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'precipitation': self.precipitation,
            'visibility': self.visibility,
            'cloud_cover': self.cloud_cover,
            'weather_condition': self.weather_condition,
            'timestamp': self.timestamp.isoformat(),
            'forecast_data': self.forecast_data or [],
        }

class WeatherService:
    """Service for fetching real-time weather data from OpenWeatherMap API (with robust fallbacks)"""

    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY', 'your-api-key-here')
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.session = None
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = 300  # 5 minutes

    async def initialize(self):
        """Deprecated: no-op (we now use per-call ClientSession to avoid cross-loop issues)."""
        return

    async def close(self):
        """Deprecated: no-op (per-call sessions are context-managed)."""
        return

    def _is_cache_valid(self, location: str) -> bool:
        """Check if cached data is still valid"""
        if location not in self.cache:
            return False

        cache_time = self.cache[location]['timestamp']
        return (datetime.utcnow() - cache_time).seconds < self.cache_duration

    async def get_current_weather(self, lat: float, lon: float, location_name: str = None, units: str = 'metric') -> Optional[WeatherData]:
        """Get current weather data for a location with OpenWeather first, then Open‑Meteo fallback."""
        try:
            await self.initialize()

            # Check cache first
            cache_key = f"{lat}_{lon}_{units}"
            if self._is_cache_valid(cache_key):
                logger.info(f"Using cached weather data for {location_name or cache_key}")
                return self.cache[cache_key]['data']

            weather_data: Optional[WeatherData] = None

            # Primary source: OpenWeather (if API key present)
            if self.api_key and self.api_key != 'your-api-key-here':
                url = f"{self.base_url}/weather"
                params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': self.api_key,
                    'units': units  # 'metric', 'imperial', or 'standard'
                }
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                weather_data = WeatherData(
                                    location=location_name or f"{lat}, {lon}",
                                    coordinates={"lat": lat, "lng": lon},
                                    temperature=float(data['main']['temp']),
                                    humidity=float(data['main']['humidity']),
                                    pressure=float(data['main']['pressure']),
                                    wind_speed=float(data['wind']['speed']),
                                    wind_direction=float(data['wind'].get('deg', 0)),
                                    precipitation=float(data['rain']['1h']) if 'rain' in data and '1h' in data['rain'] else 0.0,
                                    visibility=float(data.get('visibility', 10000)) / 1000.0,  # Convert to km
                                    cloud_cover=float(data['clouds']['all']),
                                    weather_condition=data['weather'][0]['main'],
                                    timestamp=datetime.utcnow()
                                )
                                
                                # Cache the data
                                self.cache[cache_key] = {
                                    'data': weather_data,
                                    'timestamp': datetime.utcnow()
                                }
                                
                                logger.info(f"Successfully fetched weather data from OpenWeather for {location_name or cache_key}")
                                return weather_data
                            else:
                                logger.warning(f"OpenWeather API returned status {response.status}")
                except Exception as e:
                    logger.warning(f"OpenWeather API call failed: {e}")

            # Fallback: Open‑Meteo (no API key required)
            logger.info("Falling back to Open‑Meteo API")
            weather_data = await self._current_from_open_meteo(lat, lon, location_name, units)
            
            if weather_data:
                # Cache the fallback data
                self.cache[cache_key] = {
                    'data': weather_data,
                    'timestamp': datetime.utcnow()
                }
                logger.info(f"Successfully fetched weather data from Open‑Meteo for {location_name or cache_key}")
                return weather_data
            
            logger.error(f"All weather APIs failed for coordinates {lat}, {lon}")
            return None

        except Exception as e:
            logger.error(f"Error in get_current_weather: {e}")
            return None

    async def geocode(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Geocode a location query to get coordinates. Tries multiple services."""
        try:
            await self.initialize()
            
            # Clean and normalize the query
            query = query.strip()
            if not query:
                return []
            
            # Try Open-Meteo first (no API key required)
            results = await self._geocode_open_meteo(query, limit)
            if results:
                # Filter out results with invalid coordinates
                valid_results = [r for r in results if r.get('lat') is not None and r.get('lon') is not None]
                if valid_results:
                    return valid_results
            
            # Fallback to OpenStreetMap Nominatim
            results = await self._geocode_osm(query, limit)
            if results:
                # Filter out results with invalid coordinates
                valid_results = [r for r in results if r.get('lat') is not None and r.get('lon') is not None]
                if valid_results:
                    return valid_results
            
            logger.warning(f"All geocoding services failed for query: {query}")
            return []
            
        except Exception as e:
            logger.error(f"Error in geocoding: {e}")
            return []

    async def _geocode_osm(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fallback to OpenStreetMap Nominatim geocoding. Respect usage policy: low rate and proper User-Agent."""
        try:
            await self.initialize()
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': query,
                'format': 'json',
                'addressdetails': 1,
                'limit': limit
            }
            headers = {
                'User-Agent': 'DisastroScope/1.0 (contact: support@disastroscope.local)'
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        results: List[Dict[str, Any]] = []
                        for item in data:
                            lat = item.get('lat')
                            lon = item.get('lon')
                            if lat is not None and lon is not None:
                                addr = item.get('address') or {}
                                results.append({
                                    'name': item.get('display_name') or addr.get('city') or addr.get('town') or addr.get('village') or query,
                                    'lat': float(lat),
                                    'lon': float(lon),
                                    'country': addr.get('country_code', '').upper() or addr.get('country'),
                                    'state': addr.get('state')
                                })
                        return results
                    else:
                        logger.error(f"OSM geocoding failed for '{query}' with status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"OSM geocoding error for '{query}': {e}")
            return []

    async def _geocode_open_meteo(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fallback to Open-Meteo geocoding API (no API key required)."""
        try:
            await self.initialize()
            url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {
                "name": query,
                "count": limit,
                "language": "en",
                "format": "json",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = data.get("results") or []
                        results: List[Dict[str, Any]] = []
                        for item in items:
                            lat = item.get("latitude")
                            lon = item.get("longitude")
                            if lat is not None and lon is not None:
                                results.append({
                                    "name": item.get("name") or query,
                                    "lat": float(lat),
                                    "lon": float(lon),
                                    "country": (item.get("country_code") or item.get("country")),
                                    "state": item.get("admin1")
                                })
                        return results
                    else:
                        logger.error(f"Open-Meteo geocoding failed for '{query}' with status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Open-Meteo geocoding error for '{query}': {e}")
            return []

    async def _current_from_open_meteo(self, lat: float, lon: float, location_name: Optional[str], units: str = "metric") -> Optional[WeatherData]:
        """Fallback current weather using Open-Meteo (no API key)."""
        try:
            await self.initialize()
            # Configure units
            temp_unit = "fahrenheit" if units == "imperial" else "celsius"
            wind_unit = "mph" if units == "imperial" else "ms"  # request m/s in metric
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,precipitation,visibility,cloud_cover",
                "temperature_unit": temp_unit,
                "windspeed_unit": wind_unit,
                "precipitation_unit": "mm",
                "timezone": "auto",
            }
            url = "https://api.open-meteo.com/v1/forecast"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Open-Meteo current weather failed: status={response.status}")
                        return None
                    data = await response.json()
                    cur = (data or {}).get("current") or {}
                    # Extract values with sensible defaults
                    temperature = float(cur.get("temperature_2m") or 0.0)
                    humidity = float(cur.get("relative_humidity_2m") or 0.0)
                    pressure = float(cur.get("pressure_msl") or 1013.0)
                    wind_speed = float(cur.get("wind_speed_10m") or 0.0)
                    wind_direction = float(cur.get("wind_direction_10m") or 0.0)
                    precipitation = float(cur.get("precipitation") or 0.0)
                    visibility_m = float(cur.get("visibility") or 10000.0)
                    cloud_cover = float(cur.get("cloud_cover") or 0.0)

                    # Convert "standard" units to Kelvin for temperature approximation
                    if units == "standard":
                        temperature = temperature + 273.15  # Kelvin approx from Celsius

                    # Derive a coarse weather condition
                    if precipitation > 0.1:
                        condition = "Rain"
                    else:
                        condition = "Clouds" if cloud_cover >= 75 else ("Partly Cloudy" if cloud_cover >= 25 else "Clear")

                    wd = WeatherData(
                        location=location_name or f"{lat}, {lon}",
                        coordinates={"lat": lat, "lng": lon},
                        temperature=temperature,
                        humidity=humidity,
                        pressure=pressure,
                        wind_speed=wind_speed,
                        wind_direction=wind_direction,
                        precipitation=precipitation,
                        visibility=max(0.0, visibility_m / 1000.0),  # km
                        cloud_cover=cloud_cover,
                        weather_condition=condition,
                        timestamp=datetime.utcnow(),
                    )
                    return wd
        except Exception as e:
            logger.error(f"Open-Meteo current weather error: {e}")
            return None

    async def _forecast_from_open_meteo(self, lat: float, lon: float, days: int = 3, units: str = "metric") -> List[Dict[str, Any]]:
        """Fallback forecast using Open-Meteo (no API key). Returns items shaped like OpenWeather list entries."""
        try:
            await self.initialize()
            temp_unit = "fahrenheit" if units == "imperial" else "celsius"
            wind_unit = "mph" if units == "imperial" else "ms"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,precipitation,cloud_cover,wind_speed_10m",
                "temperature_unit": temp_unit,
                "windspeed_unit": wind_unit,
                "precipitation_unit": "mm",
                "forecast_days": max(1, min(days, 5)),
                "timezone": "auto",
            }
            url = "https://api.open-meteo.com/v1/forecast"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Open-Meteo forecast failed: status={response.status}")
                        return []
                    data = await response.json()
                    hourly = (data or {}).get("hourly") or {}
                    times = hourly.get("time") or []
                    temps = hourly.get("temperature_2m") or []
                    precs = hourly.get("precipitation") or []
                    clouds = hourly.get("cloud_cover") or []
                    # Build a list with a familiar shape to the frontend
                    items: List[Dict[str, Any]] = []
                    for i in range(min(len(times), len(temps))):
                        t_iso = times[i]
                        t_val = float(temps[i] if temps[i] is not None else 0.0)
                        p_val = float(precs[i] if precs[i] is not None else 0.0)
                        c_val = float(clouds[i] if clouds[i] is not None else 0.0)
                        cond = "Rain" if p_val > 0.1 else ("Clouds" if c_val >= 75 else ("Partly Cloudy" if c_val >= 25 else "Clear"))
                        items.append({
                            "dt_txt": t_iso,
                            "main": {"temp": t_val},
                            "weather": [{"main": cond}],
                        })
                    return items
        except Exception as e:
            logger.error(f"Open-Meteo forecast error: {e}")
            return []

    async def get_weather_forecast(self, lat: float, lon: float, days: int = 5, units: str = 'metric') -> List[Dict[str, Any]]:
        """Get weather forecast for a location. Falls back to Open‑Meteo if OpenWeather fails or API key is missing."""
        try:
            await self.initialize()

            # Try OpenWeather first (if API key present)
            if self.api_key and self.api_key != 'your-api-key-here':
                url = f"{self.base_url}/forecast"
                params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': self.api_key,
                    'units': units,
                    'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
                }
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                return data.get('list', [])
                            else:
                                logger.error(f"Failed to fetch forecast via OpenWeather: {response.status}")
                except Exception as e:
                    logger.error(f"OpenWeather forecast error: {e}")
            else:
                logger.error("OPENWEATHER_API_KEY is missing. Using Open‑Meteo fallback for forecast.")

            # Fallback: Open‑Meteo (limit to 5 forecast days)
            return await self._forecast_from_open_meteo(lat, lon, days=min(days, 5), units=units)
        except Exception as e:
            logger.error(f"Error fetching forecast: {e}")
            return []

    async def get_multiple_locations_weather(self, locations: List[Dict[str, Any]]) -> List[WeatherData]:
        """Get weather data for multiple locations concurrently"""
        tasks = []
        for location in locations:
            coords = location.get('coords', {})
            lat = coords.get('lat')
            lon = coords.get('lng')
            if lat is not None and lon is not None:
                task = self.get_current_weather(
                    lat,
                    lon,
                    location.get('name')
                )
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        weather_data: List[WeatherData] = []

        for result in results:
            if isinstance(result, WeatherData):
                weather_data.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error in concurrent weather fetch: {result}")

        return weather_data

    def get_weather_risk_factors(self, weather_data: WeatherData) -> Dict[str, float]:
        """Calculate weather risk factors for disaster prediction"""
        risk_factors = {
            'flood_risk': 0.0,
            'wildfire_risk': 0.0,
            'storm_risk': 0.0,
            'heat_wave_risk': 0.0,
            'cold_wave_risk': 0.0
        }

        # Flood risk calculation
        if weather_data.precipitation > 10:  # Heavy rain
            risk_factors['flood_risk'] = min(1.0, weather_data.precipitation / 50)

        # Wildfire risk calculation
        if weather_data.temperature > 30 and weather_data.humidity < 30:
            risk_factors['wildfire_risk'] = min(1.0, (weather_data.temperature - 30) / 20)

        # Storm risk calculation
        if weather_data.wind_speed > 20:  # Strong winds
            risk_factors['storm_risk'] = min(1.0, weather_data.wind_speed / 50)

        # Heat wave risk
        if weather_data.temperature > 35:
            risk_factors['heat_wave_risk'] = min(1.0, (weather_data.temperature - 35) / 15)

        # Cold wave risk
        if weather_data.temperature < -10:
            risk_factors['cold_wave_risk'] = min(1.0, abs(weather_data.temperature + 10) / 20)

        return risk_factors

# Global weather service instance
weather_service = WeatherService()
