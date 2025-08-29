# Geocoding Setup Guide

## Real Location Data Integration

The backend now uses **OpenCage Geocoding API** to provide real location data instead of "Sample Region, Sample Country".

### Getting a Free API Key

1. **Visit OpenCage Data**: Go to https://opencagedata.com/
2. **Sign Up**: Create a free account
3. **Get API Key**: Navigate to your dashboard and copy your API key
4. **Free Tier**: 2,500 requests per day (more than enough for testing)

### Setting Up the API Key

#### Option 1: Environment Variable (Recommended)
Add to your Railway environment variables:
```
OPENCAGE_API_KEY=your_api_key_here
```

#### Option 2: Local Development
Create a `.env` file in the backend directory:
```
OPENCAGE_API_KEY=your_api_key_here
```

### Current Implementation

The backend now:
- ✅ **Real Geocoding**: Converts location names to actual coordinates
- ✅ **Reverse Geocoding**: Converts coordinates to real location names
- ✅ **Realistic Risk Analysis**: Based on actual geographical factors
- ✅ **Fallback System**: Works even without API key (uses demo mode)

### Geographical Risk Factors

The risk analysis now considers real geographical factors:

1. **Floods**: Coastal proximity, elevation, rainfall patterns
2. **Landslides**: Mountainous regions, elevation, geological factors
3. **Earthquakes**: Tectonic plate boundaries (Pacific Ring of Fire, etc.)
4. **Cyclones**: Tropical regions, coastal areas, seasonal patterns
5. **Wildfires**: Climate zones, vegetation, seasonal dryness
6. **Tsunamis**: Subduction zones, coastal proximity
7. **Droughts**: Arid regions, continental interiors, seasonal patterns

### Testing

Try searching for real locations like:
- "Tokyo, Japan"
- "New York, USA"
- "London, UK"
- "Sydney, Australia"
- "Mumbai, India"

The system will now show real location names and coordinates instead of "Sample Region, Sample Country".
