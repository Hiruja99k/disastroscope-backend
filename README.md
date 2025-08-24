# DisastroScope Backend API

A robust Flask-based backend API for disaster monitoring and prediction, designed to work seamlessly with the DisastroScope frontend.

## 🚀 Features

- **Real-time Disaster Data**: Sample disaster events and predictions
- **AI Prediction Engine**: Heuristic-based risk assessment for multiple disaster types
- **Weather Integration**: Mock weather data for cities
- **Location-based Queries**: Find disasters and predictions near specific coordinates
- **Model Management**: Track available AI models and their status
- **Health Monitoring**: Comprehensive health check endpoints
- **CORS Support**: Cross-origin resource sharing enabled
- **Error Handling**: Robust error handling with proper HTTP status codes

## 🛠️ Technology Stack

- **Framework**: Flask 2.3.3
- **WSGI Server**: Gunicorn 21.2.0
- **CORS**: Flask-CORS 4.0.0
- **Environment**: python-dotenv 1.0.0
- **HTTP Client**: requests 2.31.0
- **Async Support**: eventlet 0.33.3 (for future WebSocket support)

## 📋 API Endpoints

### Health & Status
- `GET /health` - Basic health check
- `GET /api/health` - API health check
- `GET /` - API information and endpoint list

### Data Retrieval
- `GET /api/events` - Get all disaster events
- `GET /api/predictions` - Get all predictions
- `GET /api/models` - List available AI models

### AI & Predictions
- `POST /api/ai/predict` - Generate disaster risk predictions

### Weather
- `GET /api/weather/<city>` - Get weather data for a city

### Location-based
- `POST /api/events/near` - Find events near coordinates
- `POST /api/predictions/near` - Find predictions near coordinates

## 🏃‍♂️ Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd temp-backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run the development server**
   ```bash
   python app.py
   ```

5. **Test the API**
   ```bash
   python test_backend.py
   ```

### Production Deployment (Railway)

1. **Deploy to Railway**
   - Connect your GitHub repository to Railway
   - Railway will automatically detect the Python app
   - The `Procfile` and `start.sh` will handle the deployment

2. **Environment Variables**
   - Set `PORT` (Railway provides this automatically)
   - Set `SECRET_KEY` for Flask security
   - Set `RAILWAY_ENVIRONMENT=production`

## 🔧 Configuration

### Environment Variables

```bash
# Required
PORT=5000                    # Port for the application
SECRET_KEY=your-secret-key   # Flask secret key

# Optional
RAILWAY_ENVIRONMENT=production  # Set to 'production' for Railway
DEBUG=False                     # Set to True for development
```

### Gunicorn Configuration

The application uses a custom `gunicorn.conf.py` with:
- **Worker Class**: `sync` (synchronous workers)
- **Workers**: 1 (single worker for Railway)
- **Timeout**: 120 seconds
- **Max Requests**: 1000 per worker

## 📊 Sample Data

The backend includes sample disaster events and predictions:

### Disaster Events
- Hurricane Maria (Puerto Rico)
- California Wildfire
- Mississippi Flood
- Texas Tornado
- California Earthquake

### Predictions
- Flood risk in Houston
- Wildfire risk in Los Angeles
- Storm risk in Miami
- Landslide risk in Seattle

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_backend.py
```

This will test all endpoints and verify the API is working correctly.

## 🔍 API Examples

### Get All Events
```bash
curl http://localhost:5000/api/events
```

### Get AI Predictions
```bash
curl -X POST http://localhost:5000/api/ai/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "temperature": 25.0,
    "humidity": 70.0,
    "pressure": 1013.0,
    "wind_speed": 10.0,
    "precipitation": 5.0
  }'
```

### Find Events Near Location
```bash
curl -X POST http://localhost:5000/api/events/near \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "radius": 100
  }'
```

## 🚨 Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid input data
- **404 Not Found**: Endpoint not found
- **405 Method Not Allowed**: Wrong HTTP method
- **500 Internal Server Error**: Server-side errors

All errors return JSON responses with descriptive messages.

## 🔒 Security

- CORS enabled for frontend integration
- Input validation on all endpoints
- Proper error handling without exposing internals
- Environment-based configuration

## 📈 Monitoring

- Health check endpoints for uptime monitoring
- Structured logging with timestamps
- Request/response logging via Gunicorn

## 🚀 Deployment

### Railway (Recommended)

1. Push to GitHub
2. Connect repository to Railway
3. Railway auto-deploys on push
4. Environment variables are set in Railway dashboard

### Other Platforms

The application is compatible with:
- Heroku
- DigitalOcean App Platform
- AWS Elastic Beanstalk
- Google Cloud Run

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the logs in Railway dashboard
2. Run the test suite locally
3. Verify environment variables
4. Check endpoint documentation above

---

**DisastroScope Backend** - Empowering disaster monitoring and prediction through intelligent APIs.
