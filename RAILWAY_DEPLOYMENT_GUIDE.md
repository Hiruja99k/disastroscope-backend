# Railway Deployment Guide for DisastroScope with Firebase & Tinybird

This guide will help you deploy your DisastroScope backend to Railway with Firebase and Tinybird integrations.

## Prerequisites

- Railway account (free tier available)
- Firebase project set up
- Tinybird workspace created
- GitHub repository with your code

## Step 1: Deploy to Railway

### Method 1: Connect GitHub Repository

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your `disastroscope-backend` repository
5. Railway will automatically detect it's a Python app

### Method 2: Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy
railway up
```

## Step 2: Set Environment Variables

### Core Backend Configuration
```
SECRET_KEY=disastroscope-secret-key-2024-production
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO
```

### Tinybird Configuration (Your Custom URL)
```
TINYBIRD_API_URL=https://cloud.tinybird.co/gcp/europe-west3/DisastroScope
TINYBIRD_TOKEN=your-tinybird-token-here
TINYBIRD_WORKSPACE_ID=DisastroScope
```

### Firebase Configuration (Choose ONE option)

#### Option A: Complete JSON (Recommended)
```
FIREBASE_SERVICE_ACCOUNT_KEY={"type":"service_account","project_id":"your-project-id","private_key_id":"...","private_key":"-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n","client_email":"firebase-adminsdk-xxxxx@your-project-id.iam.gserviceaccount.com","client_id":"...","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-xxxxx%40your-project-id.iam.gserviceaccount.com"}
```

#### Option B: Individual Values
```
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n
FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxxxx@your-project-id.iam.gserviceaccount.com
FIREBASE_PRIVATE_KEY_ID=your-private-key-id
FIREBASE_CLIENT_ID=your-client-id
```

### Optional: External API Keys
```
OPENWEATHER_API_KEY=your-openweather-api-key
OPENCAGE_API_KEY=your-opencage-api-key
```

## Step 3: How to Set Environment Variables in Railway

### Using Railway Dashboard

1. Go to your Railway project dashboard
2. Click on your DisastroScope backend service
3. Go to the **"Variables"** tab
4. Click **"New Variable"**
5. Add each environment variable:
   - **Name**: `TINYBIRD_API_URL`
   - **Value**: `https://cloud.tinybird.co/gcp/europe-west3/DisastroScope`
6. Click **"Add"**
7. Repeat for all variables
8. Click **"Deploy"** to apply changes

### Using Railway CLI

```bash
# Set Tinybird configuration
railway variables set TINYBIRD_API_URL=https://cloud.tinybird.co/gcp/europe-west3/DisastroScope
railway variables set TINYBIRD_TOKEN=your-tinybird-token
railway variables set TINYBIRD_WORKSPACE_ID=DisastroScope

# Set Firebase configuration (Option A - JSON)
railway variables set FIREBASE_SERVICE_ACCOUNT_KEY='{"type":"service_account",...}'

# OR Set Firebase configuration (Option B - Individual)
railway variables set FIREBASE_PROJECT_ID=your-project-id
railway variables set FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
railway variables set FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxxxx@your-project-id.iam.gserviceaccount.com

# Set core configuration
railway variables set SECRET_KEY=disastroscope-secret-key-2024-production
railway variables set ENVIRONMENT=production
railway variables set DEBUG=False
railway variables set LOG_LEVEL=INFO

# Deploy
railway up
```

## Step 4: Getting Your Credentials

### Getting Tinybird Token

1. Go to [Tinybird Console](https://ui.tinybird.co/)
2. Navigate to your **DisastroScope** workspace
3. Go to **"Tokens"** section
4. Click **"Create Token"**
5. Set permissions:
   - **Read**: Events, Pipes
   - **Write**: Events
6. Copy the generated token
7. Use it as `TINYBIRD_TOKEN`

### Getting Firebase Service Account Key

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your DisastroScope project
3. Click **Settings** (⚙️) → **Project settings**
4. Go to **"Service accounts"** tab
5. Click **"Generate new private key"**
6. Download the JSON file
7. Copy the entire JSON content for `FIREBASE_SERVICE_ACCOUNT_KEY`

## Step 5: Verify Deployment

### Check Health Endpoint
```bash
curl https://your-railway-url.up.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "integrations": {
    "firebase": {
      "available": true,
      "status": "initialized"
    },
    "tinybird": {
      "available": true,
      "status": "initialized"
    }
  }
}
```

### Check Integration Status
```bash
curl https://your-railway-url.up.railway.app/api/integrations/status
```

### Test Firebase Authentication
```bash
curl -X POST https://your-railway-url.up.railway.app/api/auth/verify \
  -H "Content-Type: application/json" \
  -d '{"token": "your-firebase-id-token"}'
```

### Test Tinybird Events
```bash
curl -X POST https://your-railway-url.up.railway.app/api/events/tinybird \
  -H "Content-Type: application/json" \
  -d '{
    "type": "disaster_event",
    "id": "test-001",
    "type": "flood",
    "severity": 3,
    "latitude": 40.7128,
    "longitude": -74.0060,
    "description": "Test disaster event"
  }'
```

## Step 6: Update Frontend Configuration

Update your frontend to use the Railway backend URL:

```typescript
// In your frontend environment configuration
VITE_API_BASE_URL=https://your-railway-url.up.railway.app
VITE_SOCKET_URL=https://your-railway-url.up.railway.app
```

## Troubleshooting

### Common Issues

#### 1. "Firebase service not available"
- Check if `FIREBASE_SERVICE_ACCOUNT_KEY` is set correctly
- Verify the JSON is properly formatted
- Ensure quotes are escaped correctly

#### 2. "Tinybird service not available"
- Check if `TINYBIRD_TOKEN` and `TINYBIRD_WORKSPACE_ID` are set
- Verify the token has correct permissions
- Ensure the workspace ID matches your Tinybird workspace

#### 3. "Import errors"
- Check Railway deployment logs
- Verify all dependencies are in `requirements.txt`
- Ensure Python version is compatible

#### 4. "Environment variables not updating"
- Redeploy the service after setting variables
- Check Railway logs for any errors
- Verify variable names are correct

### Checking Railway Logs

1. Go to Railway dashboard
2. Select your service
3. Click on **"Deployments"** tab
4. Click on the latest deployment
5. Check the logs for any errors

### Testing Individual Services

```bash
# Test basic health
curl https://your-railway-url.up.railway.app/health

# Test Firebase integration
curl https://your-railway-url.up.railway.app/api/integrations/status

# Test Tinybird integration
curl https://your-railway-url.up.railway.app/api/events/tinybird
```

## Security Considerations

1. **Never commit sensitive keys to version control**
2. **Use Railway's secure environment variable storage**
3. **Regularly rotate API keys and tokens**
4. **Monitor usage and set up alerts**
5. **Use least-privilege access for service accounts**

## Performance Optimization

1. **Enable Railway's auto-scaling**
2. **Set appropriate resource limits**
3. **Monitor memory and CPU usage**
4. **Use Railway's built-in monitoring**

## Cost Management

- Railway free tier includes 500 hours/month
- Monitor usage in Railway dashboard
- Set up billing alerts if needed
- Consider upgrading for production use

## Support

If you encounter issues:
1. Check Railway deployment logs
2. Verify environment variables
3. Test individual endpoints
4. Check Firebase and Tinybird console
5. Review this guide for common solutions

## Next Steps

After successful deployment:
1. Test all endpoints
2. Set up monitoring and alerts
3. Configure custom domain (optional)
4. Set up CI/CD pipeline
5. Monitor performance and costs
