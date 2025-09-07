# Railway Environment Variables Setup for Firebase & Tinybird

To enable Firebase and Tinybird integrations in your Railway backend, you need to set the following environment variables in your Railway dashboard.

## Required Environment Variables

### Core Backend Configuration
```
SECRET_KEY=your-secret-key-here
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO
```

### Firebase Configuration (Choose ONE option)

#### Option 1: Service Account Key as JSON String (Recommended)
```
FIREBASE_SERVICE_ACCOUNT_KEY={"type":"service_account","project_id":"your-project-id","private_key_id":"...","private_key":"-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n","client_email":"firebase-adminsdk-xxxxx@your-project-id.iam.gserviceaccount.com","client_id":"...","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-xxxxx%40your-project-id.iam.gserviceaccount.com"}
```

#### Option 2: Individual Configuration Values
```
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n
FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxxxx@your-project-id.iam.gserviceaccount.com
FIREBASE_PRIVATE_KEY_ID=your-private-key-id
FIREBASE_CLIENT_ID=your-client-id
```

### Tinybird Configuration
```
TINYBIRD_API_URL=https://cloud.tinybird.co/gcp/europe-west3/DisastroScope
TINYBIRD_TOKEN=your-tinybird-token
TINYBIRD_WORKSPACE_ID=DisastroScope
```

### Optional: External API Keys
```
OPENWEATHER_API_KEY=your-openweather-api-key
OPENCAGE_API_KEY=your-opencage-api-key
```

## How to Set Environment Variables in Railway

### Method 1: Railway Dashboard
1. Go to your Railway project dashboard
2. Select your DisastroScope backend service
3. Go to the "Variables" tab
4. Add each environment variable with its value
5. Click "Deploy" to apply changes

### Method 2: Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Link to your project
railway link

# Set environment variables
railway variables set SECRET_KEY=your-secret-key-here
railway variables set FIREBASE_PROJECT_ID=your-project-id
railway variables set FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
railway variables set FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxxxx@your-project-id.iam.gserviceaccount.com
railway variables set TINYBIRD_TOKEN=your-tinybird-token
railway variables set TINYBIRD_WORKSPACE_ID=your-workspace-id

# Deploy
railway up
```

## Getting Firebase Service Account Key

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project
3. Go to Project Settings → Service Accounts
4. Click "Generate new private key"
5. Download the JSON file
6. Copy the entire JSON content as the `FIREBASE_SERVICE_ACCOUNT_KEY` value

## Getting Tinybird Credentials

1. Go to [Tinybird Console](https://ui.tinybird.co/)
2. Go to your workspace
3. Navigate to Tokens section
4. Create a new token with appropriate permissions
5. Copy the token value
6. Get your workspace ID from the URL or workspace settings

## Verification

After setting the environment variables and deploying:

1. **Check Health Endpoint**: `https://web-production-47673.up.railway.app/health`
   - Should show integration status for Firebase and Tinybird

2. **Check Integration Status**: `https://web-production-47673.up.railway.app/api/integrations/status`
   - Should show detailed status of both services

3. **Test Firebase Auth**: `https://web-production-47673.up.railway.app/api/auth/verify`
   - Should accept Firebase ID tokens

4. **Test Tinybird Events**: `https://web-production-47673.up.railway.app/api/events/tinybird`
   - Should accept event data

## Troubleshooting

### Firebase Issues
- **"Firebase service not available"**: Check if `FIREBASE_SERVICE_ACCOUNT_KEY` is set correctly
- **"Invalid service account key"**: Ensure the JSON is properly formatted and escaped
- **"Project not found"**: Verify the project ID is correct

### Tinybird Issues
- **"Tinybird service not available"**: Check if `TINYBIRD_TOKEN` and `TINYBIRD_WORKSPACE_ID` are set
- **"API request failed"**: Verify the token has correct permissions
- **"Workspace not found"**: Check the workspace ID

### General Issues
- **Environment variables not updating**: Redeploy the service after setting variables
- **Import errors**: Check that all dependencies are in `requirements.txt`
- **Service unavailable**: Check Railway logs for detailed error messages

## Security Notes

- Never commit service account keys to version control
- Use Railway's secure environment variable storage
- Regularly rotate API keys and tokens
- Monitor usage and set up alerts for unusual activity
- Use least-privilege access for service accounts

## Support

If you encounter issues:
1. Check Railway deployment logs
2. Verify environment variables are set correctly
3. Test individual services using the health endpoints
4. Check Firebase and Tinybird console for any service issues
