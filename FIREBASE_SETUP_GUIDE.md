# Firebase Service Account Setup Guide

This guide will walk you through getting your Firebase service account credentials for the DisastroScope backend.

## Step 1: Access Firebase Console

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Sign in with your Google account
3. Select your DisastroScope project (or create one if you haven't already)

## Step 2: Enable Authentication

1. In the Firebase Console, click on **"Authentication"** in the left sidebar
2. Click on **"Get started"** if you haven't set it up yet
3. Go to the **"Sign-in method"** tab
4. Enable the authentication methods you want:
   - **Email/Password** (recommended)
   - **Google** (optional)
   - **Phone** (optional)

## Step 3: Get Service Account Key

### Method 1: Generate New Service Account Key (Recommended)

1. In the Firebase Console, click on the **gear icon** (⚙️) next to "Project Overview"
2. Select **"Project settings"**
3. Go to the **"Service accounts"** tab
4. Click **"Generate new private key"**
5. A dialog will appear - click **"Generate key"**
6. A JSON file will be downloaded to your computer

### Method 2: Use Existing Service Account

If you already have a service account:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your Firebase project
3. Go to **"IAM & Admin"** → **"Service Accounts"**
4. Find your Firebase Admin SDK service account
5. Click on it and go to **"Keys"** tab
6. Click **"Add Key"** → **"Create new key"** → **"JSON"**

## Step 4: Extract Required Values

Open the downloaded JSON file. It will look like this:

```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "abc123...",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-xxxxx@your-project-id.iam.gserviceaccount.com",
  "client_id": "123456789...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-xxxxx%40your-project-id.iam.gserviceaccount.com"
}
```

## Step 5: Set Railway Environment Variables

### Option A: Use Complete JSON (Easiest)

1. Copy the **entire JSON content** from the file
2. In Railway dashboard, add this environment variable:
   ```
   FIREBASE_SERVICE_ACCOUNT_KEY={"type":"service_account","project_id":"your-project-id",...}
   ```
   **Important**: Make sure to escape quotes properly or use Railway's JSON input mode.

### Option B: Use Individual Values (More Secure)

Extract these specific values from the JSON:

1. **Project ID**:
   ```
   FIREBASE_PROJECT_ID=your-project-id
   ```

2. **Private Key** (keep the \n characters):
   ```
   FIREBASE_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...\n-----END PRIVATE KEY-----\n
   ```

3. **Client Email**:
   ```
   FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxxxx@your-project-id.iam.gserviceaccount.com
   ```

4. **Private Key ID** (optional but recommended):
   ```
   FIREBASE_PRIVATE_KEY_ID=abc123...
   ```

5. **Client ID** (optional but recommended):
   ```
   FIREBASE_CLIENT_ID=123456789...
   ```

## Step 6: Verify Setup

After setting the environment variables and deploying:

1. **Check Health Endpoint**:
   ```
   https://web-production-47673.up.railway.app/health
   ```
   Should show: `"firebase": {"available": true, "status": "initialized"}`

2. **Test Token Verification**:
   ```bash
   curl -X POST https://web-production-47673.up.railway.app/api/auth/verify \
     -H "Content-Type: application/json" \
     -d '{"token": "your-firebase-id-token"}'
   ```

## Common Issues & Solutions

### Issue: "Firebase service not available"
**Solution**: Check if environment variables are set correctly in Railway

### Issue: "Invalid service account key"
**Solutions**:
- Ensure JSON is properly formatted
- Check that quotes are escaped correctly
- Verify the private key includes `\n` characters

### Issue: "Project not found"
**Solution**: Verify the project ID matches your Firebase project

### Issue: "Permission denied"
**Solutions**:
- Ensure the service account has Firebase Admin SDK permissions
- Check that the project ID is correct
- Verify the service account is active

## Security Best Practices

1. **Never commit the JSON file to version control**
2. **Use Railway's secure environment variable storage**
3. **Regularly rotate service account keys**
4. **Use least-privilege access**
5. **Monitor usage in Firebase Console**

## Testing Your Setup

### Test 1: Health Check
```bash
curl https://web-production-47673.up.railway.app/health
```

### Test 2: Integration Status
```bash
curl https://web-production-47673.up.railway.app/api/integrations/status
```

### Test 3: Firebase Token Verification
```bash
# Get a Firebase ID token from your frontend, then:
curl -X POST https://web-production-47673.up.railway.app/api/auth/verify \
  -H "Content-Type: application/json" \
  -d '{"token": "YOUR_FIREBASE_ID_TOKEN"}'
```

## Next Steps

Once Firebase is configured:
1. Your backend can verify Firebase ID tokens
2. User authentication will work seamlessly
3. You can track user analytics in Tinybird
4. All user data will be properly authenticated

## Support

If you encounter issues:
1. Check Railway deployment logs
2. Verify environment variables in Railway dashboard
3. Test with the health endpoints
4. Check Firebase Console for any service issues
5. Ensure your Firebase project has billing enabled (required for some features)
