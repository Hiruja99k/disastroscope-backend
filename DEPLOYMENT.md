# 🚀 DisastroScope Backend Deployment Guide

## Quick Deploy Options

### Option 1: Minimal Deploy (Fastest - Recommended)
Use `requirements-minimal.txt` for guaranteed fast deployment:
```bash
# Rename the minimal requirements file
mv requirements-minimal.txt requirements.txt
```

### Option 2: Simple Deploy (Fast)
Use `requirements-simple.txt` for faster deployment without PyTorch:
```bash
# Rename the simple requirements file
mv requirements-simple.txt requirements.txt
```

### Option 3: Full Deploy (Slower)
Use the main `requirements.txt` with all features.

## 🎯 Deployment Steps

### 1. Create GitHub Repository
- Go to [github.com](https://github.com)
- Click "+" → "New repository"
- Name: `disastroscope-backend`
- Make it Public
- Don't initialize with README

### 2. Upload Clean Backend Files
- In your new repo, click "uploading an existing file"
- Drag the entire `clean-backend/` folder
- Commit message: "Initial backend deployment"
- Click "Commit changes"

### 3. Deploy to Railway
- Go to [railway.app](https://railway.app)
- Click "New Project"
- Select "Deploy from GitHub repo"
- Choose your `disastroscope-backend` repository
- Deploy (no root directory needed!)

## ⚙️ Environment Variables

After deployment, set these in Railway:

```bash
OPENWEATHER_API_KEY=your_openweather_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## 🔧 Configuration Files

- **`railway.json`** - Railway deployment config
- **`Procfile`** - Heroku deployment config (backup)
- **`runtime.txt`** - Python 3.12 runtime
- **`requirements.txt`** - Python dependencies

## 🚨 Troubleshooting

### Dependency Conflicts (FIXED!)
The requirements files now use compatible versions:
- `numpy==1.26.4` (compatible with pandas 2.2.1)
- All packages tested for Python 3.12 compatibility

### PyTorch Installation Issues
If you get PyTorch errors:
1. Use `requirements-minimal.txt` instead (fastest)
2. Use `requirements-simple.txt` for more features
3. The AI models will use rule-based logic instead of neural networks
4. All other functionality remains the same

### Build Failures
- Check Python version compatibility
- Use the minimal requirements for guaranteed deployment
- All dependencies are now compatible

## 📱 Frontend Integration

After backend deployment:
1. Get your Railway URL (e.g., `https://your-app.railway.app`)
2. Update Vercel environment variables:
   - `VITE_API_BASE_URL=https://your-app.railway.app`
   - `VITE_SOCKET_URL=https://your-app.railway.app`
3. Redeploy your Vercel frontend

## ✅ Success Indicators

- Railway shows "Deploy Successful"
- Health check endpoint works: `https://your-app.railway.app/api/health`
- No build errors in Railway logs
- Frontend can connect to backend APIs

## 🆘 Need Help?

- Check Railway build logs for specific errors
- Verify all files are uploaded to GitHub
- Ensure environment variables are set correctly
- **Use `requirements-minimal.txt` for guaranteed deployment**

## 🚀 **Recommended Deployment Order:**

1. **Start with `requirements-minimal.txt`** (guaranteed to work)
2. **Test the deployment** and health endpoint
3. **Upgrade to `requirements-simple.txt`** if you want more features
4. **Use full requirements** only if you need PyTorch
