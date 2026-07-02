# Step 4 — Deploy Frontend to Streamlit Community Cloud

## 1. Push your latest code to GitHub

```bash
git add .
git commit -m "Add cloud deployment config"
git push origin main
```

## 2. Go to Streamlit Community Cloud

Visit: https://share.streamlit.io  
Sign in with your GitHub account.

## 3. Create a new app

Click **New app** and fill in:

| Setting | Value |
|---|---|
| Repository | `TYQ3939/FYP_Sentiment_Agentic` |
| Branch | `main` |
| Main file path | `sentiment_agentic/frontend/app.py` |

## 4. Add secrets

Click **Advanced settings** → **Secrets** and paste:

```toml
API_BASE_URL = "http://YOUR_VM_EXTERNAL_IP:8000"

GROQ_API_KEY = "your_groq_api_key_here"
TAVILY_API_KEY = "your_tavily_api_key_here"
SERPER_API_KEY = "your_serper_api_key_here"
GOOGLE_CSE_ID = "your_google_cse_id_here"
GOOGLE_API_KEY = "your_google_api_key_here"
```

Replace `YOUR_VM_EXTERNAL_IP` with the External IP from Step 1.

## 5. Deploy

Click **Deploy**. Streamlit will install packages from `requirements_frontend.txt`
and launch the app. You will get a public URL like:

```
https://fyp-sentiment-agentic.streamlit.app
```

---

## Daily workflow (to save costs)

**Before presenting / using the app:**
```
Google Cloud Console → Compute Engine → VM Instances
→ click the 3-dot menu next to your VM → Start
```
Wait ~1 minute for the backend to start, then open your Streamlit link.

**After you are done:**
```
Google Cloud Console → Compute Engine → VM Instances  
→ click the 3-dot menu next to your VM → Stop
```
The VM stops billing (you only pay ~$0.07/day for disk storage while stopped).
