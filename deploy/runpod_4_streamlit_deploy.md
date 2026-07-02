# Step 4 — Deploy Frontend to Streamlit Community Cloud

## 1. Push your code to GitHub

Make sure all your latest code is pushed:
```bash
git add .
git commit -m "deploy: RunPod + Streamlit Cloud setup"
git push origin main
```

---

## 2. Deploy to Streamlit Community Cloud

1. Go to **share.streamlit.io** → Sign in with GitHub
2. Click **New app**
3. Fill in:

| Setting | Value |
|---|---|
| Repository | `TYQ3939/FYP_Sentiment_Agentic` |
| Branch | `main` |
| Main file path | `sentiment_agentic/frontend/app.py` |
| App URL | choose a name (e.g. `fyp-sentiment`) |

4. Expand **Advanced settings** — you will see a TOML text area.

Paste this into that TOML box:

```toml
[runtime]
python_version = "3.10"
```

> If you see a **Python version dropdown** instead (not a TOML box), just select `3.10` from the dropdown. Both UI styles exist depending on which version of Streamlit Cloud you see.

5. Click **Deploy**

---

## 3. Add secrets (also TOML format)

After deployment, go to your app → **⋮ menu top-right** → **Settings** → **Secrets**

You will see a TOML text area. Paste this (replace values with your real keys):

```toml
# Get this from RunPod dashboard -> your pod -> Connect -> HTTP Service -> port 8000
# Looks like: https://abc1def2gh3i-8000.proxy.runpod.net
API_BASE_URL = "https://YOUR_POD_ID-8000.proxy.runpod.net"

GROQ_API_KEY     = "your_groq_api_key_here"
TAVILY_API_KEY   = "your_tavily_api_key_here"
SERPER_API_KEY   = "your_serper_api_key_here"
GOOGLE_CSE_ID    = "your_google_cse_id_here"
GOOGLE_API_KEY   = "your_google_api_key_here"
```

Click **Save**. The app will reboot automatically.

> **Note:** The Advanced settings TOML (step 4) and the Secrets TOML (step 3) are two **separate** TOML boxes. Do not mix them — secrets go in the Secrets section, not Advanced settings.

---

## 4. Your app is live

Your public URL will be: `https://fyp-sentiment.streamlit.app` (or whatever name you chose)

---

## Daily workflow

**Before using the system:**
1. RunPod dashboard → Start your pod (if stopped)
2. Open web terminal → run: `bash /workspace/FYP_Sentiment_Agentic/sentiment_agentic/deploy/runpod_start.sh`
3. Open your Streamlit app URL

**After demo/done:**
1. RunPod dashboard → Stop the pod
2. You only pay for network volume storage (~$0.02/GB/month)

---

## Check RunPod backend URL

The URL changes if you create a new pod. To find it:

RunPod dashboard → Your pod → **Connect** → look for **HTTP Service** on port 8000.

It looks like: `https://abc1def2gh3i-8000.proxy.runpod.net`

Update Streamlit secrets whenever the URL changes.
