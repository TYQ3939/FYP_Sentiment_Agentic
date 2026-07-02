# Deployment Guide

Follow these steps in order.

## Steps

| Step | File | Where to run |
|---|---|---|
| 1 | `runpod_1_create_pod.md` | RunPod dashboard (browser) |
| 2 | `runpod_2_setup.sh` | Inside the RunPod pod terminal |
| 3 | `runpod_3_upload_model.bat` | Your Windows PC |
| 4 | `runpod_4_streamlit_deploy.md` | Streamlit Community Cloud (browser) |

---

## Architecture

```
User browser
    |
Streamlit Community Cloud (free)
https://your-app.streamlit.app
    | HTTP requests to port 8000
RunPod Pod -- FastAPI + BERTweet (GPU)
https://{pod_id}-8000.proxy.runpod.net
    | reads/writes
MongoDB Atlas (free tier)
```

---

## Quick commands (run inside pod terminal)

**Start backend:**
```bash
bash /workspace/FYP_Sentiment_Agentic/sentiment_agentic/deploy/runpod_start.sh
```

**View live logs:**
```bash
screen -r backend
# Press Ctrl+A then D to detach without stopping
```

**Check if backend is running:**
```bash
curl http://localhost:8000/health
```

**Update code on pod:**
```bash
cd /workspace/FYP_Sentiment_Agentic
git pull origin main
screen -S backend -X quit
bash sentiment_agentic/deploy/runpod_start.sh
```

---

## Cost summary

| Resource | Cost |
|---|---|
| Streamlit Community Cloud | Free |
| RunPod pod while RUNNING (RTX 3080) | ~$0.18/hr |
| RunPod pod while STOPPED | ~$0 (only network volume) |
| RunPod Network Volume (20 GB) | ~$0.40/month |
| MongoDB Atlas M0 | Free |
| **Total for FYP demo** | **~$0.20/hr when running** |

---

## Daily workflow

1. Go to runpod.io → Start your pod
2. Open web terminal → run `bash /workspace/FYP_Sentiment_Agentic/sentiment_agentic/deploy/runpod_start.sh`
3. Open your Streamlit app → use normally
4. When done → RunPod dashboard → Stop pod
