#!/bin/bash
# ============================================================
# Start the FastAPI backend in a screen session (RunPod)
# Run this inside the pod terminal each time the pod starts
# ============================================================

cd /workspace/FYP_Sentiment_Agentic

# Install screen if missing
if ! command -v screen &>/dev/null; then
    apt-get install -y screen -q
fi

# Kill any existing backend session
screen -S backend -X quit 2>/dev/null || true

# Load env vars and start FastAPI in a detached screen session
export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
screen -dmS backend python run_backend.py

sleep 3

echo "[OK] Backend started in screen session 'backend'"
echo ""
echo " Useful commands:"
echo "   View logs:    screen -r backend"
echo "   Detach:       Ctrl+A then D"
echo "   Stop server:  screen -S backend -X quit"
echo ""
echo " Health check:"
curl -s http://localhost:8000/health && echo "" || echo "[WARN] Server not up yet, wait 10s and try: curl http://localhost:8000/health"
