#!/bin/bash
# ============================================================
# Start the FastAPI backend in a screen session (RunPod)
# Run this inside the pod terminal each time the pod starts
# ============================================================

cd /workspace/FYP_Sentiment_Agentic/sentiment_agentic

# Kill any existing backend session
screen -S backend -X quit 2>/dev/null || true

# Start FastAPI inside a detached screen session
screen -dmS backend bash -c "
    source /workspace/FYP_Sentiment_Agentic/sentiment_agentic/.env 2>/dev/null || true
    export \$(cat .env | grep -v '^#' | xargs)
    python run_backend.py
"

sleep 2

echo "[OK] Backend started in screen session 'backend'"
echo ""
echo " Useful commands:"
echo "   View logs:    screen -r backend"
echo "   Detach:       Ctrl+A then D"
echo "   Stop server:  screen -S backend -X quit"
echo ""
echo " Health check:"
curl -s http://localhost:8000/health && echo "" || echo "[WARN] Server not up yet, wait 10s and try: curl http://localhost:8000/health"
