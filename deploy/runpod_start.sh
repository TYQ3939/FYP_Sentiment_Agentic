#!/bin/bash
# ============================================================
# Start the FastAPI backend on RunPod
# Run this inside the pod terminal each time the pod starts
# ============================================================

cd /workspace/FYP_Sentiment_Agentic

# ── Venv: create on first run, reuse on subsequent runs ──────
VENV=/workspace/venv
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[SETUP] Creating venv in /workspace/venv (one-time)..."
    python -m venv $VENV
    source $VENV/bin/activate
    pip install --upgrade pip setuptools wheel -q
    pip install -r requirements.txt -q
    echo "[OK] venv ready"
else
    source $VENV/bin/activate
    echo "[OK] venv activated"
fi

# ── Load environment variables ────────────────────────────────
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
    echo "[OK] .env loaded"
else
    echo "[WARN] .env not found - API keys will be missing"
fi

# ── Start FastAPI in background ───────────────────────────────
nohup python run_backend.py > /workspace/backend.log 2>&1 &
echo $! > /workspace/backend.pid

sleep 3

echo ""
echo "[OK] Backend started (PID: $(cat /workspace/backend.pid))"
echo ""
echo " Useful commands:"
echo "   View logs:   tail -f /workspace/backend.log"
echo "   Stop server: kill \$(cat /workspace/backend.pid)"
echo ""
echo " Health check:"
curl -s http://localhost:8000/health && echo "" || echo "[WARN] Not up yet — try: curl http://localhost:8000/health"
