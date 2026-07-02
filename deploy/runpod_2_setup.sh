#!/bin/bash
# ============================================================
# Step 2: Run this script INSIDE the RunPod pod terminal
# Open the web terminal from RunPod dashboard, then paste and run
# ============================================================

set -e

echo "============================================================"
echo " FYP Sentiment Analysis - RunPod Setup"
echo "============================================================"

# ── 1. Check GPU ─────────────────────────────────────────────
echo "[1/6] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[OK] GPU ready"

# ── 2. Install system packages ────────────────────────────────
echo "[2/6] Installing system packages..."
apt-get update -y -q
apt-get install -y git screen htop nano curl -q
echo "[OK] System packages installed"

# ── 3. Clone repo into /workspace (persistent network volume) ─
echo "[3/6] Cloning repository..."
cd /workspace
if [ -d "FYP_Sentiment_Agentic" ]; then
    echo "      Repo already exists - pulling latest..."
    cd FYP_Sentiment_Agentic && git pull && cd ..
else
    git clone https://github.com/TYQ3939/FYP_Sentiment_Agentic.git
fi
cd FYP_Sentiment_Agentic/sentiment_agentic

# ── 4. Install Python packages ────────────────────────────────
# (PyTorch + CUDA already installed in RunPod Pytorch template)
echo "[4/6] Installing Python packages..."
pip install --upgrade pip -q

# Install requirements (skip torch — already in template)
pip install -r requirements.txt -q

echo "[OK] Python packages installed"

# ── 5. Create required directories ───────────────────────────
echo "[5/6] Creating directories..."
mkdir -p data/filtered_data
mkdir -p data/analysis
mkdir -p data/raw_data
mkdir -p tools/models/bertweet_finetuned
echo "[OK] Directories created"

# ── 6. Create .env file ──────────────────────────────────────
echo "[6/6] Creating .env file..."
if [ ! -f ".env" ]; then
    cat > .env << 'ENVEOF'
# ── API Keys ──────────────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
SERPER_API_KEY=your_serper_api_key_here
GOOGLE_CSE_ID=your_google_cse_id_here
GOOGLE_API_KEY=your_google_api_key_here

# ── MongoDB Atlas ─────────────────────────────────────────────
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/

# ── YouTube (optional) ───────────────────────────────────────
YOUTUBE_API_KEY=your_youtube_api_key_here
ENVEOF
    echo "[OK] .env created"
    echo "     IMPORTANT: Edit it now with your real API keys:"
    echo "     nano .env"
else
    echo "[OK] .env already exists - skipping"
fi

echo ""
echo "============================================================"
echo " Setup complete!"
echo "============================================================"
echo ""
echo " NEXT STEPS:"
echo " 1. Upload BERTweet model: run deploy/runpod_3_upload_model.bat on your Windows PC"
echo " 2. Fill in API keys:      nano .env"
echo " 3. Start the backend:     bash deploy/runpod_start.sh"
echo ""
echo " Your backend URL for Streamlit secrets:"
echo " https://\$(hostname)-8000.proxy.runpod.net"
echo " (Or check the HTTP Service URL in the RunPod dashboard)"
echo ""
