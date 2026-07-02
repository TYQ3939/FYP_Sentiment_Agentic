#!/bin/bash
# ============================================================
# Step 2: Run this script INSIDE the Google Cloud VM
# SSH into VM first, then paste and run this entire script
# ============================================================

set -e  # stop on any error

echo "============================================================"
echo " FYP Sentiment Analysis — VM Setup"
echo "============================================================"

# ── 1. System update ─────────────────────────────────────────
echo "[1/7] Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y git python3-pip python3-venv screen htop

# ── 2. Verify GPU ────────────────────────────────────────────
echo "[2/7] Checking GPU..."
if nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "[OK] GPU detected"
else
    echo "[WARN] nvidia-smi not found — Deep Learning VM image should have drivers pre-installed"
    echo "       If you did not use the Deep Learning VM image, re-create the VM with it."
    exit 1
fi

# ── 3. Clone repo ────────────────────────────────────────────
echo "[3/7] Cloning repository..."
if [ -d "FYP_Sentiment_Agentic" ]; then
    echo "      Repo already exists — pulling latest..."
    cd FYP_Sentiment_Agentic && git pull && cd ..
else
    git clone https://github.com/TYQ3939/FYP_Sentiment_Agentic.git
fi
cd FYP_Sentiment_Agentic/sentiment_agentic

# ── 4. Python virtual environment ────────────────────────────
echo "[4/7] Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 (matches the Deep Learning VM)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the requirements
pip install -r requirements.txt

echo "[OK] Python packages installed"

# ── 5. Create required directories ───────────────────────────
echo "[5/7] Creating required directories..."
mkdir -p data/filtered_data
mkdir -p data/analysis
mkdir -p data/raw_data
mkdir -p tools/models/bertweet_finetuned

# ── 6. Create .env file ──────────────────────────────────────
echo "[6/7] Creating .env file..."
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

# ── YouTube (optional fallback) ──────────────────────────────
YOUTUBE_API_KEY=your_youtube_api_key_here
ENVEOF
    echo "[OK] .env created — EDIT IT with your actual API keys before starting the backend!"
    echo "     Run: nano .env"
else
    echo "[OK] .env already exists — skipping"
fi

# ── 7. Install systemd service ───────────────────────────────
echo "[7/7] Installing systemd service for auto-start..."
WORKING_DIR="$(pwd)"
VENV_PYTHON="$(pwd)/venv/bin/python"

sudo tee /etc/systemd/system/fyp-backend.service > /dev/null << SERVICEEOF
[Unit]
Description=FYP Sentiment Analysis FastAPI Backend
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=${WORKING_DIR}
EnvironmentFile=${WORKING_DIR}/.env
ExecStart=${VENV_PYTHON} run_backend.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICEEOF

sudo systemctl daemon-reload
sudo systemctl enable fyp-backend

echo ""
echo "============================================================"
echo " Setup complete!"
echo "============================================================"
echo ""
echo " NEXT STEPS:"
echo " 1. Upload your BERTweet model files (run deploy/3_upload_model.bat on your Windows PC)"
echo " 2. Edit your API keys:  nano .env"
echo " 3. Start the backend:   sudo systemctl start fyp-backend"
echo " 4. Check status:        sudo systemctl status fyp-backend"
echo " 5. View logs:           journalctl -u fyp-backend -f"
echo ""
echo " Your VM external IP will be the API_BASE_URL for Streamlit:"
echo " http://$(curl -s ifconfig.me):8000"
echo ""
