# Step 1 — Create the Google Cloud VM

Do this once in the Google Cloud Console.

## 1. Go to Compute Engine

Console → Compute Engine → VM Instances → **Create Instance**

## 2. Fill in these settings

| Setting | Value |
|---|---|
| **Name** | `fyp-sentiment-backend` |
| **Region** | `us-central1` (Iowa — cheapest) |
| **Zone** | `us-central1-a` |
| **Machine type** | `n1-standard-2` (2 vCPU, 7.5 GB RAM) |
| **GPU** | Click "Add GPU" → NVIDIA T4 |

**Boot disk** — click "Change":
- Operating system: `Deep Learning on Linux`
- Version: `Deep Learning VM with CUDA 12.x` (PyTorch)
- Boot disk size: `50 GB`

**Firewall**:
- ✅ Allow HTTP traffic
- ✅ Allow HTTPS traffic

Click **Create**.

---

## 3. Open port 8000 for FastAPI

Console → **VPC Network** → **Firewall** → **Create Firewall Rule**

| Setting | Value |
|---|---|
| Name | `allow-fastapi` |
| Targets | All instances in the network |
| Source IPv4 ranges | `0.0.0.0/0` |
| Protocols and ports | TCP → `8000` |

Click **Create**.

---

## 4. SSH into the VM

Console → Compute Engine → VM Instances → click **SSH** button next to your VM.

A browser terminal will open. Run the setup script next (Step 2).

---

## 5. Note your External IP

On the VM Instances page, copy the **External IP** of your VM.
You will need it for the Streamlit secrets in Step 4.

Example: `34.123.45.67`  
Your backend URL will be: `http://34.123.45.67:8000`
