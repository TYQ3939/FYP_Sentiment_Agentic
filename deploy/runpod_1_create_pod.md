# Step 1 — Create a RunPod Pod

## 1. Sign up and add credits

1. Go to **runpod.io** → Sign up
2. Top-right → **Billing** → Add credits (minimum $10 recommended)

---

## 2. Add your SSH public key (needed to upload model files later)

1. RunPod dashboard → top-right profile → **Settings** → **SSH Public Keys**
2. Add your Windows SSH public key.

**If you don't have an SSH key yet**, open PowerShell and run:
```powershell
ssh-keygen -t rsa -b 4096
# Press Enter for all prompts (default location, no passphrase)
```
Then copy the public key:
```powershell
cat $HOME\.ssh\id_rsa.pub
```
Paste that entire output into RunPod's SSH key field.

---

## 3. Create a Network Volume (persistent storage)

This keeps your model files and data when the pod is stopped.

1. Left sidebar → **Storage** → **+ New Network Volume**

| Setting | Value |
|---|---|
| Name | `fyp-storage` |
| Size | `20 GB` |
| Region | `US-TX-3` or any US region |

Click **Create**.

---

## 4. Deploy a Pod

1. Left sidebar → **Pods** → **+ Deploy**

You will see a page with GPU options listed as cards.

2. Pick any of these GPUs (click the card):
   - **RTX 3080** (~$0.18/hr)
   - **RTX 3090** (~$0.22/hr)
   - **RTX 4070** (~$0.20/hr)

3. A panel will open on the right side of the screen (or below the GPU list).

---

### In that panel, configure the following:

**A — Template** (top of panel)
- Click the template dropdown
- Select `RunPod Pytorch 2.1` (look for the one that says PyTorch + CUDA)

---

**B — Customize Deployment button**

Look for a button or link that says **"Customize Deployment"** or **"Edit Template"** — it is usually shown as a small link/button below the template selector. Click it to expand more settings.

If you don't see it, look for a gear icon or an **"Advanced"** toggle.

---

**C — Storage settings** (inside Customize Deployment)

You will see two disk fields:

```
Container Disk:   [ 15 ] GB     ← set this to 15
Volume Disk:      [ dropdown ]  ← select "fyp-storage"
Volume Path:      /workspace    ← type this if empty
```

- **Container Disk** = temporary storage (lost when pod stops). Set to `15 GB`.
- **Volume Disk** = your persistent storage. Click the dropdown and select `fyp-storage` (the one you created in Step 3).
- **Volume Path** = where the volume is mounted inside the pod. Type `/workspace`.

---

**D — Ports** (also inside Customize Deployment)

Look for a **"Expose Ports"** or **"HTTP Ports / TCP Ports"** section:

```
HTTP Ports:  8000      ← type this (for FastAPI)
TCP Ports:   22        ← type this (for SSH file upload)
```

> Note: some RunPod versions show a single "Expose Ports" box — just type `8000` there.

---

4. Click **Deploy On-Demand** (or **Deploy** button at the bottom of the panel).

---

## 5. Note your Pod details

Once the pod is running, click on it to expand:

- **Pod ID** — looks like `abc1def2gh3i` (you need this for the Streamlit URL)
- **SSH command** — looks like `ssh root@ssh.runpod.io -p 12345 -i ~/.ssh/id_rsa`
- **HTTP Proxy URL for port 8000** — `https://{pod_id}-8000.proxy.runpod.net`

Copy all three. You'll need them in the next steps.

---

## 6. Open the pod terminal

Click **Connect** → **Start Web Terminal** (or use the SSH command in PowerShell).

Run the setup script next (Step 2).
