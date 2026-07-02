@echo off
REM ============================================================
REM Step 3: Upload BERTweet model files to RunPod
REM Run this on your Windows PC after the pod is running
REM ============================================================

REM ── EDIT THESE TWO LINES ─────────────────────────────────────────
set SSH_PORT=29937
REM Get your SSH port from: RunPod dashboard -> your pod -> Connect -> SSH over exposed TCP
REM It looks like:  ssh root@ssh.runpod.io -p 12345
REM                                                 ^^^^^  that number goes here
REM ─────────────────────────────────────────────────────────────────

REM Path to your local BERTweet model folder
set MODEL_DIR="E:\HP_E\01_MMU stuff\FYP\02_FYP Python files\105_Project_Code\sentiment_agentic\tools\models\bertweet_finetuned"

REM Remote path on RunPod
set REMOTE_PATH=/workspace/FYP_Sentiment_Agentic/sentiment_agentic/tools/models/bertweet_finetuned/

echo ============================================================
echo  Uploading BERTweet model to RunPod
echo  SSH port: %SSH_PORT%
echo  Local:    %MODEL_DIR%
echo  Remote:   root@ssh.runpod.io:%REMOTE_PATH%
echo ============================================================
echo.

REM Upload all model files
scp -P %SSH_PORT% -i "%USERPROFILE%\.ssh\id_rsa" -r %MODEL_DIR%\* root@ssh.runpod.io:%REMOTE_PATH%

echo.
echo ============================================================
echo  Upload complete!
echo  Verify on the pod with:
echo    ls /workspace/FYP_Sentiment_Agentic/sentiment_agentic/tools/models/bertweet_finetuned/
echo ============================================================
pause
