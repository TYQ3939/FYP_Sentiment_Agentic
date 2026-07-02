@echo off
REM ============================================================
REM Step 3: Run this on your Windows PC to upload the BERTweet
REM model files to the Google Cloud VM.
REM
REM Requirements:
REM   - Google Cloud SDK installed (gcloud CLI)
REM   - Run: gcloud auth login  (if not already logged in)
REM
REM Edit VM_NAME and ZONE below before running.
REM ============================================================

SET VM_NAME=fyp-sentiment-backend
SET ZONE=us-central1-a
SET REMOTE_PATH=/home/%USERNAME%/FYP_Sentiment_Agentic/sentiment_agentic/tools/models/bertweet_finetuned
SET LOCAL_MODEL=tools\models\bertweet_finetuned

echo ============================================================
echo  Uploading BERTweet model to Google Cloud VM
echo  VM: %VM_NAME% in %ZONE%
echo ============================================================
echo.

REM Upload only the essential model files (skip checkpoint dirs and training state)
echo [1/2] Uploading model weights and config...
gcloud compute scp "%LOCAL_MODEL%\model.safetensors" %VM_NAME%:%REMOTE_PATH%/ --zone=%ZONE%
gcloud compute scp "%LOCAL_MODEL%\config.json" %VM_NAME%:%REMOTE_PATH%/ --zone=%ZONE%

echo [2/2] Uploading tokenizer files...
gcloud compute scp "%LOCAL_MODEL%\added_tokens.json" %VM_NAME%:%REMOTE_PATH%/ --zone=%ZONE%
gcloud compute scp "%LOCAL_MODEL%\bpe.codes" %VM_NAME%:%REMOTE_PATH%/ --zone=%ZONE%
gcloud compute scp "%LOCAL_MODEL%\special_tokens_map.json" %VM_NAME%:%REMOTE_PATH%/ --zone=%ZONE%
gcloud compute scp "%LOCAL_MODEL%\tokenizer_config.json" %VM_NAME%:%REMOTE_PATH%/ --zone=%ZONE%
gcloud compute scp "%LOCAL_MODEL%\vocab.txt" %VM_NAME%:%REMOTE_PATH%/ --zone=%ZONE%

echo.
echo ============================================================
echo  Upload complete!
echo  SSH into the VM and run:
echo    sudo systemctl start fyp-backend
echo    sudo systemctl status fyp-backend
echo ============================================================
pause
