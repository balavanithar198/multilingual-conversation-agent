@echo off
echo ========================================
echo   TANGLISH AGENT - SETUP AND RUN
echo ========================================
echo.

echo Step 1: Installing packages...
pip install "livekit-agents[openai]~=1.5" python-dotenv
echo.

echo Step 2: Checking setup...
python check_setup.py
echo.

echo Step 3: Downloading model files...
python agent.py download-files
echo.

echo Step 4: Starting agent...
echo.
echo ========================================
echo   AGENT IS STARTING
echo   Open https://cloud.livekit.io
echo   Go to your project -> Agents -> Console
echo   Click "Start a session" to talk!
echo ========================================
echo.
python agent.py dev

pause