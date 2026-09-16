@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ==============================================================================
echo  🕹️ CARTPOLE-V1 PPO // STANDALONE COCKPIT DESKTOP APP (HWIHWA LAB)
echo     Cyberpunk Aerospace Control Deck · MicroDuck Style Native Window
echo ==============================================================================
echo.

python run.py

pause
