@echo off
setlocal enabledelayedexpansion

:: Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment .venv not found.
    echo Please create it and install requirements.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

:: Get video path from argument or prompt
set "VIDEO_PATH=%~1"
if "%VIDEO_PATH%"=="" (
    set /p "VIDEO_PATH=Please drag and drop your video file here or type its path: "
    set "VIDEO_PATH=!VIDEO_PATH:"=!"
)

if not exist "!VIDEO_PATH!" (
    echo Error: Video file "!VIDEO_PATH!" not found.
    pause
    exit /b 1
)

echo.
echo 🎬 Starting Kannada dubbing pipeline for: !VIDEO_PATH!
echo.

:: Run the ADK pipeline workflow
.venv\Scripts\adk.exe run video_localizer "Convert the audio of !VIDEO_PATH! to Kannada"

if %ERRORLEVEL% equ 0 (
    echo.
    echo 🎉 Dubbing completed successfully! Output is in the output/ folder.
) else (
    echo.
    echo ❌ Dubbing pipeline failed with error code %ERRORLEVEL%.
)

pause
