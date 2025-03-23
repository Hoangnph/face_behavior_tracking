@echo off
:: Windows batch script to set up the environment for Human Tracking project

echo =====================================================
echo Human Tracking - Environment Setup
echo =====================================================

:: Check if conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Conda is not installed or not in PATH.
    echo Please install Conda first: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
    exit /b 1
)

:: Get script directory and project root
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

:: Change to project root
cd /d "%PROJECT_ROOT%"

:: Check for environment.yml
if not exist environment.yml (
    echo environment.yml not found in project root!
    exit /b 1
)

:: Check if environment already exists
conda env list | findstr /C:"human_tracking " >nul
if %ERRORLEVEL% neq 0 (
    :: Create conda environment
    echo Creating conda environment from environment.yml...
    conda env create -f environment.yml
) else (
    :: Update existing environment
    echo Environment 'human_tracking' already exists. Updating...
    conda env update -f environment.yml
)

:: Check if environment creation was successful
if %ERRORLEVEL% neq 0 (
    echo Failed to create/update conda environment.
    exit /b 1
)

echo Conda environment created/updated successfully.

:: Install ByteTrack via pip
echo Installing ByteTrack from GitHub...
call conda activate human_tracking
pip install git+https://github.com/ifzhang/ByteTrack.git

if %ERRORLEVEL% neq 0 (
    echo ByteTrack installation may have failed. You may need to install it manually.
) else (
    echo ByteTrack installed successfully.
)

:: Download YOLOv8 base model if it doesn't exist
set MODEL_DIR=%PROJECT_ROOT%\models
set YOLO_MODEL=%MODEL_DIR%\yolov8n.pt

if not exist "%MODEL_DIR%" (
    echo Creating models directory...
    mkdir "%MODEL_DIR%"
)

if not exist "%YOLO_MODEL%" (
    echo Downloading YOLOv8n model...
    call conda activate human_tracking
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    
    if %ERRORLEVEL% neq 0 (
        echo Failed to download YOLOv8 model. You may need to download it manually.
    ) else (
        echo YOLOv8 model downloaded successfully.
    )
) else (
    echo YOLOv8 model already exists at %YOLO_MODEL%
)

:: Verify environment setup
echo Verifying environment setup...
call conda activate human_tracking
python "%SCRIPT_DIR%setup_environment.py"

if %ERRORLEVEL% equ 0 (
    echo =====================================================
    echo Environment setup completed successfully!
    echo =====================================================
    echo To activate the environment, run:
    echo conda activate human_tracking
) else (
    echo =====================================================
    echo Environment verification failed. Please check the errors above.
    echo =====================================================
)

exit /b 0 