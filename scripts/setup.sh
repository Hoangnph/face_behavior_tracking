#!/bin/bash
# Shell script to set up the environment for Human Tracking project

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    
    # Colors
    local RED='\033[0;31m'
    local GREEN='\033[0;32m'
    local YELLOW='\033[0;33m'
    local BLUE='\033[0;34m'
    local NC='\033[0m' # No Color
    
    case $color in
        "red") echo -e "${RED}$message${NC}" ;;
        "green") echo -e "${GREEN}$message${NC}" ;;
        "yellow") echo -e "${YELLOW}$message${NC}" ;;
        "blue") echo -e "${BLUE}$message${NC}" ;;
        *) echo "$message" ;;
    esac
}

# Print header
print_status "blue" "====================================================="
print_status "blue" "Human Tracking - Environment Setup"
print_status "blue" "====================================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_status "red" "Conda is not installed or not in PATH."
    print_status "yellow" "Please install Conda first: https://docs.conda.io/projects/conda/en/latest/user-guide/install/"
    exit 1
fi

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Check for environment.yml
if [ ! -f "environment.yml" ]; then
    print_status "red" "environment.yml not found in project root!"
    exit 1
fi

# Check if environment already exists
ENV_EXISTS=$(conda env list | grep "^human_tracking " | wc -l)

if [ $ENV_EXISTS -eq 0 ]; then
    # Create conda environment
    print_status "blue" "Creating conda environment from environment.yml..."
    conda env create -f environment.yml
else
    # Update existing environment
    print_status "yellow" "Environment 'human_tracking' already exists. Updating..."
    conda env update -f environment.yml
fi

# Check if environment creation was successful
if [ $? -ne 0 ]; then
    print_status "red" "Failed to create/update conda environment."
    exit 1
fi

print_status "green" "Conda environment created/updated successfully."

# Install ByteTrack via pip
print_status "blue" "Installing ByteTrack from GitHub..."
eval "$(conda shell.bash hook)"
conda activate human_tracking
pip install git+https://github.com/ifzhang/ByteTrack.git

if [ $? -ne 0 ]; then
    print_status "yellow" "ByteTrack installation may have failed. You may need to install it manually."
else
    print_status "green" "ByteTrack installed successfully."
fi

# Download YOLOv8 base model if it doesn't exist
MODEL_DIR="$PROJECT_ROOT/models"
YOLO_MODEL="$MODEL_DIR/yolov8n.pt"

if [ ! -d "$MODEL_DIR" ]; then
    print_status "blue" "Creating models directory..."
    mkdir -p "$MODEL_DIR"
fi

if [ ! -f "$YOLO_MODEL" ]; then
    print_status "blue" "Downloading YOLOv8n model..."
    conda activate human_tracking
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 
    
    if [ $? -ne 0 ]; then
        print_status "yellow" "Failed to download YOLOv8 model. You may need to download it manually."
    else
        print_status "green" "YOLOv8 model downloaded successfully."
    fi
else
    print_status "green" "YOLOv8 model already exists at $YOLO_MODEL"
fi

# Verify environment setup
print_status "blue" "Verifying environment setup..."
conda activate human_tracking
python "$SCRIPT_DIR/setup_environment.py"

if [ $? -eq 0 ]; then
    print_status "green" "====================================================="
    print_status "green" "Environment setup completed successfully!"
    print_status "green" "====================================================="
    print_status "blue" "To activate the environment, run:"
    print_status "yellow" "conda activate human_tracking"
else
    print_status "red" "====================================================="
    print_status "red" "Environment verification failed. Please check the errors above."
    print_status "red" "====================================================="
fi

exit 0 