#!/bin/bash

###############################################################################
# This script automates the complete pipeline:
# 1. Environment setup verification
# 2. Data preparation
# 3. Model training
# 4. Model evaluation
# 5. Sample inference
#
# Usage:
#   bash run.sh
###############################################################################

set -e  # Exit on error

echo "=========================================================================="
echo "Face Mask Detection - Automated Pipeline"
echo "=========================================================================="
echo ""

# Configuration
DATA_DIR="data/raw"
CHECKPOINT_DIR="models/checkpoints"
RESULTS_DIR="results/metrics"
TEST_DIR="tests/sample_images"
MODEL_NAME="resnet50"
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.001

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

###############################################################################
# Step 1: Check Environment
###############################################################################

echo -e "${GREEN}Step 1: Checking environment...${NC}"
echo ""

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if required packages are installed
echo "Checking required packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo -e "${RED}PyTorch not found. Please install requirements:${NC}"
    echo "pip install -r requirements.txt"
    exit 1
}

python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')" || {
    echo -e "${RED}TorchVision not found. Please install requirements.${NC}"
    exit 1
}

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo -e "${GREEN}GPU acceleration enabled${NC}"
    DEVICE="cuda"
else
    echo -e "${YELLOW}GPU not available, using CPU${NC}"
    DEVICE="cpu"
fi

echo ""

###############################################################################
# Step 2: Check Data
###############################################################################

echo -e "${GREEN}Step 2: Checking dataset...${NC}"
echo ""

if [ ! -d "$DATA_DIR/images" ] || [ ! -d "$DATA_DIR/annotations" ]; then
    echo -e "${RED}Dataset not found in $DATA_DIR${NC}"
    echo ""
    echo "Please download the dataset from:"
    echo "https://www.kaggle.com/datasets/andrewmvd/face-mask-detection"
    echo ""
    echo "And extract it to $DATA_DIR/"
    exit 1
fi

# Count images and annotations
num_images=$(find "$DATA_DIR/images" -type f | wc -l)
num_annotations=$(find "$DATA_DIR/annotations" -type f -name "*.xml" | wc -l)

echo "Found $num_images images"
echo "Found $num_annotations annotations"

if [ "$num_images" -eq 0 ] || [ "$num_annotations" -eq 0 ]; then
    echo -e "${RED}Dataset is empty or incomplete${NC}"
    exit 1
fi

echo -e "${GREEN}Dataset OK${NC}"
echo ""

###############################################################################
# Step 3: Create Directories
###############################################################################

echo -e "${GREEN}Step 3: Creating directories...${NC}"
echo ""

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$TEST_DIR"
mkdir -p "results/predictions"

echo "Directories created"
echo ""

###############################################################################
# Step 4: Training
###############################################################################

echo -e "${GREEN}Step 4: Starting model training...${NC}"
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Device: $DEVICE"
echo ""
echo "This may take 1-2 hours depending on your hardware..."
echo ""

python src/train.py \
    --data-dir "$DATA_DIR" \
    --model-name "$MODEL_NAME" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --device "$DEVICE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --results-dir "$RESULTS_DIR" \
    --patience 10 \
    --save-frequency 10

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Training completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}Training failed!${NC}"
    exit 1
fi

echo ""

###############################################################################
# Step 5: Evaluation
###############################################################################

echo -e "${GREEN}Step 5: Evaluating model...${NC}"
echo ""

# Check if best model exists
BEST_MODEL="$CHECKPOINT_DIR/best_model.pth"

if [ ! -f "$BEST_MODEL" ]; then
    echo -e "${RED}Best model not found at $BEST_MODEL${NC}"
    exit 1
fi

echo "Best model found: $BEST_MODEL"
echo ""

# Get model info
python -c "
import torch
checkpoint = torch.load('$BEST_MODEL', map_location='cpu')
print(f'Model trained for {checkpoint[\"epoch\"]+1} epochs')
print(f'Best validation accuracy: {checkpoint[\"val_acc\"]:.4f}')
print(f'Model size: {sum(p.numel() for p in checkpoint[\"model_state_dict\"].values()):,} parameters')
"

echo ""

###############################################################################
# Step 6: Sample Inference
###############################################################################

echo -e "${GREEN}Step 6: Running sample inference...${NC}"
echo ""

# Check if test images exist
if [ -d "$DATA_DIR/images" ]; then
    # Copy a few sample images for testing
    echo "Copying sample images to $TEST_DIR..."
    
    # Get first 5 images
    count=0
    for img in "$DATA_DIR/images"/*; do
        if [ $count -lt 5 ]; then
            cp "$img" "$TEST_DIR/"
            count=$((count+1))
        else
            break
        fi
    done
    
    echo "Copied $count sample images"
fi

# Run inference on test images
if [ -d "$TEST_DIR" ] && [ "$(ls -A $TEST_DIR)" ]; then
    echo ""
    echo "Running inference on test images..."
    echo ""
    
    python src/inference.py \
        --folder "$TEST_DIR" \
        --model "$BEST_MODEL" \
        --device "$DEVICE" \
        --output "results/predictions/sample_predictions.json" \
        --save-viz "results/predictions/"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}Inference completed successfully!${NC}"
    else
        echo ""
        echo -e "${YELLOW}Inference failed (non-critical)${NC}"
    fi
else
    echo -e "${YELLOW}No test images found in $TEST_DIR${NC}"
fi

echo ""

###############################################################################
# Step 7: Summary
###############################################################################

echo "=========================================================================="
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo "=========================================================================="
echo ""
echo "Generated files:"
echo "  - Model weights: $BEST_MODEL"
echo "  - Training history: $RESULTS_DIR/training_history.png"
echo "  - Training metrics: $RESULTS_DIR/training_history.json"
echo "  - Configuration: $RESULTS_DIR/config.json"
echo "  - Sample predictions: results/predictions/"
echo ""
echo "Next steps:"
echo "  1. Review training history: $RESULTS_DIR/training_history.png"
echo "  2. Check detailed metrics: $RESULTS_DIR/training_history.json"
echo "  3. Test on your own images:"
echo "     python src/inference.py --image your_image.jpg --model $BEST_MODEL --visualize"
echo ""
echo "For more information, see README.md"
echo "=========================================================================="