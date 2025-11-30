@echo off
echo ==========================================================================
echo Face Mask Detection - Automated Pipeline
echo ==========================================================================
echo.

REM Configuration
set DATA_DIR=data\raw
set CHECKPOINT_DIR=models\checkpoints
set RESULTS_DIR=results\metrics
set TEST_DIR=tests\sample_images
set MODEL_NAME=resnet50
set EPOCHS=50
set BATCH_SIZE=32
set LEARNING_RATE=0.001

REM Step 1: Check Environment
echo Step 1: Checking environment...
echo.

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python or PyTorch not found!
    echo Please install requirements: pip install -r requirements.txt
    exit /b 1
)

echo.

REM Step 2: Check Data
echo Step 2: Checking dataset...
echo.

if not exist "%DATA_DIR%\images" (
    echo ERROR: Dataset not found in %DATA_DIR%
    echo Please download from: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
    exit /b 1
)

echo Dataset OK
echo.

REM Step 3: Create Directories
echo Step 3: Creating directories...
echo.

mkdir "%CHECKPOINT_DIR%" 2>nul
mkdir "%RESULTS_DIR%" 2>nul
mkdir "%TEST_DIR%" 2>nul
mkdir "results\predictions" 2>nul

echo Directories created
echo.

REM Step 4: Training
echo Step 4: Starting model training...
echo.
echo Configuration:
echo   Model: %MODEL_NAME%
echo   Epochs: %EPOCHS%
echo   Batch size: %BATCH_SIZE%
echo   Learning rate: %LEARNING_RATE%
echo.
echo This may take 1-2 hours depending on your hardware...
echo.

python src\train.py --data-dir "%DATA_DIR%" --model-name "%MODEL_NAME%" --epochs %EPOCHS% --batch-size %BATCH_SIZE% --lr %LEARNING_RATE% --checkpoint-dir "%CHECKPOINT_DIR%" --results-dir "%RESULTS_DIR%" --patience 10 --save-frequency 10

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Training failed!
    exit /b 1
)

echo.
echo Training completed successfully!
echo.

REM Step 5: Evaluation
echo Step 5: Evaluating model...
echo.

set BEST_MODEL=%CHECKPOINT_DIR%\best_model.pth

if not exist "%BEST_MODEL%" (
    echo ERROR: Best model not found at %BEST_MODEL%
    exit /b 1
)

echo Best model found: %BEST_MODEL%
echo.

REM Step 6: Sample Inference
echo Step 6: Running sample inference...
echo.

REM Copy sample images
if exist "%DATA_DIR%\images" (
    echo Copying sample images to %TEST_DIR%...
    
    set count=0
    for %%F in ("%DATA_DIR%\images\*") do (
        if !count! LSS 5 (
            copy "%%F" "%TEST_DIR%\" >nul
            set /a count+=1
        )
    )
    
    echo Copied sample images
)

REM Run inference
if exist "%TEST_DIR%" (
    echo.
    echo Running inference on test images...
    echo.
    
    python src\inference.py --folder "%TEST_DIR%" --model "%BEST_MODEL%" --output "results\predictions\sample_predictions.json" --save-viz results\predictions
)

echo.
echo ==========================================================================
echo Pipeline completed successfully!
echo ==========================================================================
echo.
echo Generated files:
echo   - Model weights: %BEST_MODEL%
echo   - Training history: %RESULTS_DIR%\training_history.png
echo   - Training metrics: %RESULTS_DIR%\training_history.json
echo   - Configuration: %RESULTS_DIR%\config.json
echo   - Sample predictions: results\predictions\
echo.
echo Next steps:
echo   1. Review training history: %RESULTS_DIR%\training_history.png
echo   2. Check detailed metrics: %RESULTS_DIR%\training_history.json
echo   3. Test on your own images:
echo      python src\inference.py --image your_image.jpg --model %BEST_MODEL% --visualize
echo.
echo ==========================================================================

pause