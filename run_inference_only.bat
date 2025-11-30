@echo off
REM Run sample inference only

REM ==== Config (samakan dengan script utama) ====
set DATA_DIR=data\raw
set CHECKPOINT_DIR=models\checkpoints
set TEST_DIR=tests\sample_images
set BEST_MODEL=%CHECKPOINT_DIR%\best_model.pth

REM ==== Buat folder kalau belum ada ====
mkdir "%TEST_DIR%" 2>nul
mkdir "results\predictions" 2>nul

REM ==== Copy sample images ====
echo Copying sample images to %TEST_DIR%...

setlocal enabledelayedexpansion
set count=0
for %%F in ("%DATA_DIR%\images\*.*") do (
    if !count! LSS 5 (
        copy "%%F" "%TEST_DIR%\" >nul
        set /a count+=1
    )
)
endlocal

echo Copied sample images

REM ==== Run inference ====
if exist "%TEST_DIR%" (
    echo.
    echo Running inference on test images...
    echo.

    python src\inference.py ^
        --folder "%TEST_DIR%" ^
        --model "%BEST_MODEL%" ^
        --output "results\predictions\sample_predictions.json" ^
        --save-viz "results\predictions"
)

echo.
echo Done. Check results in results\predictions\
pause
