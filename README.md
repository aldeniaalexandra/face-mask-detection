# Face Mask Detection

AI model for detecting face masks using deep learning. This project implements a robust classification system to identify whether a person is wearing a mask correctly, incorrectly, or not at all.

## Project Overview

This project uses transfer learning with ResNet-50 to classify face mask usage into three categories:
- **with_mask**: Person wearing a mask correctly
- **without_mask**: Person not wearing a mask
- **mask_weared_incorrect**: Person wearing a mask incorrectly

### Approach Summary

**Model Architecture**: ResNet-50 pretrained on ImageNet with a custom classification head. We chose ResNet-50 because of its proven performance on image classification tasks and good balance between accuracy and computational efficiency.

### Design Decision: Classification vs Object Detection

**Why Classification (ResNet-50) instead of Object Detection (YOLO)?**

This project uses a **classification approach** rather than object detection for several strategic reasons:

1. **Dataset Nature**: The Kaggle dataset provides pre-computed bounding boxes in Pascal VOC XML format. Face detection is already solved - we only need to classify mask usage.

2. **Evaluation Focus**: The technical test emphasizes model evaluation metrics and code quality. Classification provides clearer metrics (accuracy, precision, recall, F1) compared to detection metrics (mAP, IoU).

3. **Resource Efficiency**: Classification trains faster (1-2 hours vs 3-5 hours for YOLO) and requires less GPU memory, which is important given the 3-day deadline.

4. **Transfer Learning Benefits**: ResNet-50 pretrained on ImageNet provides excellent feature representations for face-related tasks.

5. **Dataset Size**: With ~900 face samples, classification with transfer learning is more appropriate than training a detector from scratch.

**When would YOLO be better?**
- Real-world deployment without pre-computed bounding boxes
- Multiple faces requiring simultaneous detection
- Real-time video processing
- End-to-end pipeline from raw camera feed

**For Production**: I recommend using MTCNN/RetinaFace for face detection + this trained classifier, or training YOLOv8 end-to-end with more data.

**Training Strategy**: 
- Transfer learning from ImageNet weights
- Data augmentation (rotation, flip, brightness/contrast adjustment)
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting
- 80-20 train-validation split

**Key Design Decisions**:
1. **Preprocessing**: Extract face regions from bounding boxes with 10% padding to include context
2. **Augmentation**: Moderate augmentation to improve generalization without distorting faces
3. **Loss Function**: CrossEntropyLoss suitable for multi-class classification
4. **Optimizer**: Adam with weight decay (L2 regularization)
5. **Metrics**: Macro-averaged precision, recall, and F1-score to handle class imbalance


### Results

After 40 epochs of training (early stopping applied), the model achieved:

- **Final Training Accuracy**: 97.79%
- **Final Validation Accuracy**: 97.06%
- **Best Validation Accuracy**: 97.06% (Epoch 40)
- **Final Training Loss**: 0.0676
- **Final Validation Loss**: 0.1250

**Training Progress:**
- Initial accuracy (Epoch 1): Train 87.84%, Val 90.06%
- Steady improvement over first 15 epochs
- Learning rate reduced at epochs 12, 18, and 36 (ReduceLROnPlateau)
- Model converged with minimal overfitting (0.73% gap between train and val accuracy)

**Key Observations:**
1. **Excellent Generalization**: Only 0.73% difference between training (97.79%) and validation (97.06%) accuracy indicates the model generalizes well without significant overfitting.

2. **Effective Learning Rate Scheduling**: The ReduceLROnPlateau scheduler automatically reduced learning rate from 0.001 → 0.0005 → 0.00025 → 0.000125, allowing fine-tuning in later epochs.

3. **Stable Training**: Validation accuracy improved consistently from 90.06% to 97.06%, with minor fluctuations around epoch 7 and 12.

4. **Early Stopping Not Triggered**: Model continued improving until epoch 40, showing the benefit of patience in training.

The model performs excellently across all three classes with balanced precision and recall. Detailed metrics, training curves, and confusion matrix are available in `results/metrics/`.

## Project Structure

```
face-mask-detection/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Preprocessed data
│   └── annotations/            # XML annotations
├── models/
│   ├── checkpoints/            # Training checkpoints
│   └── best_model.pth          # Best trained model
├── notebooks/
│   └── face_mask_detection.ipynb  # Main notebook
├── src/
│   ├── __init__.py
│   ├── dataset.py              # Dataset loading
│   ├── model.py                # Model architecture
│   ├── train.py                # Training script
│   ├── utils.py                # Utility functions
│   └── inference.py            # Inference script
├── tests/
│   └── sample_images/          # Test images
├── results/
│   ├── predictions/            # Prediction outputs
│   └── metrics/                # Evaluation metrics
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── run.sh                      # Automation script
```

## Setup Instructions

### Installation

1. **Clone or download this repository**

```bash
git clone <your-repo-url>
cd face-mask-detection
```

2. **Create virtual environment**

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

Or using conda:

```bash
conda create -n face-mask python=3.9
conda activate face-mask
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download dataset**

Download the Face Mask Detection dataset from Kaggle:
https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

Extract to `data/raw/` directory:

```bash
# Using Kaggle API
pip install kaggle
kaggle datasets download -d andrewmvd/face-mask-detection
unzip face-mask-detection.zip -d data/raw/

# Or manually download and extract
```

Your directory structure should look like:
```
data/raw/
├── images/
│   ├── maksssksksss0.png
│   ├── maksssksksss1.png
│   └── ...
└── annotations/
    ├── maksssksksss0.xml
    ├── maksssksksss1.xml
    └── ...
```

## Usage

### Training

**Option 1: Using Jupyter Notebook (Recommended for exploration)**

```bash
jupyter notebook notebooks/face_mask_detection.ipynb
```

The notebook contains:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Model training with detailed explanations
- Evaluation and visualization
- Complete train-of-thought comments

**Option 2: Using Python Script**

```bash
# Basic training
python src/train.py --data-dir data/raw --epochs 50 --batch-size 32

# With custom parameters
python src/train.py \
    --data-dir data/raw \
    --model-name resnet50 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --device cuda \
    --checkpoint-dir models/checkpoints \
    --results-dir results/metrics
```

**Training Parameters**:
- `--data-dir`: Directory containing images and annotations (default: data/raw)
- `--model-name`: Model architecture (resnet50 or efficientnet)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--device`: Device (cuda or cpu)
- `--patience`: Early stopping patience (default: 10)

**Option 3: Using Automation Script**

**Linux / MacOS**

```bash
bash run.sh
```

**Windows**

Jalankan file **run.bat**:

```bat
run.bat
```

Atau lewat Command Prompt / PowerShell:

```bat
./run.bat
```

### Inference

**Single Image Prediction**

```bash
python src/inference.py \
    --image path/to/image.jpg \
    --model models/checkpoints/best_model.pth \
    --visualize
```

**Batch Prediction on Folder**

```bash
python src/inference.py \
    --folder tests/sample_images/ \
    --model models/checkpoints/best_model.pth \
    --output results/predictions/predictions.json \
    --save-viz results/predictions/
```

**Inference Parameters**:
- `--image`: Path to single image
- `--folder`: Path to folder containing images
- `--model`: Path to trained model (required)
- `--output`: Path to save predictions JSON
- `--visualize`: Display predictions
- `--save-viz`: Directory to save visualization images
- `--device`: Device (cuda or cpu)

### Expected Output

**Training Output**:
```
Epoch 1/50 [Train]: 100%|████████| 234/234 [02:15<00:00]
Epoch 1/50 [Val]:   100%|████████| 59/59 [00:28<00:00]

Epoch 1/50 Summary:
  Train Loss: 0.4521 | Train Acc: 0.8234
  Val Loss:   0.3892 | Val Acc:   0.8567
  
  Validation Metrics:
    Precision: 0.8523
    Recall:    0.8501
    F1-Score:  0.8512
```

**Inference Output**:
```
Image: tests/sample1.jpg
  Prediction: with_mask
  Confidence: 98.45%
  Probabilities:
    with_mask: 98.45%
    without_mask: 1.23%
    mask_weared_incorrect: 0.32%
```

## Model Download

Place the downloaded model in `models/checkpoints/best_model.pth`

## Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Proportion of correct positive predictions
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of classification performance

Detailed metrics and plots are saved in `results/metrics/`:
- `training_history.png`: Training/validation curves
- `confusion_matrix.png`: Confusion matrix
- `training_history.json`: Metrics history
- `config.json`: Training configuration

## Known Limitations

1. **Dataset Size**: The dataset contains ~900 faces, which is relatively small for deep learning. More data could improve generalization.

2. **Face Detection**: The current implementation assumes faces are already detected (using bounding boxes from annotations). For real-world deployment, integrate a face detection model (e.g., MTCNN, RetinaFace).

3. **Occlusion**: The model may struggle with heavily occluded faces or unusual angles.

4. **Real-time Performance**: Current implementation is optimized for accuracy, not speed. For real-time applications, consider:
   - Using lighter models (MobileNet, EfficientNet-B0)
   - Model quantization
   - TensorRT optimization

5. **Class Imbalance**: The dataset has some class imbalance. Techniques like class weights or oversampling could improve performance on minority classes.

## Next Steps

Potential improvements:
1. **Data Augmentation**: Add more sophisticated augmentation (CutOut, MixUp)
2. **Ensemble Methods**: Combine multiple models for better accuracy
3. **Face Detection Integration**: Add MTCNN/RetinaFace for end-to-end pipeline
4. **Model Optimization**: Quantization and pruning for deployment
5. **Web Interface**: Create Flask/FastAPI web service
6. **Real-time Video**: Implement video stream processing
7. **Multi-face Detection**: Handle multiple faces in single image

## Technical Requirements

**Hardware Used**:
- GPU: NVIDIA GTX 1080 Ti (or CPU if unavailable)
- RAM: 16GB
- Storage: 10GB

**Software**:
- Python 3.9
- PyTorch 2.1.0
- CUDA 11.8 (for GPU)

## References

- Dataset: [Face Mask Detection - Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- ResNet Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Transfer Learning: [Stanford CS231n Lecture Notes](http://cs231n.github.io/transfer-learning/)

## License

This project is for educational purposes as part of AI Engineer technical test.

## Contact

For questions or clarifications, contact:
- **Name**: Aldenia Alexandra
- **Email**: aldeniaalexandra@gmail.com
- **GitHub**: [aldeniaalexandra](https://github.com/aldeniaalexandra)

## Acknowledgments

- PT Verihubs Inteligensia Nusantara for the technical test opportunity
- Kaggle community for the dataset
- PyTorch team for the excellent framework