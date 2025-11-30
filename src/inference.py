"""
This script performs inference on new images using a trained model.
Supports both single image and batch (folder) inference.

Usage:
    # Single image
    python src/inference.py --image path/to/image.jpg --model models/best_model.pth
    
    # Folder of images
    python src/inference.py --folder path/to/images/ --model models/best_model.pth
    
    # With visualization
    python src/inference.py --image test.jpg --model models/best_model.pth --visualize
"""

import os
import argparse
import json
from pathlib import Path

import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import create_model
from dataset import get_transforms


class FaceMaskPredictor:
    """
    Predictor class for face mask detection
    
    Design Decision: Encapsulating inference logic in a class makes it
    reusable and easier to integrate into applications.
    """
    
    def __init__(self, model_path, device='cuda', confidence_threshold=0.5):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            confidence_threshold: Minimum confidence for prediction
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model configuration (if available)
        # Design Decision: Try to infer model config from checkpoint
        # Default to resnet50 if not found
        self.model = create_model('resnet50', num_classes=3, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Class names
        self.class_names = {
            0: 'with_mask',
            1: 'without_mask',
            2: 'mask_weared_incorrect'
        }
        
        # Transform
        self.transform = get_transforms('val')
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Classes: {list(self.class_names.values())}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model input
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed tensor and original image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        
        # Resize to model input size
        image = cv2.resize(image, (224, 224))
        
        # Apply transform
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor, original_image
    
    def predict(self, image_path, return_probs=True):
        """
        Predict mask class for an image
        
        Args:
            image_path: Path to image file
            return_probs: Whether to return class probabilities
        
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess
        image_tensor, original_image = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get prediction
        pred_class = predicted.item()
        pred_label = self.class_names[pred_class]
        pred_confidence = confidence.item()
        
        result = {
            'image_path': image_path,
            'predicted_class': pred_label,
            'confidence': pred_confidence,
            'predicted_index': pred_class
        }
        
        if return_probs:
            result['probabilities'] = {
                self.class_names[i]: probabilities[0][i].item()
                for i in range(len(self.class_names))
            }
        
        return result
    
    def predict_batch(self, image_paths, show_progress=True):
        """
        Predict mask class for multiple images
        
        Args:
            image_paths: List of image paths
            show_progress: Whether to show progress bar
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        iterator = tqdm(image_paths, desc="Processing images") if show_progress else image_paths
        
        for image_path in iterator:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction on image
        
        Design Decision: Visual feedback helps verify model performance
        and is useful for presentations/reports.
        """
        # Get prediction
        result = self.predict(image_path)
        
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot image
        ax.imshow(image)
        
        # Add prediction text
        pred_text = f"Prediction: {result['predicted_class']}"
        conf_text = f"Confidence: {result['confidence']:.2%}"
        
        # Color based on prediction
        color_map = {
            'with_mask': 'green',
            'without_mask': 'red',
            'mask_weared_incorrect': 'orange'
        }
        text_color = color_map.get(result['predicted_class'], 'black')
        
        # Add text box
        textstr = f"{pred_text}\n{conf_text}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=text_color, linewidth=3)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props, color=text_color, weight='bold')
        
        # Add probability bar
        if 'probabilities' in result:
            prob_text = "Class Probabilities:\n"
            for class_name, prob in result['probabilities'].items():
                prob_text += f"  {class_name}: {prob:.2%}\n"
            
            props2 = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.02, 0.02, prob_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='bottom', bbox=props2, family='monospace')
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result


def main():
    """
    Main inference function
    """
    parser = argparse.ArgumentParser(
        description='Face Mask Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python src/inference.py --image test.jpg --model models/best_model.pth
  
  # Batch inference on folder
  python src/inference.py --folder tests/sample_images/ --model models/best_model.pth
  
  # With visualization
  python src/inference.py --image test.jpg --model models/best_model.pth --visualize
  
  # Save predictions to JSON
  python src/inference.py --folder tests/ --model models/best_model.pth --output predictions.json
        """
    )
    
    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to single image')
    group.add_argument('--folder', type=str, help='Path to folder containing images')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    
    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions (JSON format)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--save-viz', type=str, default=None,
                        help='Directory to save visualization images')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    parser.add_argument('--confidence-threshold', type=float, default=0.0,
                        help='Minimum confidence threshold for predictions')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = FaceMaskPredictor(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )
    
    # Get image paths
    if args.image:
        image_paths = [args.image]
    else:
        # Get all images from folder
        folder = Path(args.folder)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            str(p) for p in folder.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            print(f"No images found in {args.folder}")
            return
        
        print(f"Found {len(image_paths)} images in {args.folder}")
    
    # Run inference
    print("\nRunning inference...")
    results = predictor.predict_batch(image_paths)
    
    # Print results
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)
    
    for result in results:
        if 'error' in result:
            print(f"\nImage: {result['image_path']}")
            print(f"  Error: {result['error']}")
        else:
            print(f"\nImage: {result['image_path']}")
            print(f"  Prediction: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            
            if 'probabilities' in result:
                print("  Probabilities:")
                for class_name, prob in result['probabilities'].items():
                    print(f"    {class_name}: {prob:.2%}")
    
    print("="*80)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {args.output}")
    
    # Visualize predictions
    if args.visualize or args.save_viz:
        print("\nGenerating visualizations...")
        
        if args.save_viz:
            os.makedirs(args.save_viz, exist_ok=True)
        
        for i, (image_path, result) in enumerate(zip(image_paths, results)):
            if 'error' in result:
                continue
            
            if args.save_viz:
                save_path = os.path.join(
                    args.save_viz,
                    f"prediction_{i}_{Path(image_path).stem}.png"
                )
                predictor.visualize_prediction(image_path, save_path=save_path)
            elif args.visualize and i < 5:  # Show first 5 only
                predictor.visualize_prediction(image_path)
    
    print("\nInference completed successfully!")


if __name__ == '__main__':
    main()