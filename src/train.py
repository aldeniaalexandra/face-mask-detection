"""
This script handles the complete training pipeline including:
- Model training with validation
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Metrics logging
"""

import os
import argparse
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from dataset import create_data_loaders
from model import create_model
from utils import (
    calculate_metrics, 
    plot_training_history, 
    save_checkpoint,
    load_checkpoint,
    EarlyStopping
)


class Trainer:
    """
    Trainer class to encapsulate training logic
    
    Design Decision: Using a class-based approach makes the code more
    organized and allows easy state management across epochs.
    """
    
    def __init__(self, model, train_loader, val_loader, class_to_idx, 
                 criterion, optimizer, scheduler, device, config):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            class_to_idx: Dictionary mapping class names to indices
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on (cuda/cpu)
            config: Training configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.001)
        )
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Design Decision: We track both loss and accuracy during training
        to monitor model performance. Using tqdm provides good UX with
        progress bars and real-time metrics.
        """
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')
        
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            # Design Decision: Clipping at 1.0 helps with training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total_samples,
                'acc': running_corrects.double().item() / total_samples
            })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def validate(self, epoch):
        """
        Validate the model
        
        Design Decision: We use torch.no_grad() to disable gradient computation
        during validation, which saves memory and speeds up inference.
        """
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Val]  ')
        
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
                # Store predictions for detailed metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / total_samples,
                    'acc': running_corrects.double().item() / total_samples
                })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        # Calculate detailed metrics
        metrics = calculate_metrics(all_labels, all_preds, self.idx_to_class)
        
        return epoch_loss, epoch_acc.item(), metrics
    
    def train(self):
        """
        Complete training loop
        
        Design Decision: We implement:
        1. Learning rate scheduling based on validation loss
        2. Model checkpointing (save best model)
        3. Early stopping to prevent overfitting
        4. Comprehensive logging
        """
        print("\n" + "="*80)
        print(f"Starting training on {self.device}")
        print(f"Model: {self.config['model_name']}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print("="*80 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate(epoch)
            
            # Update learning rate based on validation loss
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"\n  Validation Metrics:")
            print(f"    Precision: {val_metrics['precision']:.4f}")
            print(f"    Recall:    {val_metrics['recall']:.4f}")
            print(f"    F1-Score:  {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                
                checkpoint_path = os.path.join(
                    self.config['checkpoint_dir'],
                    'best_model.pth'
                )
                
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_acc,
                    self.history,
                    checkpoint_path
                )
                
                print(f"  >>> Saved best model (val_acc: {val_acc:.4f})")
            
            # Save checkpoint every N epochs
            if (epoch + 1) % self.config.get('save_frequency', 10) == 0:
                checkpoint_path = os.path.join(
                    self.config['checkpoint_dir'],
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_acc,
                    self.history,
                    checkpoint_path
                )
            
            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
                break
            
            print("-" * 80)
        
        # Training completed
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        
        # Plot training history
        plot_path = os.path.join(self.config['results_dir'], 'training_history.png')
        plot_training_history(self.history, save_path=plot_path)
        
        # Save training history
        history_path = os.path.join(self.config['results_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        return self.history


def main():
    """
    Main training function
    
    Design Decision: Using argparse allows flexible configuration via command line
    while maintaining sensible defaults. This is crucial for experimentation.
    """
    parser = argparse.ArgumentParser(description='Train Face Mask Detection Model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/raw',
                        help='Directory containing images and annotations')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet'],
                        help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.001,
                        dest='learning_rate',
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'],
                        help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--results-dir', type=str, default='results/metrics',
                        help='Directory to save results')
    parser.add_argument('--save-frequency', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, val_loader, class_to_idx = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    num_classes = len(class_to_idx)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(class_to_idx.keys())}")
    
    # Create model
    print(f"\nCreating model: {args.model_name}")
    model = create_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=args.pretrained
    )
    model = model.to(device)
    
    # Loss function with class weights for imbalanced dataset
    # Design Decision: Using weighted loss helps with class imbalance
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    # Design Decision: Adam optimizer with weight decay (AdamW behavior)
    # Works well for most computer vision tasks
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
    
    # Training configuration
    config = {
        'model_name': args.model_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'patience': args.patience,
        'checkpoint_dir': args.checkpoint_dir,
        'results_dir': args.results_dir,
        'save_frequency': args.save_frequency,
        'device': str(device),
        'seed': args.seed
    }
    
    # Save config
    config_path = os.path.join(args.results_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Create trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_to_idx=class_to_idx,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    history = trainer.train()
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()