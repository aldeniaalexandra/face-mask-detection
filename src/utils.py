"""
This module contains helper functions for:
- Metrics calculation
- Visualization
- Model checkpointing
- Early stopping
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Dictionary mapping indices to class names
    
    Returns:
        Dictionary containing all metrics
    
    Design Decision: We use macro averaging for multi-class metrics
    to give equal weight to each class, which is important for
    imbalanced datasets.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    # Per-class metrics
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=[class_names[i] for i in sorted(class_names.keys())],
        output_dict=True,
        zero_division=0
    )
    metrics['per_class'] = report
    
    return metrics


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Design Decision: Using seaborn's heatmap provides clear visualization
    of classification performance across classes.
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=[class_names[i] for i in sorted(class_names.keys())],
        yticklabels=[class_names[i] for i in sorted(class_names.keys())],
        cbar_kws={'label': 'Percentage'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics over epochs
    
    Design Decision: Visualizing loss and accuracy curves helps identify
    overfitting and training issues.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-s', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot learning rate
    axes[1, 0].plot(epochs, history['learning_rates'], 'g-^', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot accuracy gap (overfitting indicator)
    acc_gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
    axes[1, 1].plot(epochs, acc_gap, 'purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy Gap', fontsize=12)
    axes[1, 1].set_title('Train-Val Accuracy Gap (Overfitting Indicator)', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, history, save_path):
    """
    Save model checkpoint
    
    Design Decision: Saving complete state allows resuming training
    from any checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc,
        'history': history
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_path=None):
    """
    Load model checkpoint
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer (optional, for resuming training)
        scheduler: Scheduler (optional, for resuming training)
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Tuple of (epoch, val_acc, history)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    val_acc = checkpoint.get('val_acc', 0.0)
    history = checkpoint.get('history', None)
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {epoch}, Val Acc: {val_acc:.4f}")
    
    return epoch, val_acc, history


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving
    
    Design Decision: Early stopping prevents wasting computational resources
    and helps avoid overfitting.
    """
    
    def __init__(self, patience=10, min_delta=0.001):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        """
        Check if training should stop
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def visualize_predictions(images, labels, predictions, class_names, 
                         num_samples=16, save_path=None):
    """
    Visualize model predictions on sample images
    
    Design Decision: Visual inspection helps identify failure cases
    and understand model behavior.
    """
    num_samples = min(num_samples, len(images))
    
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Convert tensor to numpy and denormalize
        img = images[idx].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Get labels
        true_label = class_names[labels[idx].item()]
        pred_label = class_names[predictions[idx].item()]
        
        # Determine color (green if correct, red if wrong)
        color = 'green' if labels[idx] == predictions[idx] else 'red'
        
        # Plot
        ax.imshow(img)
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', 
                     color=color, fontsize=10)
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def count_parameters(model):
    """
    Count total and trainable parameters in model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_summary(model):
    """
    Print model architecture summary
    """
    params = count_parameters(model)
    
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(f"Total Parameters:       {params['total']:,}")
    print(f"Trainable Parameters:   {params['trainable']:,}")
    print(f"Non-trainable Parameters: {params['non_trainable']:,}")
    print("="*80 + "\n")