import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def normalize_data(data):
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    return (data - min_val) / (max_val - min_val)

def calculate_performance_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def plot_seismic_data(data: np.ndarray, title: str = "Seismic Data", 
                     save_path: str = None) -> None:
    """Plot seismic data"""
    plt.figure(figsize=(12, 8))
    
    if len(data.shape) == 3:  # Multiple samples
        data = data[0]  # Show first sample
    
    plt.imshow(data.T, cmap='seismic', aspect='auto', 
               extent=[0, data.shape[0], data.shape[1], 0])
    plt.colorbar(label='Amplitude')
    plt.xlabel('Trace Number')
    plt.ylabel('Time Sample')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(history: Dict, save_path: str = None) -> None:
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if 'loss' in history:
        axes[0].plot(history['loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
    
    # Plot metrics
    if 'mae' in history:
        axes[1].plot(history['mae'], label='Training MAE')
        axes[1].plot(history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_predictions(original_data: np.ndarray, predictions: np.ndarray, 
                         sample_idx: int = 0, save_path: str = None) -> None:
    """Visualize predictions overlaid on original data"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original data
    axes[0].imshow(original_data[sample_idx].T, cmap='seismic', aspect='auto')
    axes[0].set_title('Original Seismic Data')
    axes[0].set_xlabel('Trace Number')
    axes[0].set_ylabel('Time Sample')
    
    # Predictions
    if len(predictions.shape) == 4:  # CNN output
        pred_to_show = predictions[sample_idx, :, :, 0]
    else:
        # Reshape flattened predictions
        sample_size = int(np.sqrt(predictions.shape[1]))
        pred_reshaped = predictions.reshape(predictions.shape[0], sample_size, -1)
        pred_to_show = pred_reshaped[sample_idx]
    
    axes[1].imshow(pred_to_show.T, cmap='hot', aspect='auto')
    axes[1].set_title('Model Predictions')
    axes[1].set_xlabel('Trace Number')
    axes[1].set_ylabel('Time Sample')
    
    # Overlay
    axes[2].imshow(original_data[sample_idx].T, cmap='seismic', aspect='auto', alpha=0.7)
    axes[2].imshow(pred_to_show.T, cmap='hot', aspect='auto', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].set_xlabel('Trace Number')
    axes[2].set_ylabel('Time Sample')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_project_directories():
    """Create necessary project directories"""
    dirs = ['saved_models', 'predictions', 'plots', 'data']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    print("Project directories created.")

def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except:
        # Return default config if file doesn't exist
        return {
            'model': {
                'type': 'neural_network',
                'epochs': 50,
                'batch_size': 32
            },
            'data': {
                'n_samples': 500,
                'n_traces': 100,
                'trace_length': 1000,
                'n_horizons': 3
            }
        }