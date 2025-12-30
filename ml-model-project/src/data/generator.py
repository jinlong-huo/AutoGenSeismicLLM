import numpy as np
import pandas as pd
from typing import Dict, Tuple
import pywt

def create_ricker_wavelet(length: int, sigma: float = 1.0) -> np.ndarray:
    """Create a Ricker wavelet using PyWavelets or manual implementation"""
    try:
        # Try using PyWavelets if available
        wavelet = pywt.central_frequency('mexh') * sigma
        t = np.arange(-length//2, length//2 + 1)
        ricker = (2/(np.sqrt(3*sigma)*np.pi**(1/4))) * \
                (1 - (t/sigma)**2) * np.exp(-(t**2)/(2*sigma**2))
        return ricker
    except:
        # Manual Ricker wavelet implementation as fallback
        t = np.linspace(-length//2, length//2, length)
        sigma = sigma
        ricker = (2/(np.sqrt(3*sigma)*np.pi**(1/4))) * \
                (1 - (t/sigma)**2) * np.exp(-(t**2)/(2*sigma**2))
        return ricker

def generate_synthetic_seismic_traces(n_traces: int = 100, n_samples: int = 1000, 
                                    n_horizons: int = 3) -> np.ndarray:
    """Generate synthetic seismic traces with embedded horizons"""
    traces = np.zeros((n_traces, n_samples))
    
    # Create base horizons
    for horizon_idx in range(n_horizons):
        # Base depth for this horizon
        base_depth = 200 + horizon_idx * 250
        amplitude = 1.0 - horizon_idx * 0.2
        
        for trace_idx in range(n_traces):
            # Add structural variation (dip, faults, etc.)
            structural_variation = 50 * np.sin(trace_idx * 0.05) + \
                                 20 * np.cos(trace_idx * 0.1)
            
            # Add some random noise to horizon position
            noise = np.random.normal(0, 10)
            horizon_depth = int(base_depth + structural_variation + noise)
            
            # Ensure within bounds
            if 0 <= horizon_depth < n_samples:
                # Create a wavelet-like response using our custom Ricker wavelet
                wavelet = create_ricker_wavelet(21, 4.0)  # Replace deprecated signal.ricker
                start_idx = max(0, horizon_depth - 10)
                end_idx = min(n_samples, horizon_depth + 11)
                wavelet_section = wavelet[:end_idx - start_idx]
                
                traces[trace_idx, start_idx:end_idx] += amplitude * wavelet_section
    
    # Add background noise
    traces += np.random.normal(0, 0.1, traces.shape)
    
    return traces

def generate_horizon_labels(n_traces: int = 100, n_samples: int = 1000, 
                          n_horizons: int = 3) -> np.ndarray:
    """Generate ground truth horizon labels"""
    labels = np.zeros((n_traces, n_samples, n_horizons))
    
    for horizon_idx in range(n_horizons):
        base_depth = 200 + horizon_idx * 250
        
        for trace_idx in range(n_traces):
            structural_variation = 50 * np.sin(trace_idx * 0.05) + \
                                 20 * np.cos(trace_idx * 0.1)
            noise = np.random.normal(0, 10)
            horizon_depth = int(base_depth + structural_variation + noise)
            
            # Create a Gaussian-like label around the horizon
            if 0 <= horizon_depth < n_samples:
                for sample_idx in range(max(0, horizon_depth - 5), 
                                      min(n_samples, horizon_depth + 6)):
                    distance = abs(sample_idx - horizon_depth)
                    labels[trace_idx, sample_idx, horizon_idx] = np.exp(-distance**2 / 8)
    
    return labels

def generate_random_dataset(n_samples: int = 500, n_traces: int = 100, 
                          trace_length: int = 1000, n_horizons: int = 3) -> Dict:
    """Generate complete seismic dataset with features and labels"""
    
    print(f"Generating dataset with {n_samples} samples...")
    
    all_features = []
    all_labels = []
    
    for i in range(n_samples):
        if i % 50 == 0:
            print(f"Generated {i} samples...")
            
        # Generate seismic traces
        traces = generate_synthetic_seismic_traces(n_traces, trace_length, n_horizons)
        
        # Generate corresponding labels
        labels = generate_horizon_labels(n_traces, trace_length, n_horizons)
        
        all_features.append(traces)
        all_labels.append(labels)
    
    features = np.array(all_features)
    labels = np.array(all_labels)
    
    print(f"Dataset generated successfully!")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return {
        'features': features,
        'labels': labels,
        'metadata': {
            'n_samples': n_samples,
            'n_traces': n_traces,
            'trace_length': trace_length,
            'n_horizons': n_horizons,
            'sampling_rate': 4,  # ms
            'description': 'Synthetic seismic data with horizon labels'
        }
    }