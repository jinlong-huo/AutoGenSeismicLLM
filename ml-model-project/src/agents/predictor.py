import numpy as np
import pickle
import json
import os
from typing import Dict, List, Tuple, Union
import tensorflow as tf
from tensorflow import keras

class ModelPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_type = None
        self.metadata = {}
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model"""
        print(f"Loading model from {filepath}")
        
        if filepath.endswith('.pkl'):
            # Load sklearn model
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.metadata = model_data.get('training_history', {})
            
        elif filepath.endswith('.h5'):
            # Load Keras model
            self.model = keras.models.load_model(filepath)
            
            # Load metadata
            metadata_path = filepath.replace('.h5', '_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.scaler = metadata['scaler']
                self.model_type = metadata['model_type']
                self.metadata = metadata.get('training_history', {})
        
        print(f"Model loaded successfully! Type: {self.model_type}")
    
    def preprocess_input(self, input_data: np.ndarray) -> np.ndarray:
        """Preprocess input data for prediction"""
        if self.model_type == 'simple_cnn':
            # For CNN, reshape and normalize
            if len(input_data.shape) == 3:  # (n_samples, n_traces, trace_length)
                processed_data = input_data.reshape(
                    input_data.shape[0], input_data.shape[1], input_data.shape[2], 1
                )
            else:
                processed_data = input_data
            
            # Normalize
            processed_data = (processed_data - processed_data.min()) / \
                           (processed_data.max() - processed_data.min())
        else:
            # For other models, flatten and scale
            if len(input_data.shape) > 2:
                processed_data = input_data.reshape(input_data.shape[0], -1)
            else:
                processed_data = input_data
            
            if self.scaler is not None:
                processed_data = self.scaler.transform(processed_data)
        
        return processed_data
    
    def make_prediction(self, input_data: np.ndarray) -> np.ndarray:
        """Make predictions on input data"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"Making predictions on data shape: {input_data.shape}")
        
        # Preprocess input
        processed_data = self.preprocess_input(input_data)
        
        # Make prediction
        predictions = self.model.predict(processed_data)
        
        print(f"Predictions shape: {predictions.shape}")
        
        return predictions
    
    def extract_horizons_from_predictions(self, predictions: np.ndarray, 
                                        confidence_threshold: float = 0.5) -> List[Dict]:
        """Extract horizon information from model predictions"""
        horizons = []
        
        # Handle different prediction formats
        if len(predictions.shape) == 4:  # CNN output (n_samples, n_traces, trace_length, 1)
            predictions = predictions.squeeze(-1)  # Remove last dimension
        elif len(predictions.shape) == 2:  # Flattened output, need to reshape
            # Assume square-ish reshaping for now
            sample_size = int(np.sqrt(predictions.shape[1]))
            predictions = predictions.reshape(predictions.shape[0], sample_size, -1)
        
        for sample_idx in range(predictions.shape[0]):
            sample_predictions = predictions[sample_idx]
            
            # Find peaks in the predictions (potential horizons)
            from scipy.signal import find_peaks
            
            for trace_idx in range(sample_predictions.shape[0]):
                trace_pred = sample_predictions[trace_idx]
                
                # Find peaks above threshold
                peaks, properties = find_peaks(
                    trace_pred, 
                    height=confidence_threshold,
                    distance=20  # Minimum distance between horizons
                )
                
                for peak_idx, peak_pos in enumerate(peaks):
                    confidence = trace_pred[peak_pos]
                    
                    horizon = {
                        "id": f"H{sample_idx}_{trace_idx}_{peak_idx}",
                        "sample_id": sample_idx,
                        "trace_id": trace_idx,
                        "depth_sample": int(peak_pos),
                        "depth_time_ms": int(peak_pos * 4),  # Assuming 4ms sampling
                        "confidence": float(confidence),
                        "continuity": "good" if confidence > 0.8 else "moderate",
                        "interpretation": f"ML-detected horizon with {confidence:.2f} confidence",
                        "model_predicted": True
                    }
                    
                    horizons.append(horizon)
        
        # Sort by confidence
        horizons.sort(key=lambda x: x['confidence'], reverse=True)
        
        return horizons
    
    def predict_and_interpret(self, input_data: np.ndarray, 
                            confidence_threshold: float = 0.5) -> str:
        """Make predictions and return formatted interpretation results"""
        
        # Make predictions
        predictions = self.make_prediction(input_data)
        
        # Extract horizons
        horizons = self.extract_horizons_from_predictions(predictions, confidence_threshold)
        
        # Create summary statistics
        total_horizons = len(horizons)
        high_confidence_horizons = len([h for h in horizons if h['confidence'] > 0.8])
        avg_confidence = np.mean([h['confidence'] for h in horizons]) if horizons else 0
        
        # Format results
        results = {
            "horizons": horizons[:20],  # Limit to top 20 for readability
            "total_horizons": total_horizons,
            "high_confidence_horizons": high_confidence_horizons,
            "average_confidence": float(avg_confidence),
            "quality": "excellent" if avg_confidence > 0.8 else "good" if avg_confidence > 0.6 else "moderate",
            "model_info": {
                "model_type": self.model_type,
                "confidence_threshold": confidence_threshold,
                "prediction_method": "ML_based"
            },
            "recommendations": self._generate_recommendations(horizons)
        }
        
        return json.dumps(results, indent=2)
    
    def _generate_recommendations(self, horizons: List[Dict]) -> List[str]:
        """Generate interpretation recommendations based on detected horizons"""
        recommendations = []
        
        if not horizons:
            recommendations.append("No horizons detected above threshold - consider lowering confidence threshold")
            return recommendations
        
        avg_confidence = np.mean([h['confidence'] for h in horizons])
        
        if avg_confidence > 0.8:
            recommendations.append("High-quality horizon detection - proceed with structural interpretation")
        elif avg_confidence > 0.6:
            recommendations.append("Moderate-quality detection - validate with additional seismic attributes")
        else:
            recommendations.append("Low-confidence detection - consider data reprocessing or parameter tuning")
        
        # Check for clustering
        depths = [h['depth_sample'] for h in horizons]
        if len(set(depths)) < len(depths) * 0.7:  # If many horizons at similar depths
            recommendations.append("Horizon clustering detected - check for multiples or processing artifacts")
        
        recommendations.append("Consider correlating with well data if available")
        recommendations.append("Run additional seismic attributes for validation")
        
        return recommendations

    def save_predictions(self, predictions: np.ndarray, filepath: str, 
                        metadata: Dict = None) -> None:
        """Save predictions to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        results = {
            'predictions': predictions.tolist(),
            'shape': predictions.shape,
            'model_type': self.model_type,
            'metadata': metadata or {}
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Predictions saved to {filepath}")