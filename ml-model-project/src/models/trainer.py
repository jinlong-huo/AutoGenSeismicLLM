import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, model_type: str = 'neural_network'):
        """
        Initialize the model trainer
        
        Args:
            model_type: 'neural_network', 'random_forest', or 'simple_cnn'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = {}
        
    def prepare_data(self, dataset: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        features = dataset['features']
        labels = dataset['labels']
        
        print(f"Original features shape: {features.shape}")
        print(f"Original labels shape: {labels.shape}")
        
        if self.model_type == 'simple_cnn':
            # For CNN, keep the 2D structure
            # Reshape to (n_samples, n_traces, trace_length, 1) for CNN
            X = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
            # For labels, take the maximum along horizons dimension to create single-channel output
            y = np.max(labels, axis=-1)  # Shape: (n_samples, n_traces, trace_length)
            y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
        else:
            # For other models, flatten the features
            X = features.reshape(features.shape[0], -1)  # Flatten traces
            y = labels.reshape(labels.shape[0], -1)      # Flatten labels
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features for non-CNN models
        if self.model_type != 'simple_cnn':
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        else:
            # For CNN, normalize to [0, 1]
            X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
            X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
        
        print(f"Prepared X_train shape: {X_train.shape}")
        print(f"Prepared y_train shape: {y_train.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def create_neural_network(self, input_shape: int, output_shape: int) -> keras.Model:
        """Create a simple neural network"""
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(input_shape,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(output_shape, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_simple_cnn(self, input_shape: Tuple) -> keras.Model:
        """Create a simple CNN for 2D seismic data"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape[1:]),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Upsampling to match input size
            keras.layers.UpSampling2D((2, 2)),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.UpSampling2D((2, 2)),
            keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, dataset: Dict) -> None:
        """Train the model on the dataset"""
        print(f"Training {self.model_type} model...")
        
        X_train, X_test, y_train, y_test = self.prepare_data(dataset)
        
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            print("Training Random Forest...")
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            self.training_history = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
            
        elif self.model_type == 'neural_network':
            self.model = self.create_neural_network(X_train.shape[1], y_train.shape[1])
            
            print("Training Neural Network...")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                verbose=1
            )
            
            self.training_history = history.history
            
        elif self.model_type == 'simple_cnn':
            self.model = self.create_simple_cnn(X_train.shape)
            
            print("Training CNN...")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=30,
                batch_size=16,
                verbose=1
            )
            
            self.training_history = history.history
        
        print("Training completed!")
        self.print_training_summary()
    
    def print_training_summary(self):
        """Print training summary"""
        print("\n=== Training Summary ===")
        if self.model_type == 'random_forest':
            print(f"Train MSE: {self.training_history['train_mse']:.4f}")
            print(f"Test MSE: {self.training_history['test_mse']:.4f}")
            print(f"Train R²: {self.training_history['train_r2']:.4f}")
            print(f"Test R²: {self.training_history['test_r2']:.4f}")
        else:
            final_loss = self.training_history['loss'][-1]
            final_val_loss = self.training_history['val_loss'][-1]
            print(f"Final Training Loss: {final_loss:.4f}")
            print(f"Final Validation Loss: {final_val_loss:.4f}")
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.model_type == 'random_forest':
            # Save sklearn model and scaler
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'training_history': self.training_history
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
        else:
            # Save Keras model
            self.model.save(filepath.replace('.pkl', '.h5'))
            # Save scaler and metadata separately
            metadata = {
                'scaler': self.scaler,
                'model_type': self.model_type,
                'training_history': self.training_history
            }
            with open(filepath.replace('.pkl', '_metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
        
        print(f"Model saved to {filepath}")