import os
import json
import numpy as np
from src.data.generator import generate_random_dataset
from src.models.trainer import ModelTrainer
from src.agents.predictor import ModelPredictor
from src.utils.helpers import (
    plot_seismic_data, plot_training_history, 
    visualize_predictions, create_project_directories
)

def main():
    print("=== AutoGen Seismic ML Pipeline ===\n")
    
    # Create necessary directories
    create_project_directories()
    
    # Configuration
    config = {
        'data': {
            'n_samples': 200,  # Reduced for faster training
            'n_traces': 50,
            'trace_length': 500,
            'n_horizons': 3
        },
        'model': {
            'type': 'neural_network',  # Options: 'neural_network', 'random_forest', 'simple_cnn'
            'epochs': 30,
            'batch_size': 16
        }
    }
    
    print("Step 1: Generating synthetic seismic dataset...")
    dataset = generate_random_dataset(
        n_samples=config['data']['n_samples'],
        n_traces=config['data']['n_traces'],
        trace_length=config['data']['trace_length'],
        n_horizons=config['data']['n_horizons']
    )
    
    # Visualize sample data
    print("\nStep 2: Visualizing sample data...")
    plot_seismic_data(dataset['features'], "Sample Seismic Data", "plots/sample_data.png")
    
    print("\nStep 3: Training the model...")
    trainer = ModelTrainer(model_type=config['model']['type'])
    trainer.train_model(dataset)
    
    # Plot training history
    if trainer.training_history and 'loss' in trainer.training_history:
        plot_training_history(trainer.training_history, "plots/training_history.png")
    
    # Save model
    model_path = f"saved_models/{config['model']['type']}_model"
    if config['model']['type'] == 'random_forest':
        model_path += '.pkl'
    else:
        model_path += '.h5'
    
    trainer.save_model(model_path)
    
    print("\nStep 4: Loading model and making predictions...")
    predictor = ModelPredictor()
    predictor.load_model(model_path)
    
    # Use test data for predictions
    test_data = dataset['features'][:5]  # First 5 samples
    predictions = predictor.make_prediction(test_data)
    
    # Visualize predictions
    print("\nStep 5: Visualizing predictions...")
    visualize_predictions(test_data, predictions, sample_idx=0, save_path="plots/predictions.png")
    
    # Generate interpretation
    print("\nStep 6: Generating interpretation...")
    interpretation = predictor.predict_and_interpret(test_data, confidence_threshold=0.3)
    
    # Save results
    results = {
        'interpretation': json.loads(interpretation),
        'raw_predictions_shape': predictions.shape,
        'model_type': config['model']['type'],
        'dataset_info': dataset['metadata']
    }
    
    with open('predictions/interpretation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nStep 7: Results Summary")
    print("=" * 50)
    interp_data = json.loads(interpretation)
    print(f"Total horizons detected: {interp_data['total_horizons']}")
    print(f"High confidence horizons: {interp_data['high_confidence_horizons']}")
    print(f"Average confidence: {interp_data['average_confidence']:.3f}")
    print(f"Overall quality: {interp_data['quality']}")
    print("\nRecommendations:")
    for rec in interp_data['recommendations']:
        print(f"- {rec}")
    
    print(f"\nAll results saved to 'predictions/interpretation_results.json'")
    print("Pipeline completed successfully!")
    
    return predictor, results

# Function to integrate with AutoGen workflow
def create_ml_horizon_analyzer(model_path: str):
    """Create a function that can be used by AutoGen agents"""
    
    def analyze_horizons_ml(data_info: str, human_guidance: str = None) -> str:
        """ML-based horizon analysis function for AutoGen"""
        try:
            # Load the trained model
            predictor = ModelPredictor()
            predictor.load_model(model_path)
            
            # For demo, generate some test data
            # In real implementation, you'd load actual seismic data
            test_dataset = generate_random_dataset(n_samples=5, n_traces=50, trace_length=500)
            test_data = test_dataset['features']
            
            # Make predictions and interpret
            interpretation = predictor.predict_and_interpret(
                test_data, 
                confidence_threshold=0.4 if human_guidance and 'sensitive' in human_guidance.lower() else 0.5
            )
            
            return interpretation
            
        except Exception as e:
            # Fallback to basic analysis
            return json.dumps({
                "error": f"ML model failed: {str(e)}",
                "fallback": True,
                "horizons": [
                    {
                        "id": "H1_fallback",
                        "average_depth": 200,
                        "confidence": 0.7,
                        "interpretation": "Fallback horizon detection"
                    }
                ]
            })
    
    return analyze_horizons_ml

if __name__ == "__main__":
    predictor, results = main()
    
    # Demo the AutoGen integration
    print("\n" + "="*50)
    print("AUTOGEN INTEGRATION DEMO")
    print("="*50)
    
    model_path = f"saved_models/neural_network_model.h5"
    ml_analyzer = create_ml_horizon_analyzer(model_path)
    
    # Test the function as AutoGen would call it
    test_result = ml_analyzer("2D seismic data with 50 traces", "be sensitive to weak reflectors")
    print("\nAutoGen function result:")
    print(test_result)