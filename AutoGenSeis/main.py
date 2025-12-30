import torch
from models.dataset import SyntheticDataset
from agents.supervisor import SupervisorAgent
from agents.worker import WorkerAgent
import matplotlib.pyplot as plt

def visualize_results(original, prediction, title):
    """Visualize the original image and prediction"""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze(), cmap='gray')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(prediction.squeeze().detach().numpy(), cmap='gray')
    plt.title('Prediction')
    plt.suptitle(title)
    plt.show()

def main():
        # Train the models
    print("Training circle detection model...")
    circle_worker.train_model(circle_images, circle_masks, num_epochs=5)
    
    print("Training rectangle detection model...")
    rectangle_worker.train_model(rectangle_images, rectangle_masks, num_epochs=5)

    # Test the models with the supervisor's coordination
    print("\nTesting the models...")
    
    # Test circle detection
    test_circle = circle_images[0]  # Use first image as test
    circle_task = supervisor.coordinate_task("circle", test_circle)
    if circle_task.get("model") == "circle_unet":
        circle_results = circle_worker.process_task(test_circle)
        if circle_results["success"]:
            print("Circle detection successful")
            visualize_results(test_circle, circle_results["data"], "Circle Detection")
        else:
            print(f"Circle detection failed: {circle_results['error']}")

    # Test rectangle detection
    test_rectangle = rectangle_images[0]  # Use first image as test
    rectangle_task = supervisor.coordinate_task("rectangle", test_rectangle)
    if rectangle_task.get("model") == "rectangle_unet":
        rectangle_results = rectangle_worker.process_task(test_rectangle)
        if rectangle_results["success"]:
            print("Rectangle detection successful")
            visualize_results(test_rectangle, rectangle_results["data"], "Rectangle Detection")
        else:
            print(f"Rectangle detection failed: {rectangle_results['error']}")

if __name__ == "__main__":
    main() # Create dataset
    dataset = SyntheticDataset(size=128, num_samples=100)
    circle_images, circle_masks = dataset.generate_circle_data()
    rectangle_images, rectangle_masks = dataset.generate_rectangle_data()

    # Initialize agents
    supervisor = SupervisorAgent()
    circle_worker = WorkerAgent("circle")
    rectangle_worker = WorkerAgent("rectangle")
