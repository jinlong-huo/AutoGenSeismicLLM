import autogen
import torch
from models.unet import create_unet

class WorkerAgent:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = create_unet(in_channels=1, out_channels=1)
        self.agent = autogen.AssistantAgent(
            name=f"worker_{model_type}",
            system_message=f"""You are a worker agent responsible for running the {model_type} UNet model.
            Your role is to:
            1. Receive data from the supervisor
            2. Process the segmentation task
            3. Return the results
            4. Handle any errors that occur""",
            llm_config={
                "config_list": [{"model": "gpt-3.5-turbo", "api_key": "YOUR_API_KEY"}]
            }
        )

    def process_task(self, data):
        """Process the segmentation task"""
        try:
            # Ensure model is in evaluation mode
            self.model.eval()
            
            # Process data through the model
            with torch.no_grad():
                output = self.model(data)
            
            return {
                "success": True,
                "data": output,
                "message": f"Successfully processed {self.model_type} segmentation"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error processing {self.model_type} segmentation"
            }

    def train_model(self, train_data, train_masks, num_epochs=10):
        """Train the UNet model"""
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = torch.nn.BCELoss()
        
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data, mask in zip(train_data, train_masks):
                optimizer.zero_grad()
                output = self.model(data.unsqueeze(0))
                loss = criterion(output, mask.unsqueeze(0))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_data):.4f}")