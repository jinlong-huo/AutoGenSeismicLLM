# AutoGen UNet Segmentation Project

This project demonstrates the use of AutoGen to coordinate multiple UNet models for different segmentation tasks. It includes a supervisor agent that coordinates two worker agents, each specialized in different shape detection tasks (circles and rectangles).

## Project Structure
```
project/
├── requirements.txt
├── README.md
├── models/
│   ├── __init__.py
│   ├── unet.py
│   └── utils.py
├── agents/
│   ├── __init__.py
│   ├── supervisor.py
│   └── worker.py
├── data/
│   └── dataset.py
└── main.py
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Configure your OpenAI API key:
- Open `agents/supervisor.py` and `agents/worker.py`
- Replace `"YOUR_API_KEY"` with your actual OpenAI API key

## Running the Demo

To run the demo:
```bash
python main.py
```

This will:
1. Generate synthetic training data (circles and rectangles)
2. Train two UNet models for different segmentation tasks
3. Test the models with example images
4. Display the results using matplotlib

## Components

- **SupervisorAgent**: Coordinates the segmentation tasks between models
- **WorkerAgent**: Handles the actual segmentation using UNet models
- **SyntheticDataset**: Generates training data with circles and rectangles
- **UNet**: Implementation of the UNet architecture for segmentation

## Notes

- The project uses synthetic data for demonstration purposes
- Each UNet model is trained for 5 epochs by default
- The supervisor agent uses GPT-3.5-turbo for task coordination
- Results are visualized using matplotlib