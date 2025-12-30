# AutoGen Seismic LLM

An AI-powered seismic data interpretation system using Microsoft AutoGen for multi-agent workflows. This project demonstrates the application of Large Language Models (LLMs) to geoscience data processing, specifically seismic interpretation tasks including horizon picking, fault detection, and visualization.

## 🌟 Features

- **Multi-Agent Architecture**: Uses AutoGen to coordinate specialized AI agents for different interpretation tasks
- **Automated Seismic Interpretation Pipeline**:
  - Synthetic seismic data generation
  - Data cleaning and preprocessing  
  - Horizon picking and extraction
  - Fault detection and analysis
  - Comprehensive visualization
- **Flexible Execution Modes**: Run fully automated or with LLM-coordinated agents
- **Interactive Workflow**: Alternative script with human-in-the-loop capabilities

## 📁 Project Structure

```
AutoGenSeismicLLM/
├── AutoGenSeic.py          # Main workflow script (DAG-based execution)
├── Seis_Ag_Wf.py           # Interactive interpretation with human guidance
├── .env.example            # Example environment configuration
├── LICENSE                 # MIT License
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/AutoGenSeismicLLM.git
   cd AutoGenSeismicLLM
   ```

2. **Install dependencies**:
   ```bash
   pip install pyautogen numpy matplotlib scipy networkx
   ```

3. **Configure API key** (for LLM mode):
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```

   Supported LLM providers:
   - **Alibaba Cloud DashScope** (Qwen models) - Default
   - **DeepSeek**
   - Any OpenAI-compatible API

### Usage

#### Main Workflow Script (`AutoGenSeic.py`)

Run the complete seismic interpretation workflow:

```bash
python AutoGenSeic.py
```

This will:
1. Generate synthetic seismic data (or load existing)
2. Clean and preprocess the data
3. Pick horizons using amplitude thresholding
4. Detect faults based on horizon discontinuities
5. Create visualizations

Output files are saved to `workspace/data/`:
- `raw/` - Raw seismic volumes
- `processed/` - Cleaned data, horizons, faults
- `visualizations/` - Generated plots and figures

#### Interactive Script (`Seis_Ag_Wf.py`)

For an interactive interpretation experience with human guidance:

```bash
python Seis_Ag_Wf.py
```

Menu options:
1. **Run Interactive Interpretation** - Guide agents through the workflow
2. **Test Individual Functions** - Test specific analysis functions
3. **View Instructions** - See available commands
4. **Exit**

Interactive commands during workflow:
- `approve` - Accept current interpretation
- `modify` - Request changes
- `focus on [area]` - Direct attention to specific features
- `skip` - Skip current analysis
- `exit` - End workflow

## 🏗️ Architecture

### Agents

| Agent | Role |
|-------|------|
| **WorkflowManager / LeadGeophysicist** | Coordinates overall workflow execution |
| **DataProcessor** | Handles data loading and preprocessing |
| **HorizonInterpreter** | Specializes in horizon identification |
| **FaultInterpreter** | Detects and analyzes fault structures |
| **VisualizationExpert** | Creates visual representations |
| **AttributeAnalyst** | Computes seismic attributes (interactive mode) |

### Workflow DAG

```
load_data → clean_data → pick_horizons → detect_faults
                                      ↘ visualize_horizons
                                           ↘ create_final_visualization
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with:

```bash
QWEN_API_KEY="your-api-key-here"
QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

| Variable | Description | Default |
|----------|-------------|---------|
| `QWEN_API_KEY` | API key for Qwen/DashScope | Required for LLM mode |
| `QWEN_BASE_URL` | API endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

### Supported Models

- `qwen-plus` (Alibaba Cloud DashScope)
- `deepseek-chat` (DeepSeek)
- Any OpenAI-compatible model

## 📊 Output Examples

The workflow generates:
- **Seismic slices** with interpreted horizons overlaid
- **Fault likelihood maps** showing structural discontinuities
- **Horizon surface visualizations** in 2D and 3D
- **Comprehensive multi-panel visualizations** combining all interpretations

## 🔧 Customization

### Using Your Own Seismic Data

Modify `SeismicDataGenerator.generate_seismic_volume()` or replace with your data loading function:

```python
# Load your own data instead of generating synthetic
volume = np.load("your_seismic_data.npy")
```

### Adjusting Interpretation Parameters

| Parameter | Location | Description |
|-----------|----------|-------------|
| `noise_threshold` | `extract_horizons()` | Amplitude threshold for horizon detection |
| `shape` | `generate_seismic_volume()` | Dimensions of synthetic data |
| `sigma` | `execute_data_cleaning()` | Gaussian smoothing parameter |
| `depth_tolerance` | `extract_horizons()` | Horizon grouping tolerance |

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 References

- [Microsoft AutoGen](https://github.com/microsoft/autogen) - Multi-agent conversation framework
- [Alibaba Cloud DashScope](https://dashscope.aliyuncs.com/) - Qwen LLM API

## 🙏 Acknowledgments

- Microsoft AutoGen team for the multi-agent framework
- Alibaba Cloud for Qwen model access
