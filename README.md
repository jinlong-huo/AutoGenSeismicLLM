# AutoGen Seismic LLM

An AI-powered seismic data interpretation system using Microsoft AutoGen for multi-agent workflows. This project applies Large Language Models (LLMs) to geoscience data processing—horizon picking, fault detection, and visualization.

## Features

- Multi-agent architecture using AutoGen
- Automated seismic interpretation pipeline (data loading → cleaning → horizon picking → fault detection → visualization)
- Flexible execution: fully automated or LLM-coordinated
- Interactive mode with human-in-the-loop

## Getting Started

### Installation

```bash
git clone https://github.com/yourusername/AutoGenSeismicLLM.git
cd AutoGenSeismicLLM
pip install pyautogen numpy matplotlib scipy networkx
```

### Configuration

Copy `.env.example` to `.env` and add your API key:

```bash
QWEN_API_KEY="your-api-key-here"
QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

Supports: Alibaba Cloud DashScope (Qwen), DeepSeek, or any OpenAI-compatible API.

### Run

```bash
python AutoGenSeic.py      # Main workflow
python Seis_Ag_Wf.py       # Interactive mode
```

## Architecture

### Agents

| Agent | Role |
|-------|------|
| WorkflowManager | Coordinates workflow execution |
| DataProcessor | Data loading and preprocessing |
| HorizonInterpreter | Horizon identification |
| FaultInterpreter | Fault detection and analysis |
| VisualizationExpert | Visual representations |

### Workflow

```
load_data → clean_data → pick_horizons → detect_faults → visualize
```

## License

MIT License - see [LICENSE](LICENSE)

## References

- [Microsoft AutoGen](https://github.com/microsoft/autogen)
- [Alibaba Cloud DashScope](https://dashscope.aliyuncs.com/)
