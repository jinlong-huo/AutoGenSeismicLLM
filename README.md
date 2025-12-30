# AutoGen Seismic LLM

AI-powered seismic data interpretation using Microsoft AutoGen multi-agent workflows. Applies LLMs to horizon picking, fault detection, and visualization.

## Features

- Multi-agent architecture (AutoGen)
- Three execution modes: Automated, Human-in-the-Loop, LLM-Coordinated
- Seismic interpretation pipeline: load → clean → horizon picking → fault detection → visualization

## Installation

```bash
pip install pyautogen numpy matplotlib scipy networkx
```

## Configuration

Set your API key (for LLM mode):

```bash
export QWEN_API_KEY="your-api-key"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

## Run

```bash
python AutoGenSeic.py
```

Menu options:
1. **Fully Automated** - Runs all tasks without interaction
2. **Human-in-the-Loop** - Review and approve each task
3. **LLM-Coordinated** - AI agents coordinate the workflow
4. **Show workflow info**
5. **Exit**

## Architecture

| Agent | Role |
|-------|------|
| WorkflowManager | Coordinates execution |
| DataProcessor | Data loading/preprocessing |
| HorizonExpert | Horizon identification |
| FaultExpert | Fault detection |
| VisualizationExpert | Visualizations |

```
load_data → clean_data → pick_horizons → detect_faults → visualize
```

## License

MIT - see [LICENSE](LICENSE)

## References

- [Microsoft AutoGen](https://github.com/microsoft/autogen)
- [Alibaba Cloud DashScope](https://dashscope.aliyuncs.com/)
