# StepWise OSWorld Evaluation

StepWise is the Green Agent that benchmarks multimodal White Agents on OSWorld’s executable desktop tasks [@https://os-world.github.io/]. It orchestrates Docker-based VM snapshots, streams observations to agents, executes returned actions, runs OSWorld evaluation scripts, and logs metrics for reproducible comparison.

## Quick Start

```bash
# Setup OSWorld repository, virtualenv, and monitor
./setup_osworld.sh

# Activate environment
source env_osworld/bin/activate

# Configure API key
echo "OPENAI_API_KEY=sk-..." >> .env

# Run StepWise over a subset of tasks (Docker provider)
python -m berkeley-agentic-ai --mode docker --tasks osworld/evaluation_examples/test_small.json --limit 2
```

## Requirements

- Docker Desktop running with virtualization support
- Python 3.11+
- Valid OpenAI API key (GPT-5 preview access)
- macOS/Linux host with 16 GB+ RAM recommended

## Project Layout

- `stepwise.py` — StepWise Green Agent implementation with StepWise responsibilities
- `run_osworld_docker.py` — CLI launcher that boots StepWise against Docker VMs
- `osworld/` — Submodule containing official OSWorld environment, tasks, and evaluators
- `results/stepwise/` — Output directory for per-agent logs, trajectories, and summaries

## Resources

- Benchmark website: https://os-world.github.io
- OSWorld GitHub: https://github.com/xlang-ai/OSWorld
- Paper: https://arxiv.org/abs/2404.07972
