# Green Agent Evaluation

Run AI agents and evaluate them properly.

## What This Does

Most agent evaluations just give you a pass/fail score. That sucks.

This framework evaluates agents in 3 ways:
1. **Deterministic checks** - Did it actually do the thing?
2. **Behavioral metrics** - How well did it do it?
3. **AI judges** - What would a human think?

Then it combines all three with confidence weighting.

## Quick Start

```bash
# Setup (first time only)
cd ../OSWorld
python3 -m venv osworld-env
source osworld-env/bin/activate
pip install openai pyyaml httpx

# Run
export OPENAI_API_KEY="your-key"
python3 run_agent.py
```

Check `results/REPORT.md` for the summary.

## What You Get

```
results/
├── REPORT.md           # Human-readable summary
├── actions.json        # Everything the agent did
├── evaluation.json     # Full analysis
└── result.json         # Basic scores
```

## How It Works

```
Agent runs → Logs every action → Evaluation framework analyzes:
  ├─ Deterministic (binary checks)
  ├─ Heuristics (efficiency, coherence, quality, resources)
  └─ LLM Judges (GPT-4 evaluation with reasoning)
     ├─ G-Eval (chain of thought)
     └─ DeepEval (multi-metric)

→ Weighted score + confidence + detailed breakdown
```

## Example Output

```
Original Score:  0.750
Green Agent:     0.817
Confidence:      0.885

Breakdown:
  deterministic: 0.750
  heuristic    : 0.887
  llm_judge    : 0.830

AI Judges:
  g_eval       : 0.900
  "Task completed successfully with logical approach..."
```

## The Framework

Located in `evaluation/comprehensive_framework/`:

- `core.py` - Main orchestrator
- `llm_judges.py` - G-Eval, DeepEval, custom rubrics
- `heuristics.py` - Behavioral metrics
- `scoring.py` - Aggregation and policies
- `tac_integration.py` - TheAgentCompany adapter

## Why This Matters

Single scores hide everything. Was the agent slow? Did it waste actions? 
Was the approach logical? You don't know.

This gives you:
- Component scores (what worked, what didn't)
- Confidence levels (how certain we are)
- AI reasoning (why this score)
- Behavioral analysis (how it performed)

Basically, you can actually understand what happened.

## Extending It

Add your own metrics:

```python
from heuristics import BaseMetric

class MyMetric(BaseMetric):
    def evaluate(self, trajectory, result):
        # Your logic
        return score, confidence
```

Or custom judges:

```python
from llm_judges import BaseLLMJudge

class MyJudge(BaseLLMJudge):
    def evaluate_task(self, task_data):
        # Your evaluation
        return result
```

## Notes

- Costs ~$0.02 per evaluation (GPT-4 API)
- Takes ~10-15 seconds
- Parallelizable for batch runs
- All evaluations are reproducible

## That's It

Read `SETUP.md` for detailed setup.

Run `python3 run_agent.py` to try it.

Check the code if you want to understand how it works.

