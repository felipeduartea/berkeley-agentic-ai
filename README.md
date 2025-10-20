# Green Agent Evaluation Framework

This implements a comprehensive evaluation system for AI agents, inspired by OSWorld but adapted for TheAgentCompany benchmark.

## What's Here

```/
├── run_agent.py              # Run and evaluate an agent
├── README_GREENAGENT.md      # Full explanation
├── SETUP.md                  # Setup instructions
├── results/                  # Output from runs
│   ├── REPORT.md            # Summary
│   ├── actions.json         # Agent actions
│   └── evaluation.json      # Full analysis
└── evaluation/
    └── comprehensive_framework/
        ├── core.py          # Main framework
        ├── llm_judges.py    # G-Eval, DeepEval
        ├── heuristics.py    # Behavioral metrics
        └── scoring.py       # Score aggregation
```

## Quick Start

```bash
# Install
pip3 install openai pyyaml httpx

# Run
export OPENAI_API_KEY="your-key"
python3 run_agent.py

# Check results
cat results/REPORT.md
```

## How It Works

Instead of just pass/fail, evaluates in 3 ways:

1. **Deterministic** - Binary checks (did it work?)
2. **Heuristics** - Behavioral analysis (efficiency, coherence, quality)
3. **LLM Judges** - AI assessment (G-Eval with chain-of-thought, DeepEval)

Then combines all three with confidence weighting to give you a real understanding of what happened.

## What You Get

```
Original Score:  0.750
Green Agent:     0.817
Confidence:      0.885

Breakdown:
  deterministic: 0.750
  heuristic    : 0.887
  llm_judge    : 0.830

Top Metrics:
  1. task_efficiency    : 0.950
  2. action_coherence   : 0.750
  3. completion_quality : 0.850

AI Judges:
  g_eval: 0.900
  "Task completed successfully with logical approach..."
```

Plus full reports with reasoning, timings, and recommendations.

## Files

Read these in order:

./SETUP.md` - How to install
./README_GREENAGENT.md` - Full explanation
./run_agent.py` - The actual code

## Notes

- Costs ~$0.02 per eval (GPT-4 API)
- Takes ~10-15 seconds
- Works with any agent that logs actions
- Extensible with custom metrics/judges

That's it.

