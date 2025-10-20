# Quick Reference

## Setup (once)

```bash
pip3 install openai pyyaml httpx
export OPENAI_API_KEY="your-key"
```

## Run

```bash
python3 run_agent.py
```

## Results

```bash
cat results/REPORT.md
```

## Files Created

- `REPORT.md` - Summary
- `actions.json` - What the agent did
- `evaluation.json` - Full analysis
- `result.json` - Basic scores

## Scores

- **deterministic** - Binary checks
- **heuristic** - Behavioral metrics
- **llm_judge** - AI assessment

Combined with confidence weighting → overall score

## That's It

Read `SETUP.md` for troubleshooting.

Read `README_GREENAGENT.md` for details.

