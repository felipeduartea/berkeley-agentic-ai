#!/usr/bin/env python3
"""
Demo mode - shows the evaluation with better output for presentations
"""

import os
import sys
import json
import time
from pathlib import Path

def print_section(title):
    """Print a section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def print_step(number, text):
    """Print a step"""
    print(f"→ Step {number}: {text}")

def print_result(label, value, width=20):
    """Print a result line"""
    print(f"  {label:<{width}}: {value}")

def pause(seconds=1):
    """Pause for dramatic effect"""
    time.sleep(seconds)

# Check if we have results
results_dir = Path("results")
if not results_dir.exists():
    print("❌ No results found. Run 'python3 run_agent.py' first.")
    sys.exit(1)

# Load results
with open(results_dir / "result.json") as f:
    result = json.load(f)

with open(results_dir / "evaluation.json") as f:
    evaluation = json.load(f)

with open(results_dir / "actions.json") as f:
    actions = json.load(f)

with open(results_dir / "task.txt") as f:
    task = f.read()

# Demo starts
os.system('clear')

print_section("Green Agent Evaluation Demo")

pause(1)

print("Most agent evaluations give you one number.")
print("That tells you almost nothing.\n")
print("Let me show you a better way...\n")

pause(2)

# Part 1: The Task
print_section("1. The Task")

print("The agent was given this task:\n")
print(f'"{task[:200]}..."\n')
print(f"It took {len(actions)} actions over {result['time']:.1f} seconds.")

pause(3)

# Part 2: Traditional Evaluation
print_section("2. Traditional Evaluation")

print("Here's what a typical evaluation tells you:\n")
print_result("Success", "✓ Yes" if result['success'] else "✗ No")
print_result("Score", f"{result['score']:.3f}")
print("\nThat's it. Not very helpful, right?")

pause(3)

# Part 3: Green Agent Evaluation
print_section("3. Green Agent Evaluation")

print("Now let's see what our framework found:\n")

original_score = result['score']
comprehensive_score = evaluation['evaluation_summary']['overall_score']
confidence = evaluation['evaluation_summary']['confidence']

print_result("Traditional Score", f"{original_score:.3f}")
print_result("Green Agent Score", f"{comprehensive_score:.3f} ({'higher' if comprehensive_score > original_score else 'lower'})", width=20)
print_result("Confidence", f"{confidence:.1%}")

pause(2)

print("\n→ Why the difference? Let's break it down...")

pause(2)

# Part 4: Component Breakdown
print_section("4. Three-Layer Analysis")

components = evaluation['component_scores']

print("The framework evaluates in three ways:\n")

pause(1)

print("1. DETERMINISTIC CHECKS (Binary Pass/Fail)")
print_result("   Score", f"{components['deterministic']['score']:.3f}")
print_result("   Confidence", f"{components['deterministic']['confidence']:.1%}")
print_result("   Weight", f"{components['deterministic']['weight']:.1%}")

pause(1.5)

print("\n2. BEHAVIORAL METRICS (How Well?)")
print_result("   Score", f"{components['heuristic']['score']:.3f}")
print_result("   Confidence", f"{components['heuristic']['confidence']:.1%}")
print_result("   Weight", f"{components['heuristic']['weight']:.1%}")

pause(1.5)

print("\n3. AI JUDGES (What Would a Human Think?)")
print_result("   Score", f"{components['llm_judge']['score']:.3f}")
print_result("   Confidence", f"{components['llm_judge']['confidence']:.1%}")
print_result("   Weight", f"{components['llm_judge']['weight']:.1%}")

pause(2)

# Part 5: Detailed Metrics
print_section("5. Behavioral Metrics")

metrics = evaluation['heuristic_metrics']

print("How the agent actually performed:\n")

for i, metric in enumerate(metrics[:4], 1):
    metric_name = metric['metric_id'].replace('_', ' ').title()
    score = metric['score']
    
    # Color code
    if score >= 0.8:
        indicator = "✓"
    elif score >= 0.6:
        indicator = "~"
    else:
        indicator = "✗"
    
    print(f"{i}. {metric_name:<25s} {indicator} {score:.3f}")
    
    pause(0.8)

pause(2)

# Part 6: AI Judge Reasoning
print_section("6. AI Judge Assessment")

if 'llm_judges' in evaluation:
    print("Two AI judges evaluated the execution:\n")
    
    for judge_name, judge_data in evaluation['llm_judges'].items():
        pause(1)
        
        print(f"\n{judge_name.upper()}: {judge_data['score']:.3f}")
        print(f"Confidence: {judge_data['confidence']:.1%}\n")
        
        rationale = judge_data['rationale'][:200]
        print(f'"{rationale}..."')
        
        pause(2)

# Part 7: The Insight
print_section("7. What This Tells Us")

pause(1)

print("With traditional evaluation, you get: Task passed, 0.75\n")

pause(1.5)

print("With Green Agent, you learn:")
print(f"  • The agent was {'efficient' if metrics[0]['score'] > 0.8 else 'somewhat efficient'} ({metrics[0]['score']:.2f})")
print(f"  • Actions were {'coherent' if metrics[1]['score'] > 0.7 else 'could be more coherent'} ({metrics[1]['score']:.2f})")
print(f"  • Quality was {'high' if metrics[2]['score'] > 0.8 else 'good'} ({metrics[2]['score']:.2f})")
print(f"  • Resource usage was {'optimal' if metrics[3]['score'] > 0.9 else 'reasonable'} ({metrics[3]['score']:.2f})")

if 'llm_judges' in evaluation:
    g_eval_score = evaluation['llm_judges']['g_eval']['score']
    print(f"  • AI assessment: {g_eval_score:.2f} - 'logical and structured approach'")

pause(3)

# Part 8: Where to Learn More
print_section("8. Learn More")

print("Files generated from this run:\n")
print("  results/REPORT.md       - Human-readable summary")
print("  results/actions.json    - All agent actions")
print("  results/evaluation.json - Complete analysis")
print("  results/result.json     - Basic scores\n")

pause(1)

print("Documentation:\n")
print("  QUICK.md               - 30-second reference")
print("  SETUP.md               - Installation guide")
print("  README_GREENAGENT.md   - Full explanation")
print("  DEMO.md                - This demo script\n")

pause(1)

print("To run this yourself:\n")
print("  1. pip3 install openai pyyaml httpx")
print("  2. export OPENAI_API_KEY='your-key'")
print("  3. python3 run_agent.py\n")

pause(2)

# Final screen
print_section("That's Green Agent Evaluation")

print("Better insights. Better debugging. Better understanding.\n")
print(f"This evaluation took {evaluation['evaluation_summary']['evaluation_duration_seconds']:.1f}s")
print(f"and cost ~$0.02 in API calls.\n")
print("Questions?")

print("\n" + "="*70 + "\n")

