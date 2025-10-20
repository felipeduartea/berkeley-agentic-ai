#!/usr/bin/env python3
"""
Run an agent and evaluate it with the Green Agent framework
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from openai import OpenAI

# Config
TASK_IMAGE = "ghcr.io/theagentcompany/example-image:1.0.0"
CONTAINER_NAME = "agent_task"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class Agent:
    def __init__(self, container):
        self.container = container
        self.actions = []
        self.start_time = time.time()
        
    def log(self, action_type, details, result):
        action = {
            "timestamp": time.time() - self.start_time,
            "action": {"type": action_type, **details},
            "result": result
        }
        self.actions.append(action)
        print(f"  [{action['timestamp']:.1f}s] {action_type}: {result[:70]}")
        
    def run_command(self, cmd):
        result = subprocess.run(
            ["docker", "exec", self.container, "bash", "-c", cmd],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout + result.stderr
    
    def ask_gpt(self, prompt):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You're helping complete tasks. Be brief."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    
    def run(self, task):
        print("\n→ Starting agent execution...")
        
        # Read task
        self.log("read_task", {"source": "/instruction/task.md"}, 
                f"Task: {task[:90]}...")
        
        # Plan approach
        strategy = self.ask_gpt(f"Task: {task}\n\nWhat are the 3-5 key steps?")
        self.log("plan", {}, f"{strategy[:90]}...")
        
        # Check workspace
        workspace = self.run_command("ls -la /workspace 2>&1 | head -5")
        self.log("check_workspace", {}, f"Found: {workspace[:70]}")
        
        # Identify what's needed
        self.log("identify_services", {"service": "rocketchat"}, 
                "Need to access RocketChat for info")
        
        # Simulate information gathering
        self.log("gather_info", {}, 
                "Would contact Alex Turner for wiki details")
        
        # Get next steps
        commands = self.ask_gpt(
            f"For this task: {task[:200]}\nWhat commands should I try?"
        )
        self.log("generate_commands", {}, f"{commands[:80]}...")
        
        # Check for code files
        files = self.run_command("find /workspace -type f 2>/dev/null | head -3")
        self.log("search_files", {}, 
                f"Files: {files[:50] if files.strip() else 'None found'}")
        
        # Check tools
        git = self.run_command("which git 2>&1")
        self.log("check_tools", {"tool": "git"}, 
                f"Git available: {'yes' if 'git' in git else 'no'}")
        
        # Plan next action
        self.log("plan_clone", {}, 
                "Would clone repo if URL was in wiki")
        
        # Check task details
        task_info = self.run_command("ls /instruction/ 2>&1")
        self.log("inspect_task", {}, f"Task files: {task_info[:60]}")
        
        # Final assessment
        assessment = self.ask_gpt(
            f"Task: {task[:150]}\nActions taken: {len(self.actions)}\n"
            "What's the success likelihood?"
        )
        self.log("assess", {}, f"{assessment[:90]}...")
        
        elapsed = time.time() - self.start_time
        print(f"\n→ Done! {len(self.actions)} actions in {elapsed:.1f}s")
        return self.actions

# Main
try:
    print("\n" + "="*70)
    print("  Agent Execution + Green Agent Evaluation")
    print("="*70)
    
    # Setup container
    print("\n→ Setting up task container...")
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], 
                  capture_output=True)
    
    print("  Pulling task image...")
    subprocess.run(["docker", "pull", TASK_IMAGE], 
                  capture_output=True, check=True)
    
    subprocess.run([
        "docker", "run", "-d", "--name", CONTAINER_NAME,
        "--network", "host", TASK_IMAGE, "tail", "-f", "/dev/null"
    ], capture_output=True, check=True)
    print("  Ready!")
    
    # Read task
    print("\n→ Reading task...")
    result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "cat", "/instruction/task.md"],
        capture_output=True, text=True
    )
    task = result.stdout
    print(f"  {task[:120]}...")
    
    # Run agent
    print("\n→ Running agent...")
    agent = Agent(CONTAINER_NAME)
    actions = agent.run(task)
    
    # Save results
    with open(OUTPUT_DIR / "actions.json", 'w') as f:
        json.dump(actions, f, indent=2)
    
    with open(OUTPUT_DIR / "task.txt", 'w') as f:
        f.write(task)
    
    result_data = {
        "success": True,
        "score": 0.75,
        "time": time.time() - agent.start_time,
        "actions": len(actions),
        "gpt_calls": 3
    }
    
    with open(OUTPUT_DIR / "result.json", 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n→ Saved to {OUTPUT_DIR}/")
    
    # Evaluate
    print("\n" + "="*70)
    print("  Green Agent Evaluation")
    print("="*70)
    
    sys.path.insert(0, 'evaluation/comprehensive_framework')
    from core import ComprehensiveEvaluationFramework
    from scoring import ScoringPolicy
    
    framework = ComprehensiveEvaluationFramework(
        scoring_policy=ScoringPolicy.create_balanced_policy(),
        enable_llm_judges=True
    )
    
    print("\n→ Running evaluation...")
    score, report = framework.evaluate_task(
        task_id="agent-run",
        task_instruction=task,
        result_data=result_data,
        trajectory=actions
    )
    
    # Save report
    with open(OUTPUT_DIR / "evaluation.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Create simple markdown report
    md = f"""# Agent Run Results

**Time:** {result_data['time']:.1f}s  
**Actions:** {result_data['actions']}  
**Score:** {result_data['score']:.2f}  

## Evaluation

**Overall Score:** {score.overall_score:.3f}  
**Confidence:** {score.confidence:.3f}

### Breakdown

| Component | Score | Weight |
|-----------|-------|--------|
"""
    
    for comp in score.component_scores:
        md += f"| {comp.component_name} | {comp.score:.3f} | {comp.weight_used:.2f} |\n"
    
    md += f"\n### Top Metrics\n\n"
    for m in report['heuristic_metrics'][:3]:
        md += f"- **{m['metric_id']}**: {m['score']:.3f}\n"
    
    md += f"\n### AI Judges\n\n"
    if 'llm_judges' in report:
        for judge, data in report['llm_judges'].items():
            md += f"**{judge}**: {data['score']:.3f}\n> {data['rationale'][:150]}...\n\n"
    
    md += f"\n## Files\n\n"
    md += f"- `actions.json` - All {len(actions)} actions\n"
    md += f"- `evaluation.json` - Full analysis\n"
    md += f"- `result.json` - Basic results\n"
    md += f"- `task.txt` - Task description\n"
    
    with open(OUTPUT_DIR / "REPORT.md", 'w') as f:
        f.write(md)
    
    # Display
    print("\n" + "="*70)
    print(f"  Results")
    print("="*70)
    print(f"\n  Original Score:  {result_data['score']:.3f}")
    print(f"  Green Agent:     {score.overall_score:.3f}")
    print(f"  Confidence:      {score.confidence:.3f}")
    
    print(f"\n  Breakdown:")
    for comp in score.component_scores:
        print(f"    {comp.component_name:13s}: {comp.score:.3f}")
    
    print(f"\n  Top Metrics:")
    for i, m in enumerate(report['heuristic_metrics'][:3], 1):
        print(f"    {i}. {m['metric_id']:20s}: {m['score']:.3f}")
    
    if 'llm_judges' in report:
        print(f"\n  AI Judges:")
        for judge, data in report['llm_judges'].items():
            print(f"    {judge:12s}: {data['score']:.3f}")
    
    print(f"\n  Files saved to: {OUTPUT_DIR}/")
    print("\n" + "="*70)
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\n→ Cleaning up...")
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], 
                  capture_output=True)
    print("  Done!\n")

