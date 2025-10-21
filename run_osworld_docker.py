#!/usr/bin/env python3
"""Launch StepWise Green Agent inside Docker-managed OSWorld VMs."""

import os
import sys
import typing
from pathlib import Path

from dotenv import load_dotenv

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("PyYAML is required. Install with `pip install pyyaml`.") from exc

try:  # pragma: no cover - compatibility shim
    typing.TypeAlias  # type: ignore[attr-defined]
except AttributeError:
    from typing_extensions import TypeAlias as _TypeAlias

    typing.TypeAlias = _TypeAlias  # type: ignore[attr-defined,assignment]

# Ensure OSWorld modules are importable
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "osworld"))

load_dotenv()

from stepwise import (  # noqa: E402
    A11yWhiteAgent,
    ScreenshotWhiteAgent,
    StepWiseGreenAgent,
    WhiteAgentConfig,
)


OUTPUT_DIR = Path("results/stepwise")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def check_docker():
    """Check if Docker is available"""
    import subprocess
    try:
        subprocess.run(["docker", "version"], 
                      capture_output=True, 
                      check=True, 
                      timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

def run_stepwise_docker(task_file: Path, limit: int) -> bool:
    print("\n" + "=" * 70)
    print("  StepWise OSWorld Docker Execution")
    print("=" * 70)

    print("\nChecking Docker availability...")
    if not check_docker():
        print("❌ Docker not ready. Start Docker Desktop and retry.")
        return False
    print("✅ Docker detected")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-key-here":
        print("❌ OPENAI_API_KEY missing. Update .env with a valid key.")
        return False
    print("✅ OPENAI_API_KEY loaded")

    tasks_meta = task_file
    examples_dir = ROOT_DIR / "osworld" / "evaluation_examples"

    print(f"\nUsing task index: {tasks_meta}")

    try:
        wa_s_short = ScreenshotWhiteAgent(model="gpt-5", history_window=1)
        wa_s_long = ScreenshotWhiteAgent(model="gpt-5", history_window=3)
        wa_a_short = A11yWhiteAgent(model="gpt-5", history_window=1)
        wa_a_long = A11yWhiteAgent(model="gpt-5", history_window=3)
    except Exception as exc:
        print(f"❌ Failed to initialize White Agents: {exc}")
        return False

    configs = [
        WhiteAgentConfig(name="WA-S-1step", agent=wa_s_short, observation_type="screenshot"),
        WhiteAgentConfig(name="WA-S-3step", agent=wa_s_long, observation_type="screenshot"),
        WhiteAgentConfig(name="WA-A-1step", agent=wa_a_short, observation_type="a11y_tree"),
        WhiteAgentConfig(name="WA-A-3step", agent=wa_a_long, observation_type="a11y_tree"),
    ]

    stepwise = StepWiseGreenAgent(
        provider_name="docker",
        tasks_meta_path=tasks_meta,
        examples_base_path=examples_dir,
        output_dir=OUTPUT_DIR,
        white_agents=configs,
        max_steps=15,
        max_episode_seconds=180,
        sleep_after_action=1.0,
        headless=True,
    )

    summary = stepwise.run(task_limit=limit)

    summary_path = OUTPUT_DIR / "summary.yaml"
    with summary_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle)

    print("\n" + "=" * 70)
    print("  StepWise Run Complete")
    print("=" * 70)
    print(f"  Agents executed: {', '.join(summary.keys())}")
    print(f"  Summary saved to: {summary_path}")
    print("=" * 70)
    return True


def run_simulation_mode():
    """Run in simulation mode (no Docker required)"""
    print("\n" + "="*70)
    print("  OSWorld Simulation Mode")
    print("="*70)
    print("\nRunning basic simulation...")
    print("For Docker-based execution, ensure Docker Desktop is running")
    
    # Run the basic demo
    import subprocess
    result = subprocess.run([sys.executable, "run_osworld_agent.py"])
    return result.returncode == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run OSWorld with Docker")
    parser.add_argument("--mode", choices=["docker", "simulation"], default="docker")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT_DIR / "osworld" / "evaluation_examples" / "test_small.json",
        help="Path to OSWorld task index JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Maximum number of tasks to run (per agent)",
    )
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  🌍 OSWorld Docker Runner")
    print("="*70)
    
    if args.mode == "docker":
        success = run_stepwise_docker(args.tasks, args.limit)
    else:
        success = run_simulation_mode()
    
    sys.exit(0 if success else 1)

