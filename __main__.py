#!/usr/bin/env python3
"""CLI entrypoint for running StepWise Green Agent."""

from pathlib import Path

import argparse

from run_osworld_docker import run_stepwise_docker, run_simulation_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StepWise OSWorld runner")
    parser.add_argument(
        "--mode",
        choices=["docker", "simulation"],
        default="docker",
        help="Execution mode",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("osworld/evaluation_examples/test_small.json"),
        help="Task index JSON path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Maximum number of tasks to run per agent",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "docker":
        ok = run_stepwise_docker(args.tasks, args.limit)
    else:
        ok = run_simulation_mode()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

