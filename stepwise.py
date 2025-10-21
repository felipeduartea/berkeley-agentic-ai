#!/usr/bin/env python3
"""StepWise Green Agent coordinator for OSWorld."""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from desktop_env.desktop_env import DesktopEnv


load_dotenv()


LOGGER = logging.getLogger("stepwise")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _now() -> float:
    return time.perf_counter()


def _summarize_accessibility_tree(tree: Optional[Dict]) -> str:
    if not tree:
        return "Accessibility tree unavailable."
    elements = tree.get("children") or tree.get("elements") or []
    lines: List[str] = []
    stack: List[Tuple[Dict, int]] = []
    if isinstance(elements, list):
        for child in elements:
            stack.append((child, 0))
    elif isinstance(elements, dict):
        stack.append((elements, 0))
    visited = 0
    while stack and visited < 40:
        node, depth = stack.pop()
        visited += 1
        name = node.get("name") or node.get("text") or node.get("title") or "(no name)"
        role = node.get("role") or node.get("tag") or node.get("type") or "node"
        bounds = node.get("bounds") or node.get("bbox") or node.get("rect")
        bounds_text = f" bounds={bounds}" if bounds else ""
        lines.append(f"{'  '*depth}{role}: {name}{bounds_text}")
        child_nodes = node.get("children") or []
        if isinstance(child_nodes, list):
            for child in reversed(child_nodes):
                if isinstance(child, dict):
                    stack.append((child, depth + 1))
    if not lines:
        return "Accessibility tree empty."
    return "\n".join(lines)


def _format_latency(latencies: Sequence[float]) -> float:
    if not latencies:
        return 0.0
    return sum(latencies) / len(latencies)


def _classify_score(score: float) -> str:
    if score >= 0.99:
        return "success"
    if score >= 0.1:
        return "partial"
    return "fail"


def _load_task_index(meta_path: Path, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    pairs: List[Tuple[str, str]] = []
    for domain, ids in meta.items():
        for example_id in ids:
            pairs.append((domain, example_id))
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def _load_task_config(base_dir: Path, domain: str, example_id: str) -> Dict:
    task_path = base_dir / "examples" / domain / f"{example_id}.json"
    with task_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _encode_screenshot(png_bytes: Optional[bytes]) -> Optional[str]:
    if not png_bytes:
        return None
    return base64.b64encode(png_bytes).decode("utf-8")


@dataclass
class ActionLog:
    step: int
    raw_action: str
    reward: float
    done: bool
    latency: float
    info: Dict[str, Dict]


@dataclass
class EpisodeResult:
    task_id: str
    instruction: str
    outcome: str
    score: float
    duration: float
    steps: int
    action_logs: List[ActionLog] = field(default_factory=list)
    final_info: Dict[str, Dict] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class WhiteAgentBase:
    """Base class for White Agents (WA-S / WA-A)."""

    def __init__(
        self,
        name: str,
        model: str,
        history_window: int,
        use_screenshot: bool,
        use_a11y: bool,
        max_tokens: int = 900,
        temperature: float = 0.7,
    ) -> None:
        self.name = name
        self.model = model
        self.history_window = history_window
        self.use_screenshot = use_screenshot
        self.use_a11y = use_a11y
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._history: List[str] = []
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing for StepWise agents")
        self._client = OpenAI(api_key=api_key)

    def reset(self) -> None:
        self._history.clear()

    def _system_prompt(self) -> str:
        channel = []
        if self.use_screenshot:
            channel.append("screenshots")
        if self.use_a11y:
            channel.append("accessibility trees")
        modality = " and ".join(channel)
        return (
            f"You are {self.name}, a White Agent for StepWise. "
            f"You receive {modality} and must emit pyautogui commands, one per line. "
            "Available commands: pyautogui.click(x, y), pyautogui.moveTo(x, y), pyautogui.typewrite(text), "
            "pyautogui.press(key), pyautogui.hotkey(k1, k2), pyautogui.scroll(amount), WAIT, DONE, FAIL. "
            "Each response must contain a short reasoning section followed by a fenced code block with Python commands."
        )

    def _build_observation_text(self, instruction: str, obs: Dict) -> str:
        lines = ["# Task", instruction.strip()]
        if self.use_screenshot:
            b64 = _encode_screenshot(obs.get("screenshot"))
            if b64:
                lines.append("\n# Screenshot\nbase64://" + b64[:2048])
            else:
                lines.append("\n# Screenshot\nUnavailable")
        if self.use_a11y:
            lines.append("\n# Accessibility Tree\n" + _summarize_accessibility_tree(obs.get("accessibility_tree")))
        if obs.get("terminal"):
            snippet = obs["terminal"].decode("utf-8", errors="ignore")[-2048:]
            lines.append("\n# Terminal\n" + snippet)
        if self._history:
            lines.append("\n# Recent History\n" + "\n".join(self._history[-self.history_window :]))
        return "\n".join(lines)

    def _extract_commands(self, response: str) -> List[str]:
        code: Optional[str] = None
        if "```" in response:
            start = response.find("```")
            end = response.find("```", start + 3)
            if end != -1:
                code = response[start + 3 : end]
        if code is None:
            code = response
        commands: List[str] = []
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            commands.append(stripped)
        if not commands:
            return ["WAIT"]
        return commands

    def plan(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        user_prompt = self._build_observation_text(instruction, obs)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_prompt},
        ]

        LOGGER.debug("%s sending prompt (%d chars)", self.name, len(user_prompt))

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        response_text = completion.choices[0].message.content.strip()
        commands = self._extract_commands(response_text)
        self._history.append(f"Actions: {'; '.join(commands)}")
        return response_text, commands


class ScreenshotWhiteAgent(WhiteAgentBase):
    def __init__(self, model: str, history_window: int) -> None:
        super().__init__(
            name="WA-S",
            model=model,
            history_window=history_window,
            use_screenshot=True,
            use_a11y=False,
        )


class A11yWhiteAgent(WhiteAgentBase):
    def __init__(self, model: str, history_window: int) -> None:
        super().__init__(
            name="WA-A",
            model=model,
            history_window=history_window,
            use_screenshot=False,
            use_a11y=True,
        )


@dataclass
class WhiteAgentConfig:
    name: str
    agent: WhiteAgentBase
    observation_type: str


class StepWiseGreenAgent:
    """Green Agent orchestrator satisfying OSWorld StepWise spec."""

    def __init__(
        self,
        provider_name: str,
        tasks_meta_path: Path,
        examples_base_path: Path,
        output_dir: Path,
        white_agents: Sequence[WhiteAgentConfig],
        max_steps: int = 15,
        max_episode_seconds: int = 180,
        sleep_after_action: float = 1.0,
        headless: bool = False,
        os_type: str = "Ubuntu",
        client_password: str = "",
    ) -> None:
        self.provider_name = provider_name
        self.tasks_meta_path = tasks_meta_path
        self.examples_base_path = examples_base_path
        self.output_dir = output_dir
        self.white_agents = list(white_agents)
        self.max_steps = max_steps
        self.max_episode_seconds = max_episode_seconds
        self.sleep_after_action = sleep_after_action
        self.headless = headless
        self.os_type = os_type
        self.client_password = client_password
        _ensure_dir(self.output_dir)

    def run(self, task_limit: Optional[int] = None) -> Dict[str, Dict]:
        LOGGER.info("Loading task index from %s", self.tasks_meta_path)
        task_pairs = _load_task_index(self.tasks_meta_path, task_limit)
        LOGGER.info("Loaded %d tasks", len(task_pairs))

        summary: Dict[str, Dict] = {}

        for config in self.white_agents:
            LOGGER.info("Running agent %s", config.name)
            agent_dir = self.output_dir / config.name
            _ensure_dir(agent_dir)
            results = self._run_agent(config, task_pairs, agent_dir)
            summary[config.name] = results

        summary_path = self.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        LOGGER.info("StepWise run complete. Summary saved to %s", summary_path)
        return summary

    def _run_agent(
        self,
        config: WhiteAgentConfig,
        task_pairs: Iterable[Tuple[str, str]],
        agent_dir: Path,
    ) -> Dict[str, float]:
        require_tree = config.observation_type in {"a11y_tree", "screenshot_a11y_tree"}
        env = DesktopEnv(
            provider_name=self.provider_name,
            os_type=self.os_type,
            action_space="pyautogui",
            require_a11y_tree=require_tree,
            headless=self.headless,
            client_password=self.client_password,
        )
        scores: List[float] = []
        latencies: List[float] = []
        steps_taken: List[int] = []
        outcomes: Dict[str, int] = {"success": 0, "partial": 0, "fail": 0}

        try:
            for domain, example_id in task_pairs:
                task_config = _load_task_config(self.examples_base_path, domain, example_id)
                result = self._run_task(env, config.agent, task_config)
                scores.append(result.score)
                latencies.extend([log.latency for log in result.action_logs])
                steps_taken.append(result.steps)
                outcomes[result.outcome] += 1
                self._write_episode(agent_dir, domain, result)
        finally:
            env.close()

        total = max(1, len(scores))
        avg_score = sum(scores) / total
        avg_steps = sum(steps_taken) / len(steps_taken) if steps_taken else 0.0
        avg_latency = _format_latency(latencies)
        success_rate = outcomes["success"] / total

        metrics = {
            "agent": config.name,
            "episodes": len(scores),
            "average_score": avg_score,
            "average_steps": avg_steps,
            "average_latency": avg_latency,
            "success_rate": success_rate,
            "outcomes": outcomes,
        }

        metrics_path = agent_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        LOGGER.info("Agent %s metrics saved to %s", config.name, metrics_path)
        return metrics

    def _run_task(self, env: DesktopEnv, agent: WhiteAgentBase, task_config: Dict) -> EpisodeResult:
        task_id = task_config.get("id", "unknown")
        instruction = task_config.get("instruction", "")
        agent.reset()
        LOGGER.info("Starting task %s", task_id)

        try:
            observation = env.reset(task_config=task_config)
        except Exception as exc:
            LOGGER.exception("Environment reset failed for %s", task_id)
            return EpisodeResult(
                task_id=task_id,
                instruction=instruction,
                outcome="fail",
                score=0.0,
                duration=0.0,
                steps=0,
                errors=[f"reset_error: {exc}"],
            )

        time.sleep(10)
        done = False
        step = 0
        action_logs: List[ActionLog] = []
        start = _now()

        while not done and step < self.max_steps:
            if _now() - start > self.max_episode_seconds:
                LOGGER.warning("Task %s timed out", task_id)
                break
            try:
                response_text, commands = agent.plan(instruction, observation)
            except Exception as exc:
                LOGGER.exception("Agent failure on task %s", task_id)
                return EpisodeResult(
                    task_id=task_id,
                    instruction=instruction,
                    outcome="fail",
                    score=0.0,
                    duration=_now() - start,
                    steps=step,
                    action_logs=action_logs,
                    errors=[f"agent_error: {exc}"],
                )

            LOGGER.debug("Task %s step %d response: %s", task_id, step + 1, response_text[:2000])

            for raw_action in commands:
                step_start = _now()
                try:
                    observation, reward, done, info = env.step(raw_action, self.sleep_after_action)
                except Exception as exc:
                    LOGGER.exception("Environment step failed on %s", task_id)
                    return EpisodeResult(
                        task_id=task_id,
                        instruction=instruction,
                        outcome="fail",
                        score=0.0,
                        duration=_now() - start,
                        steps=step,
                        action_logs=action_logs,
                        errors=[f"step_error: {exc}"],
                    )

                latency = _now() - step_start
                action_logs.append(
                    ActionLog(
                        step=step + 1,
                        raw_action=raw_action,
                        reward=reward,
                        done=done,
                        latency=latency,
                        info=info,
                    )
                )

                if done:
                    break

            step += 1

        elapsed = _now() - start

        try:
            score = env.evaluate()
        except Exception as exc:
            LOGGER.exception("Evaluation failed for %s", task_id)
            score = 0.0
            eval_error = [f"evaluation_error: {exc}"]
        else:
            eval_error = []

        outcome = _classify_score(score)
        LOGGER.info("Task %s finished with score %.2f (%s)", task_id, score, outcome)

        return EpisodeResult(
            task_id=task_id,
            instruction=instruction,
            outcome=outcome,
            score=score,
            duration=elapsed,
            steps=step,
            action_logs=action_logs,
            final_info={"done": done},
            errors=eval_error,
        )

    def _write_episode(self, agent_dir: Path, domain: str, result: EpisodeResult) -> None:
        domain_dir = agent_dir / domain
        _ensure_dir(domain_dir)
        payload = {
            "task_id": result.task_id,
            "instruction": result.instruction,
            "outcome": result.outcome,
            "score": result.score,
            "duration": result.duration,
            "steps": result.steps,
            "actions": [
                {
                    "step": log.step,
                    "raw_action": log.raw_action,
                    "reward": log.reward,
                    "done": log.done,
                    "latency": log.latency,
                    "info": log.info,
                }
                for log in result.action_logs
            ],
            "final_info": result.final_info,
            "errors": result.errors,
        }
        out_path = domain_dir / f"{result.task_id}.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        LOGGER.debug("Episode %s written to %s", result.task_id, out_path)


__all__ = [
    "StepWiseGreenAgent",
    "ScreenshotWhiteAgent",
    "A11yWhiteAgent",
    "WhiteAgentConfig",
]

