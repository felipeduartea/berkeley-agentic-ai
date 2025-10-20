"""
TheAgentCompany Integration Adapter

Provides seamless integration between TheAgentCompany task evaluation
and the comprehensive evaluation framework.
"""

import json
import logging
import subprocess
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

try:
    from .core import ComprehensiveEvaluationFramework
    from .scoring import ScoringPolicy
except ImportError:
    from core import ComprehensiveEvaluationFramework
    from scoring import ScoringPolicy

logger = logging.getLogger(__name__)


class TheAgentCompanyEvaluator:
    """
    Wrapper for evaluating TheAgentCompany tasks with comprehensive framework.
    
    This integrates with TheAgentCompany's existing evaluation system
    while adding multi-layered analysis (heuristics + LLM judges).
    """
    
    def __init__(self,
                 scoring_policy: str = "balanced",
                 enable_llm_judges: bool = True,
                 llm_model: str = "gpt-4o",
                 llm_api_key: Optional[str] = None):
        """
        Initialize TheAgentCompany evaluator.
        
        Args:
            scoring_policy: Policy name ('conservative', 'balanced', 'ai_focused', 'strict')
            enable_llm_judges: Enable G-Eval and DeepEval judges
            llm_model: Model to use for LLM judges
            llm_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Select policy
        if scoring_policy == 'conservative':
            policy = ScoringPolicy.create_conservative_policy()
        elif scoring_policy == 'ai_focused':
            policy = ScoringPolicy.create_ai_focused_policy()
        elif scoring_policy == 'strict':
            policy = ScoringPolicy.create_strict_policy()
        else:
            policy = ScoringPolicy.create_balanced_policy()
        
        # Initialize framework
        self.framework = ComprehensiveEvaluationFramework(
            scoring_policy=policy,
            enable_llm_judges=enable_llm_judges,
            llm_model=llm_model,
            llm_api_key=llm_api_key
        )
        
        self.logger.info(
            f"TheAgentCompany evaluator initialized with policy: {scoring_policy}"
        )
    
    def evaluate_task_container(self,
                               container_name: str,
                               task_id: str,
                               trajectory_path: Optional[str] = None,
                               env_vars: Optional[Dict[str, str]] = None) -> Tuple[Dict, Dict]:
        """
        Evaluate a task running in a Docker container.
        
        This runs the standard TheAgentCompany eval.py script inside the container,
        then adds comprehensive analysis on top.
        
        Args:
            container_name: Name of the Docker container running the task
            task_id: Task identifier
            trajectory_path: Path to trajectory file (inside container or host)
            env_vars: Environment variables for eval.py (LITELLM_API_KEY, etc.)
            
        Returns:
            Tuple of (original_result, comprehensive_report)
        """
        self.logger.info(f"Evaluating task {task_id} in container {container_name}")
        
        # Step 1: Run original TheAgentCompany evaluator
        original_result = self._run_tac_evaluator(
            container_name, trajectory_path, env_vars
        )
        
        # Step 2: Load task instruction
        task_instruction = self._load_task_instruction(container_name)
        
        # Step 3: Load trajectory if provided
        trajectory = None
        if trajectory_path:
            trajectory = self._load_trajectory(container_name, trajectory_path)
        
        # Step 4: Run comprehensive evaluation
        aggregated_score, comprehensive_report = self.framework.evaluate_task(
            task_id=task_id,
            task_instruction=task_instruction,
            result_data=original_result,
            trajectory=trajectory
        )
        
        self.logger.info(
            f"Task {task_id} evaluation complete. "
            f"Original score: {original_result.get('score', 0):.3f}, "
            f"Comprehensive score: {aggregated_score.overall_score:.3f}"
        )
        
        return original_result, comprehensive_report
    
    def _run_tac_evaluator(self,
                          container_name: str,
                          trajectory_path: Optional[str],
                          env_vars: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """Run the standard TheAgentCompany evaluator in the container."""
        self.logger.info("Running standard TheAgentCompany evaluator...")
        
        # Build docker exec command
        cmd = ["docker", "exec"]
        
        # Add environment variables
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])
        
        # Add required env vars if not provided
        if not env_vars or 'DECRYPTION_KEY' not in env_vars:
            cmd.extend(["-e", "DECRYPTION_KEY=theagentcompany is all you need"])
        
        cmd.append(container_name)
        
        # Build eval.py command
        eval_cmd = ["python_default", "/utils/eval.py"]
        if trajectory_path:
            eval_cmd.extend(["--trajectory_path", trajectory_path])
        eval_cmd.extend(["--output_path", "/tmp/tac_eval_output.json"])
        
        cmd.extend(eval_cmd)
        
        try:
            # Run evaluation
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.error(f"Evaluator failed: {result.stderr}")
                return {'success': False, 'score': 0.0, 'error': result.stderr}
            
            # Read output
            read_cmd = ["docker", "exec", container_name, "cat", "/tmp/tac_eval_output.json"]
            read_result = subprocess.run(read_cmd, capture_output=True, text=True)
            
            if read_result.returncode == 0:
                output = json.loads(read_result.stdout)
                self.logger.info(f"Original evaluation: score={output.get('score', 0)}")
                return output
            else:
                self.logger.warning("Could not read evaluation output")
                return {'success': True, 'score': 1.0 if result.returncode == 0 else 0.0}
                
        except Exception as e:
            self.logger.error(f"Failed to run evaluator: {e}")
            return {'success': False, 'score': 0.0, 'error': str(e)}
    
    def _load_task_instruction(self, container_name: str) -> str:
        """Load task instruction from container."""
        try:
            cmd = ["docker", "exec", container_name, "cat", "/instruction/task.md"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return result.stdout
            else:
                self.logger.warning("Could not load task instruction")
                return "Task instruction not available"
        except Exception as e:
            self.logger.warning(f"Failed to load task instruction: {e}")
            return "Task instruction not available"
    
    def _load_trajectory(self, container_name: str, trajectory_path: str) -> Optional[List[Dict]]:
        """Load trajectory from container."""
        try:
            # Check if path is absolute (inside container) or relative (on host)
            if trajectory_path.startswith('/'):
                # Inside container
                cmd = ["docker", "exec", container_name, "cat", trajectory_path]
            else:
                # On host - copy from container first
                cmd = ["docker", "cp", f"{container_name}:{trajectory_path}", "/tmp/trajectory.json"]
                subprocess.run(cmd, check=True)
                cmd = ["cat", "/tmp/trajectory.json"]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                trajectory = json.loads(result.stdout)
                self.logger.info(f"Loaded trajectory with {len(trajectory)} steps")
                return trajectory
            else:
                self.logger.warning("Could not load trajectory")
                return None
        except Exception as e:
            self.logger.warning(f"Failed to load trajectory: {e}")
            return None
    
    def evaluate_from_files(self,
                           task_id: str,
                           task_instruction_path: str,
                           result_json_path: str,
                           trajectory_path: Optional[str] = None) -> Tuple[Dict, Dict]:
        """
        Evaluate a task from local files (without Docker container).
        
        Args:
            task_id: Task identifier
            task_instruction_path: Path to task instruction file
            result_json_path: Path to evaluation result JSON
            trajectory_path: Optional path to trajectory JSON
            
        Returns:
            Tuple of (original_result, comprehensive_report)
        """
        self.logger.info(f"Evaluating task {task_id} from files")
        
        # Load task instruction
        with open(task_instruction_path, 'r') as f:
            task_instruction = f.read()
        
        # Load result
        with open(result_json_path, 'r') as f:
            result_data = json.load(f)
        
        # Load trajectory if provided
        trajectory = None
        if trajectory_path and os.path.exists(trajectory_path):
            with open(trajectory_path, 'r') as f:
                trajectory = json.load(f)
        
        # Run comprehensive evaluation
        aggregated_score, comprehensive_report = self.framework.evaluate_task(
            task_id=task_id,
            task_instruction=task_instruction,
            result_data=result_data,
            trajectory=trajectory
        )
        
        return result_data, comprehensive_report
    
    def batch_evaluate_from_outputs(self,
                                    outputs_dir: str,
                                    comprehensive_output_dir: Optional[str] = None) -> List[Tuple[str, Dict, Dict]]:
        """
        Batch evaluate all tasks from an outputs directory.
        
        Expects outputs_dir structure:
            outputs_dir/
                task_1/
                    result.json
                    trajectory.json (optional)
                task_2/
                    result.json
                    trajectory.json (optional)
                ...
        
        Args:
            outputs_dir: Directory containing task results
            comprehensive_output_dir: Optional directory to save comprehensive reports
            
        Returns:
            List of (task_id, original_result, comprehensive_report) tuples
        """
        outputs_path = Path(outputs_dir)
        if not outputs_path.exists():
            raise ValueError(f"Outputs directory not found: {outputs_dir}")
        
        if comprehensive_output_dir:
            Path(comprehensive_output_dir).mkdir(parents=True, exist_ok=True)
        
        results = []
        task_dirs = [d for d in outputs_path.iterdir() if d.is_dir()]
        
        self.logger.info(f"Batch evaluating {len(task_dirs)} tasks")
        
        for task_dir in task_dirs:
            task_id = task_dir.name
            result_file = task_dir / "result.json"
            trajectory_file = task_dir / "trajectory.json"
            instruction_file = task_dir / "task.md"
            
            if not result_file.exists():
                self.logger.warning(f"No result.json found for {task_id}, skipping")
                continue
            
            # Default instruction
            task_instruction = f"Task {task_id}"
            if instruction_file.exists():
                with open(instruction_file, 'r') as f:
                    task_instruction = f.read()
            
            try:
                # Load result
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                # Load trajectory if exists
                trajectory = None
                if trajectory_file.exists():
                    with open(trajectory_file, 'r') as f:
                        trajectory = json.load(f)
                
                # Evaluate
                aggregated_score, comprehensive_report = self.framework.evaluate_task(
                    task_id=task_id,
                    task_instruction=task_instruction,
                    result_data=result_data,
                    trajectory=trajectory
                )
                
                results.append((task_id, result_data, comprehensive_report))
                
                # Save comprehensive report
                if comprehensive_output_dir:
                    report_path = Path(comprehensive_output_dir) / f"{task_id}_comprehensive.json"
                    with open(report_path, 'w') as f:
                        json.dump(comprehensive_report, f, indent=2)
                    self.logger.info(f"Saved comprehensive report to {report_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {task_id}: {e}")
        
        self.logger.info(f"Batch evaluation complete: {len(results)} tasks")
        return results

