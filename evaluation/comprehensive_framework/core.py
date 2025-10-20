"""
Core Framework - Main orchestrator for comprehensive evaluation

Integrates all evaluation components and provides a unified interface for
evaluating TheAgentCompany tasks.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

try:
    from .llm_judges import (
        LLMJudgeEngine, GEvalJudge, DeepEvalJudge, RubricBasedJudge,
        LLMJudgmentResult, EvaluationRubric
    )
    from .heuristics import HeuristicsEngine, MetricResult
    from .scoring import ScoringAggregator, ScoringPolicy, AggregatedScore
except ImportError:
    from llm_judges import (
        LLMJudgeEngine, GEvalJudge, DeepEvalJudge, RubricBasedJudge,
        LLMJudgmentResult, EvaluationRubric
    )
    from heuristics import HeuristicsEngine, MetricResult
    from scoring import ScoringAggregator, ScoringPolicy, AggregatedScore

logger = logging.getLogger(__name__)


class ComprehensiveEvaluationFramework:
    """
    Main evaluation framework for TheAgentCompany tasks.
    
    Architecture Flow:
        [Task Result] + [Trajectory]
                │
                ▼
         [Load & Parse Data]
                │
                ├──► [Heuristics & Metrics] ─────────────► behavioral scores
                │
                ├──► [LLM Judges (G-Eval/DeepEval)] ─────► AI-powered scores
                │
                ▼
         [Scoring Aggregator + Policy]
                │
                ▼
         [Comprehensive Report]
    """
    
    def __init__(self,
                 scoring_policy: Optional[ScoringPolicy] = None,
                 enable_llm_judges: bool = True,
                 llm_model: str = "gpt-4o",
                 llm_api_key: Optional[str] = None):
        """
        Initialize the comprehensive evaluation framework.
        
        Args:
            scoring_policy: Policy for aggregating scores (default: balanced)
            enable_llm_judges: Whether to enable LLM-based evaluation
            llm_model: Model name for LLM judges
            llm_api_key: API key for LLM (uses OPENAI_API_KEY env if not provided)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.heuristics_engine = HeuristicsEngine()
        self.llm_judge_engine = None
        self.scoring_aggregator = ScoringAggregator(
            scoring_policy or ScoringPolicy.create_balanced_policy()
        )
        
        # Initialize LLM judges if enabled
        if enable_llm_judges:
            self.llm_judge_engine = LLMJudgeEngine()
            # Add default judges
            self.llm_judge_engine.add_judge(GEvalJudge(model_name=llm_model))
            self.llm_judge_engine.add_judge(DeepEvalJudge(model_name=llm_model))
            self.logger.info("LLM judges enabled (G-Eval, DeepEval)")
        else:
            self.logger.info("LLM judges disabled")
        
        self.enable_llm_judges = enable_llm_judges
        
        self.logger.info("Comprehensive Evaluation Framework initialized")
    
    def evaluate_task(self,
                     task_id: str,
                     task_instruction: str,
                     result_data: Dict[str, Any],
                     trajectory: Optional[List[Dict]] = None,
                     trajectory_path: Optional[str] = None) -> Tuple[AggregatedScore, Dict[str, Any]]:
        """
        Evaluate a single task with comprehensive analysis.
        
        Args:
            task_id: Unique task identifier
            task_instruction: Task instruction text
            result_data: Result from task evaluation (must include 'success' and 'score')
            trajectory: Optional execution trajectory (list of action dicts)
            trajectory_path: Optional path to trajectory JSON file
            
        Returns:
            Tuple of (AggregatedScore, detailed_report_dict)
        """
        evaluation_start = datetime.now()
        self.logger.info(f"Starting comprehensive evaluation for task: {task_id}")
        
        # Load trajectory if path provided
        if trajectory_path and os.path.exists(trajectory_path):
            try:
                with open(trajectory_path, 'r') as f:
                    trajectory = json.load(f)
                self.logger.info(f"Loaded trajectory from {trajectory_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load trajectory: {e}")
                trajectory = None
        
        # Prepare task data
        task_data = {
            'id': task_id,
            'instruction': task_instruction,
            'evaluated_at': evaluation_start.isoformat()
        }
        
        # Phase 1: Heuristic Metrics Evaluation
        self.logger.info("Phase 1: Running heuristic metrics")
        heuristic_results = self.heuristics_engine.evaluate_all(
            task_data, trajectory, result_data
        )
        heuristic_score = self.heuristics_engine.aggregate_metric_scores(heuristic_results)
        heuristic_confidence = sum(r.confidence for r in heuristic_results) / len(heuristic_results) if heuristic_results else 0.7
        
        self.logger.info(f"Heuristics complete: score={heuristic_score:.3f}, confidence={heuristic_confidence:.3f}")
        
        # Phase 2: LLM Judge Evaluation
        llm_results = {}
        llm_scores = {}
        llm_confidences = {}
        
        if self.enable_llm_judges and self.llm_judge_engine:
            self.logger.info("Phase 2: Running LLM judges")
            try:
                llm_results = self.llm_judge_engine.evaluate_all(
                    task_data, trajectory, result_data
                )
                
                for judge_id, judge_result in llm_results.items():
                    llm_scores[judge_id] = judge_result.overall_score
                    llm_confidences[judge_id] = judge_result.confidence
                
                avg_llm_score = sum(llm_scores.values()) / len(llm_scores) if llm_scores else 0.0
                self.logger.info(f"LLM judges complete: avg score={avg_llm_score:.3f}")
            except Exception as e:
                self.logger.error(f"LLM judge evaluation failed: {e}")
        else:
            self.logger.info("Phase 2: Skipped (LLM judges disabled)")
        
        # Phase 3: Extract deterministic score from result_data
        deterministic_score = float(result_data.get('score', 0.0))
        deterministic_confidence = 1.0 if result_data.get('success', False) else 0.8
        
        self.logger.info(f"Deterministic score: {deterministic_score:.3f}")
        
        # Phase 4: Score Aggregation
        self.logger.info("Phase 3: Aggregating scores")
        aggregated_score = self.scoring_aggregator.aggregate_evaluation(
            deterministic_score=deterministic_score,
            deterministic_confidence=deterministic_confidence,
            heuristic_score=heuristic_score,
            heuristic_confidence=heuristic_confidence,
            llm_scores=llm_scores,
            llm_confidences=llm_confidences,
            deterministic_details={'raw_result': result_data},
            heuristic_details={'metrics': [r.__dict__ for r in heuristic_results]},
            llm_details={'judges': {k: v.__dict__ for k, v in llm_results.items()}}
        )
        
        evaluation_duration = (datetime.now() - evaluation_start).total_seconds()
        
        self.logger.info(
            f"Evaluation complete in {evaluation_duration:.2f}s - "
            f"Overall score: {aggregated_score.overall_score:.3f} "
            f"(confidence: {aggregated_score.confidence:.3f})"
        )
        
        # Phase 5: Build detailed report
        detailed_report = self._build_detailed_report(
            task_data=task_data,
            result_data=result_data,
            trajectory=trajectory,
            heuristic_results=heuristic_results,
            llm_results=llm_results,
            aggregated_score=aggregated_score,
            evaluation_duration=evaluation_duration
        )
        
        return aggregated_score, detailed_report
    
    def _build_detailed_report(self,
                              task_data: Dict[str, Any],
                              result_data: Dict[str, Any],
                              trajectory: Optional[List[Dict]],
                              heuristic_results: List[MetricResult],
                              llm_results: Dict[str, LLMJudgmentResult],
                              aggregated_score: AggregatedScore,
                              evaluation_duration: float) -> Dict[str, Any]:
        """Build a detailed evaluation report."""
        report = {
            'task_info': task_data,
            'evaluation_summary': {
                'overall_score': aggregated_score.overall_score,
                'confidence': aggregated_score.confidence,
                'policy_used': aggregated_score.policy_used,
                'evaluation_duration_seconds': evaluation_duration,
                'timestamp': datetime.now().isoformat()
            },
            'original_result': result_data,
            'component_scores': {
                cs.component_name: {
                    'score': cs.score,
                    'confidence': cs.confidence,
                    'weight': cs.weight_used
                }
                for cs in aggregated_score.component_scores
            },
            'score_breakdown': aggregated_score.breakdown,
            'warnings': aggregated_score.warnings,
            'metadata': aggregated_score.metadata
        }
        
        # Add heuristic metrics details
        report['heuristic_metrics'] = [
            {
                'metric_id': r.metric_id,
                'type': r.metric_type.value,
                'score': r.score,
                'confidence': r.confidence,
                'explanation': r.explanation,
                'details': r.details
            }
            for r in heuristic_results
        ]
        
        # Add LLM judge details
        if llm_results:
            report['llm_judges'] = {}
            for judge_id, judge_result in llm_results.items():
                report['llm_judges'][judge_id] = {
                    'judgment_type': judge_result.judgment_type.value,
                    'score': judge_result.overall_score,
                    'confidence': judge_result.confidence,
                    'rationale': judge_result.rationale,
                    'criterion_scores': judge_result.criterion_scores,
                    'chain_of_thought': judge_result.chain_of_thought if hasattr(judge_result, 'chain_of_thought') else None,
                    'processing_time': judge_result.processing_time
                }
        
        # Add trajectory summary
        if trajectory:
            report['trajectory_summary'] = {
                'total_actions': len(trajectory),
                'action_types': list(set(
                    step.get('action', {}).get('action_type', 'unknown')
                    for step in trajectory
                )),
                'errors_encountered': sum(1 for step in trajectory if step.get('error'))
            }
        
        return report
    
    def batch_evaluate(self,
                      tasks: List[Dict[str, Any]],
                      output_dir: Optional[str] = None) -> List[Tuple[str, AggregatedScore, Dict]]:
        """
        Evaluate multiple tasks in batch.
        
        Args:
            tasks: List of task dictionaries, each containing:
                - task_id: str
                - task_instruction: str
                - result_data: Dict
                - trajectory: Optional[List[Dict]]
                - trajectory_path: Optional[str]
            output_dir: Optional directory to save individual reports
            
        Returns:
            List of (task_id, aggregated_score, detailed_report) tuples
        """
        results = []
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, task in enumerate(tasks):
            task_id = task.get('task_id', f'task_{i}')
            self.logger.info(f"Batch evaluation {i+1}/{len(tasks)}: {task_id}")
            
            try:
                aggregated_score, report = self.evaluate_task(
                    task_id=task_id,
                    task_instruction=task.get('task_instruction', ''),
                    result_data=task.get('result_data', {}),
                    trajectory=task.get('trajectory'),
                    trajectory_path=task.get('trajectory_path')
                )
                
                results.append((task_id, aggregated_score, report))
                
                # Save individual report if output_dir provided
                if output_dir:
                    report_path = Path(output_dir) / f"{task_id}_comprehensive_report.json"
                    with open(report_path, 'w') as f:
                        json.dump(report, f, indent=2)
                    self.logger.info(f"Saved report to {report_path}")
                
            except Exception as e:
                self.logger.error(f"Task {task_id} evaluation failed: {e}")
                # Add failed result
                failed_score = AggregatedScore(
                    overall_score=0.0,
                    confidence=0.0,
                    policy_used="failed",
                    warnings=[f"Evaluation failed: {e}"]
                )
                results.append((task_id, failed_score, {}))
        
        self.logger.info(f"Batch evaluation complete: {len(results)} tasks processed")
        return results
    
    def compare_policies(self,
                        task_id: str,
                        task_instruction: str,
                        result_data: Dict[str, Any],
                        trajectory: Optional[List[Dict]] = None,
                        policies: Optional[List[ScoringPolicy]] = None) -> Dict[str, AggregatedScore]:
        """
        Compare different scoring policies on the same task.
        
        Args:
            task_id: Task identifier
            task_instruction: Task instruction
            result_data: Result data
            trajectory: Optional trajectory
            policies: List of policies to compare (uses default 3 if None)
            
        Returns:
            Dictionary mapping policy names to aggregated scores
        """
        if policies is None:
            policies = [
                ScoringPolicy.create_conservative_policy(),
                ScoringPolicy.create_balanced_policy(),
                ScoringPolicy.create_ai_focused_policy()
            ]
        
        self.logger.info(f"Comparing {len(policies)} policies for task {task_id}")
        
        # Evaluate once to get component scores
        original_policy = self.scoring_aggregator.policy
        
        # Get component scores
        task_data = {'id': task_id, 'instruction': task_instruction}
        heuristic_results = self.heuristics_engine.evaluate_all(task_data, trajectory, result_data)
        heuristic_score = self.heuristics_engine.aggregate_metric_scores(heuristic_results)
        
        llm_scores = {}
        if self.enable_llm_judges and self.llm_judge_engine:
            llm_results = self.llm_judge_engine.evaluate_all(task_data, trajectory, result_data)
            llm_scores = {j_id: j_res.overall_score for j_id, j_res in llm_results.items()}
        
        deterministic_score = float(result_data.get('score', 0.0))
        
        # Compare policies
        comparison = self.scoring_aggregator.compare_policies(
            policies=policies,
            deterministic_score=deterministic_score,
            heuristic_score=heuristic_score,
            llm_scores=llm_scores
        )
        
        # Restore original policy
        self.scoring_aggregator.policy = original_policy
        
        return comparison
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get status and configuration of the framework."""
        return {
            'components': {
                'heuristics_engine': len(self.heuristics_engine.metrics),
                'llm_judge_engine': (
                    len(self.llm_judge_engine.judges)
                    if self.llm_judge_engine else 0
                ),
                'scoring_policy': self.scoring_aggregator.policy.name
            },
            'configuration': {
                'enable_llm_judges': self.enable_llm_judges,
                'policy_details': {
                    'name': self.scoring_aggregator.policy.name,
                    'method': self.scoring_aggregator.policy.aggregation_method.value,
                    'weights': {
                        'deterministic': self.scoring_aggregator.policy.component_weights.deterministic,
                        'heuristic': self.scoring_aggregator.policy.component_weights.heuristic,
                        'llm_judge': self.scoring_aggregator.policy.component_weights.llm_judge
                    }
                }
            },
            'available_policies': [
                'conservative', 'balanced', 'ai_focused', 'strict'
            ]
        }


# Convenience functions

def evaluate_task(task_id: str,
                 task_instruction: str,
                 result_data: Dict[str, Any],
                 trajectory: Optional[List[Dict]] = None,
                 trajectory_path: Optional[str] = None,
                 policy: str = "balanced",
                 enable_llm: bool = True) -> Tuple[float, Dict[str, Any]]:
    """
    Convenience function to evaluate a single task.
    
    Args:
        task_id: Task identifier
        task_instruction: Task instruction text
        result_data: Result dictionary with 'success' and 'score'
        trajectory: Optional execution trajectory
        trajectory_path: Optional path to trajectory JSON
        policy: Scoring policy ('conservative', 'balanced', 'ai_focused', 'strict')
        enable_llm: Whether to enable LLM judges
        
    Returns:
        Tuple of (overall_score, detailed_report_dict)
    """
    # Create framework with specified policy
    if policy == 'conservative':
        scoring_policy = ScoringPolicy.create_conservative_policy()
    elif policy == 'ai_focused':
        scoring_policy = ScoringPolicy.create_ai_focused_policy()
    elif policy == 'strict':
        scoring_policy = ScoringPolicy.create_strict_policy()
    else:
        scoring_policy = ScoringPolicy.create_balanced_policy()
    
    framework = ComprehensiveEvaluationFramework(
        scoring_policy=scoring_policy,
        enable_llm_judges=enable_llm
    )
    
    aggregated_score, report = framework.evaluate_task(
        task_id=task_id,
        task_instruction=task_instruction,
        result_data=result_data,
        trajectory=trajectory,
        trajectory_path=trajectory_path
    )
    
    return aggregated_score.overall_score, report


def batch_evaluate_tasks(tasks: List[Dict[str, Any]],
                        output_dir: str = "./evaluation_results",
                        policy: str = "balanced",
                        enable_llm: bool = True) -> List[Tuple[str, float, Dict]]:
    """
    Convenience function to batch evaluate multiple tasks.
    
    Args:
        tasks: List of task dictionaries (see batch_evaluate for structure)
        output_dir: Directory to save reports
        policy: Scoring policy to use
        enable_llm: Whether to enable LLM judges
        
    Returns:
        List of (task_id, score, report) tuples
    """
    if policy == 'conservative':
        scoring_policy = ScoringPolicy.create_conservative_policy()
    elif policy == 'ai_focused':
        scoring_policy = ScoringPolicy.create_ai_focused_policy()
    elif policy == 'strict':
        scoring_policy = ScoringPolicy.create_strict_policy()
    else:
        scoring_policy = ScoringPolicy.create_balanced_policy()
    
    framework = ComprehensiveEvaluationFramework(
        scoring_policy=scoring_policy,
        enable_llm_judges=enable_llm
    )
    
    results = framework.batch_evaluate(tasks, output_dir)
    
    # Convert to simpler format
    return [(task_id, score.overall_score, report) for task_id, score, report in results]

