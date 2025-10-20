"""
Heuristics & Metrics Engine - Behavioral analysis and performance metrics

Analyzes execution patterns, efficiency, coherence, and quality.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of heuristic metrics."""
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    BEHAVIORAL = "behavioral"
    PERFORMANCE = "performance"


@dataclass
class MetricResult:
    """Result from a single heuristic metric."""
    metric_id: str
    metric_type: MetricType
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    explanation: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure scores are valid."""
        self.score = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))


class BaseMetric(ABC):
    """Base class for all heuristic metrics."""
    
    def __init__(self, metric_id: str, metric_type: MetricType):
        self.metric_id = metric_id
        self.metric_type = metric_type
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    @abstractmethod
    def evaluate(self, task_data: Dict[str, Any], trajectory: Optional[List[Dict]],
                result: Dict[str, Any]) -> MetricResult:
        """Evaluate and return metric result."""
        pass


class TaskEfficiencyMetric(BaseMetric):
    """
    Measures task execution efficiency.
    
    Analyzes:
    - Number of actions taken vs expected
    - Execution time
    - Redundant actions
    - Action efficiency
    """
    
    def __init__(self):
        super().__init__("task_efficiency", MetricType.EFFICIENCY)
        
    def evaluate(self, task_data: Dict[str, Any], trajectory: Optional[List[Dict]],
                result: Dict[str, Any]) -> MetricResult:
        """Evaluate task efficiency."""
        if not trajectory:
            return MetricResult(
                metric_id=self.metric_id,
                metric_type=self.metric_type,
                score=0.5,
                confidence=0.3,
                explanation="No trajectory provided for efficiency analysis",
                details={}
            )
        
        num_actions = len(trajectory)
        execution_time = result.get('execution_time', 0)
        success = result.get('success', False)
        
        # Calculate efficiency score components
        scores = []
        details = {}
        
        # 1. Action count efficiency (fewer is often better)
        # Assume reasonable range is 5-50 actions
        if num_actions == 0:
            action_efficiency = 0.0
        elif num_actions <= 10:
            action_efficiency = 1.0
        elif num_actions <= 30:
            action_efficiency = 0.8
        elif num_actions <= 50:
            action_efficiency = 0.6
        else:
            action_efficiency = max(0.3, 1.0 - (num_actions - 50) / 100)
        
        scores.append(action_efficiency)
        details['action_count'] = num_actions
        details['action_efficiency'] = action_efficiency
        
        # 2. Time efficiency (faster is better, but not too fast)
        # Reasonable range: 10s - 300s
        if execution_time == 0:
            time_efficiency = 0.5
        elif execution_time < 10:
            time_efficiency = 0.7  # Too fast might indicate shortcuts
        elif execution_time <= 120:
            time_efficiency = 1.0
        elif execution_time <= 300:
            time_efficiency = 0.8
        else:
            time_efficiency = max(0.3, 1.0 - (execution_time - 300) / 600)
        
        scores.append(time_efficiency)
        details['execution_time'] = execution_time
        details['time_efficiency'] = time_efficiency
        
        # 3. Redundancy analysis
        redundancy_score = self._analyze_redundancy(trajectory)
        scores.append(redundancy_score)
        details['redundancy_score'] = redundancy_score
        
        # 4. Success factor - boost score if successful
        if success:
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # Calculate overall efficiency score
        overall_score = sum(scores) / len(scores)
        
        explanation = (
            f"Efficiency analysis: {num_actions} actions in {execution_time:.1f}s. "
            f"Action efficiency: {action_efficiency:.2f}, "
            f"Time efficiency: {time_efficiency:.2f}, "
            f"Redundancy: {redundancy_score:.2f}"
        )
        
        return MetricResult(
            metric_id=self.metric_id,
            metric_type=self.metric_type,
            score=overall_score,
            confidence=0.8,
            explanation=explanation,
            details=details
        )
    
    def _analyze_redundancy(self, trajectory: List[Dict]) -> float:
        """Analyze redundant actions in trajectory."""
        if len(trajectory) < 2:
            return 1.0
        
        # Check for repeated similar actions
        action_types = [step.get('action', {}).get('action_type', '') for step in trajectory]
        
        # Count consecutive duplicates
        consecutive_dupes = 0
        for i in range(len(action_types) - 1):
            if action_types[i] == action_types[i+1] and action_types[i]:
                consecutive_dupes += 1
        
        # Calculate redundancy score (lower redundancy = higher score)
        redundancy_ratio = consecutive_dupes / len(trajectory)
        redundancy_score = max(0.3, 1.0 - redundancy_ratio * 2)
        
        return redundancy_score


class ActionCoherenceMetric(BaseMetric):
    """
    Measures logical coherence of actions.
    
    Analyzes:
    - Logical flow of actions
    - Goal-directed behavior
    - Context awareness
    """
    
    def __init__(self):
        super().__init__("action_coherence", MetricType.BEHAVIORAL)
        
    def evaluate(self, task_data: Dict[str, Any], trajectory: Optional[List[Dict]],
                result: Dict[str, Any]) -> MetricResult:
        """Evaluate action coherence."""
        if not trajectory or len(trajectory) < 2:
            return MetricResult(
                metric_id=self.metric_id,
                metric_type=self.metric_type,
                score=0.5,
                confidence=0.3,
                explanation="Insufficient trajectory data for coherence analysis",
                details={}
            )
        
        scores = []
        details = {}
        
        # 1. Action diversity score (should have variety)
        action_types = [step.get('action', {}).get('action_type', '') for step in trajectory]
        unique_actions = len(set(action_types))
        total_actions = len(action_types)
        
        diversity_ratio = unique_actions / total_actions if total_actions > 0 else 0
        diversity_score = min(1.0, diversity_ratio * 2)  # 50% diversity = perfect
        scores.append(diversity_score)
        details['action_diversity'] = diversity_ratio
        
        # 2. Progressive complexity (actions should build on each other)
        progression_score = self._analyze_progression(trajectory)
        scores.append(progression_score)
        details['progression_score'] = progression_score
        
        # 3. Error recovery patterns
        errors_present = any(step.get('error') for step in trajectory)
        if errors_present:
            recovery_score = self._analyze_error_recovery(trajectory)
            scores.append(recovery_score)
            details['error_recovery'] = recovery_score
        else:
            details['error_recovery'] = "no_errors"
        
        overall_score = sum(scores) / len(scores)
        
        explanation = (
            f"Coherence analysis: {unique_actions} unique action types from {total_actions} total. "
            f"Diversity: {diversity_score:.2f}, Progression: {progression_score:.2f}"
        )
        
        return MetricResult(
            metric_id=self.metric_id,
            metric_type=self.metric_type,
            score=overall_score,
            confidence=0.7,
            explanation=explanation,
            details=details
        )
    
    def _analyze_progression(self, trajectory: List[Dict]) -> float:
        """Analyze if actions show logical progression."""
        # Simple heuristic: later actions should be different from early ones
        # This indicates progression rather than loops
        
        if len(trajectory) < 4:
            return 0.7
        
        first_quarter = trajectory[:len(trajectory)//4]
        last_quarter = trajectory[-len(trajectory)//4:]
        
        first_actions = {step.get('action', {}).get('action_type', '') for step in first_quarter}
        last_actions = {step.get('action', {}).get('action_type', '') for step in last_quarter}
        
        # If actions are evolving, there should be some difference
        if not first_actions or not last_actions:
            return 0.5
        
        overlap = len(first_actions & last_actions)
        total = len(first_actions | last_actions)
        
        # 50% overlap is good (shows progress while maintaining consistency)
        if total == 0:
            return 0.5
        
        overlap_ratio = overlap / total
        # Ideal is around 0.5 (some continuity, some progress)
        progression_score = 1.0 - abs(0.5 - overlap_ratio)
        
        return progression_score
    
    def _analyze_error_recovery(self, trajectory: List[Dict]) -> float:
        """Analyze how well errors were handled."""
        errors = []
        for i, step in enumerate(trajectory):
            if step.get('error'):
                errors.append(i)
        
        if not errors:
            return 1.0
        
        # Check if actions after errors show recovery attempts
        recovery_attempts = 0
        for error_idx in errors:
            if error_idx < len(trajectory) - 1:
                # Look at next action after error
                next_step = trajectory[error_idx + 1]
                # If action type changes, it's a recovery attempt
                current_action = trajectory[error_idx].get('action', {}).get('action_type', '')
                next_action = next_step.get('action', {}).get('action_type', '')
                if next_action != current_action:
                    recovery_attempts += 1
        
        recovery_ratio = recovery_attempts / len(errors) if errors else 0
        return min(1.0, recovery_ratio + 0.3)  # At least 0.3 for trying


class CompletionQualityMetric(BaseMetric):
    """
    Measures quality of task completion.
    
    Analyzes:
    - Task completion status
    - Result accuracy
    - Completeness of solution
    """
    
    def __init__(self):
        super().__init__("completion_quality", MetricType.QUALITY)
        
    def evaluate(self, task_data: Dict[str, Any], trajectory: Optional[List[Dict]],
                result: Dict[str, Any]) -> MetricResult:
        """Evaluate completion quality."""
        success = result.get('success', False)
        score = result.get('score', 0.0)
        
        details = {
            'success': success,
            'raw_score': score,
        }
        
        # Base quality on result score
        quality_score = float(score)
        
        # Adjust based on trajectory if available
        if trajectory:
            num_actions = len(trajectory)
            errors_count = sum(1 for step in trajectory if step.get('error'))
            
            # Penalize for errors
            if errors_count > 0:
                error_penalty = min(0.3, errors_count * 0.05)
                quality_score = max(0.0, quality_score - error_penalty)
                details['error_penalty'] = error_penalty
                details['errors_count'] = errors_count
            
            # Bonus for clean execution (no errors and reasonable action count)
            if errors_count == 0 and 5 <= num_actions <= 30:
                quality_score = min(1.0, quality_score + 0.1)
                details['clean_execution_bonus'] = 0.1
        
        explanation = (
            f"Quality analysis: Task {'succeeded' if success else 'failed'} "
            f"with score {score:.2f}. "
        )
        
        if trajectory:
            explanation += f"Executed with {len(trajectory)} actions."
        
        confidence = 0.9 if success else 0.7
        
        return MetricResult(
            metric_id=self.metric_id,
            metric_type=self.metric_type,
            score=quality_score,
            confidence=confidence,
            explanation=explanation,
            details=details
        )


class ResourceUtilizationMetric(BaseMetric):
    """
    Measures resource utilization efficiency.
    
    Analyzes:
    - API calls made
    - Time per action
    - Resource efficiency
    """
    
    def __init__(self):
        super().__init__("resource_utilization", MetricType.PERFORMANCE)
        
    def evaluate(self, task_data: Dict[str, Any], trajectory: Optional[List[Dict]],
                result: Dict[str, Any]) -> MetricResult:
        """Evaluate resource utilization."""
        if not trajectory:
            return MetricResult(
                metric_id=self.metric_id,
                metric_type=self.metric_type,
                score=0.5,
                confidence=0.3,
                explanation="No trajectory data for resource analysis",
                details={}
            )
        
        num_actions = len(trajectory)
        execution_time = result.get('execution_time', 0)
        
        details = {}
        
        # Calculate time per action
        if num_actions > 0 and execution_time > 0:
            time_per_action = execution_time / num_actions
            details['time_per_action'] = time_per_action
            
            # Ideal range: 1-5 seconds per action
            if 1 <= time_per_action <= 5:
                time_efficiency = 1.0
            elif time_per_action < 1:
                # Too fast might indicate skipped validations
                time_efficiency = 0.7 + time_per_action * 0.3
            else:
                # Too slow
                time_efficiency = max(0.3, 1.0 - (time_per_action - 5) / 20)
        else:
            time_efficiency = 0.5
            details['time_per_action'] = 0
        
        details['time_efficiency'] = time_efficiency
        
        # Overall resource score
        resource_score = time_efficiency
        
        explanation = (
            f"Resource utilization: {num_actions} actions in {execution_time:.1f}s "
            f"({time_per_action:.2f}s per action on average)"
        )
        
        return MetricResult(
            metric_id=self.metric_id,
            metric_type=self.metric_type,
            score=resource_score,
            confidence=0.7,
            explanation=explanation,
            details=details
        )


class HeuristicsEngine:
    """Engine for running heuristic metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, BaseMetric] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Register default metrics
        self._register_default_metrics()
        
    def _register_default_metrics(self):
        """Register default heuristic metrics."""
        default_metrics = [
            TaskEfficiencyMetric(),
            ActionCoherenceMetric(),
            CompletionQualityMetric(),
            ResourceUtilizationMetric()
        ]
        
        for metric in default_metrics:
            self.add_metric(metric)
        
    def add_metric(self, metric: BaseMetric):
        """Add a metric to the engine."""
        self.metrics[metric.metric_id] = metric
        self.logger.info(f"Added metric: {metric.metric_id}")
        
    def remove_metric(self, metric_id: str):
        """Remove a metric from the engine."""
        if metric_id in self.metrics:
            del self.metrics[metric_id]
            self.logger.info(f"Removed metric: {metric_id}")
            
    def evaluate_all(self, task_data: Dict[str, Any], 
                    trajectory: Optional[List[Dict]],
                    result: Dict[str, Any],
                    metric_ids: Optional[List[str]] = None) -> List[MetricResult]:
        """
        Evaluate using all (or specified) metrics.
        
        Args:
            task_data: Task information
            trajectory: Optional execution trajectory
            result: Evaluation result
            metric_ids: Optional list of specific metric IDs to use
            
        Returns:
            List of metric results
        """
        metrics_to_run = metric_ids or list(self.metrics.keys())
        results = []
        
        for metric_id in metrics_to_run:
            if metric_id not in self.metrics:
                self.logger.warning(f"Metric {metric_id} not found, skipping")
                continue
                
            try:
                self.logger.info(f"Running metric: {metric_id}")
                result_obj = self.metrics[metric_id].evaluate(task_data, trajectory, result)
                results.append(result_obj)
                self.logger.info(
                    f"Metric {metric_id} completed: score={result_obj.score:.3f}, "
                    f"confidence={result_obj.confidence:.3f}"
                )
            except Exception as e:
                self.logger.error(f"Metric {metric_id} failed: {e}")
                # Add failed result
                results.append(MetricResult(
                    metric_id=metric_id,
                    metric_type=MetricType.BEHAVIORAL,
                    score=0.0,
                    confidence=0.0,
                    explanation=f"Metric execution failed: {e}",
                    details={}
                ))
        
        return results
    
    def aggregate_metric_scores(self, results: List[MetricResult]) -> float:
        """
        Aggregate multiple metric results into a single score.
        
        Args:
            results: List of metric results
            
        Returns:
            Aggregated score (0.0 to 1.0)
        """
        if not results:
            return 0.0
        
        # Weight by confidence
        weighted_sum = sum(r.score * r.confidence for r in results)
        total_confidence = sum(r.confidence for r in results)
        
        if total_confidence == 0:
            return sum(r.score for r in results) / len(results)
        
        return weighted_sum / total_confidence

