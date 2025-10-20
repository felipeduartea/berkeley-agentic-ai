"""
Scoring Aggregator - Combines evaluation results using configurable policies

Provides flexible scoring policies for aggregating deterministic, heuristic,
and LLM judge results into final scores.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating scores."""
    WEIGHTED_AVERAGE = "weighted_average"
    MINIMUM = "minimum"  # Pessimistic: lowest score
    MAXIMUM = "maximum"  # Optimistic: highest score
    MEDIAN = "median"
    PRODUCT = "product"  # Multiplicative (all must be good)


@dataclass
class ComponentWeight:
    """Weights for each evaluation component."""
    deterministic: float = 0.33
    heuristic: float = 0.33
    llm_judge: float = 0.34
    
    def __post_init__(self):
        """Validate weights sum to approximately 1.0."""
        total = self.deterministic + self.heuristic + self.llm_judge
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Component weights sum to {total}, not 1.0. Normalizing...")
            self.deterministic /= total
            self.heuristic /= total
            self.llm_judge /= total


@dataclass
class ScoringPolicy:
    """Policy for aggregating evaluation scores."""
    name: str
    aggregation_method: AggregationMethod
    component_weights: ComponentWeight
    confidence_threshold: float = 0.5  # Minimum confidence to accept score
    adaptive_weighting: bool = False  # Adjust weights based on confidence
    deterministic_override: bool = False  # Deterministic failure = overall failure
    llm_override_enabled: bool = False  # Allow LLM to override low scores
    
    @staticmethod
    def create_conservative_policy() -> 'ScoringPolicy':
        """
        Conservative policy - heavily weights deterministic results.
        
        Best for: High-stakes evaluation, regulatory compliance
        Features: Deterministic failures override all, limited LLM influence
        """
        return ScoringPolicy(
            name="conservative",
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            component_weights=ComponentWeight(
                deterministic=0.60,
                heuristic=0.30,
                llm_judge=0.10
            ),
            confidence_threshold=0.7,
            adaptive_weighting=False,
            deterministic_override=True,
            llm_override_enabled=False
        )
    
    @staticmethod
    def create_balanced_policy() -> 'ScoringPolicy':
        """
        Balanced policy - equal weighting across components.
        
        Best for: General purpose evaluation, development testing
        Features: Equal weights, moderate LLM influence
        """
        return ScoringPolicy(
            name="balanced",
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            component_weights=ComponentWeight(
                deterministic=0.33,
                heuristic=0.33,
                llm_judge=0.34
            ),
            confidence_threshold=0.5,
            adaptive_weighting=True,
            deterministic_override=False,
            llm_override_enabled=True
        )
    
    @staticmethod
    def create_ai_focused_policy() -> 'ScoringPolicy':
        """
        AI-focused policy - prioritizes LLM judge results.
        
        Best for: Research, nuanced evaluation, human-like assessment
        Features: High LLM weight, adaptive weighting, override enabled
        """
        return ScoringPolicy(
            name="ai_focused",
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            component_weights=ComponentWeight(
                deterministic=0.20,
                heuristic=0.20,
                llm_judge=0.60
            ),
            confidence_threshold=0.4,
            adaptive_weighting=True,
            deterministic_override=False,
            llm_override_enabled=True
        )
    
    @staticmethod
    def create_strict_policy() -> 'ScoringPolicy':
        """
        Strict policy - requires all components to score well.
        
        Best for: Quality control, acceptance testing
        Features: Product aggregation (all must pass), high threshold
        """
        return ScoringPolicy(
            name="strict",
            aggregation_method=AggregationMethod.PRODUCT,
            component_weights=ComponentWeight(
                deterministic=0.33,
                heuristic=0.33,
                llm_judge=0.34
            ),
            confidence_threshold=0.8,
            adaptive_weighting=False,
            deterministic_override=True,
            llm_override_enabled=False
        )


@dataclass
class ComponentScore:
    """Score from a single evaluation component."""
    component_name: str
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    weight_used: float = 1.0


@dataclass
class AggregatedScore:
    """Final aggregated score from all evaluation components."""
    overall_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    policy_used: str
    component_scores: List[ComponentScore] = field(default_factory=list)
    breakdown: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure scores are valid."""
        self.overall_score = max(0.0, min(1.0, self.overall_score))
        self.confidence = max(0.0, min(1.0, self.confidence))


class ScoringAggregator:
    """Aggregates scores from multiple evaluation components."""
    
    def __init__(self, policy: ScoringPolicy):
        self.policy = policy
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def update_policy(self, policy: ScoringPolicy):
        """Update the scoring policy."""
        self.policy = policy
        self.logger.info(f"Updated scoring policy to: {policy.name}")
        
    def aggregate_evaluation(self,
                           deterministic_score: Optional[float] = None,
                           deterministic_confidence: float = 1.0,
                           heuristic_score: Optional[float] = None,
                           heuristic_confidence: float = 0.7,
                           llm_scores: Optional[Dict[str, float]] = None,
                           llm_confidences: Optional[Dict[str, float]] = None,
                           deterministic_details: Optional[Dict] = None,
                           heuristic_details: Optional[Dict] = None,
                           llm_details: Optional[Dict] = None) -> AggregatedScore:
        """
        Aggregate scores from all evaluation components.
        
        Args:
            deterministic_score: Score from deterministic checks (0-1)
            deterministic_confidence: Confidence in deterministic score
            heuristic_score: Score from heuristic metrics (0-1)
            heuristic_confidence: Confidence in heuristic score
            llm_scores: Scores from LLM judges {judge_id: score}
            llm_confidences: Confidences from LLM judges {judge_id: confidence}
            deterministic_details: Details from deterministic evaluation
            heuristic_details: Details from heuristic evaluation
            llm_details: Details from LLM judges
            
        Returns:
            AggregatedScore with overall score and breakdown
        """
        component_scores = []
        warnings = []
        
        # Collect component scores
        if deterministic_score is not None:
            component_scores.append(ComponentScore(
                component_name="deterministic",
                score=deterministic_score,
                confidence=deterministic_confidence,
                details=deterministic_details or {},
                weight_used=self.policy.component_weights.deterministic
            ))
        else:
            warnings.append("No deterministic score provided")
        
        if heuristic_score is not None:
            component_scores.append(ComponentScore(
                component_name="heuristic",
                score=heuristic_score,
                confidence=heuristic_confidence,
                details=heuristic_details or {},
                weight_used=self.policy.component_weights.heuristic
            ))
        else:
            warnings.append("No heuristic score provided")
        
        # Aggregate LLM scores
        if llm_scores and len(llm_scores) > 0:
            avg_llm_score = sum(llm_scores.values()) / len(llm_scores)
            avg_llm_confidence = (
                sum(llm_confidences.values()) / len(llm_confidences)
                if llm_confidences else 0.7
            )
            component_scores.append(ComponentScore(
                component_name="llm_judge",
                score=avg_llm_score,
                confidence=avg_llm_confidence,
                details=llm_details or {},
                weight_used=self.policy.component_weights.llm_judge
            ))
        else:
            warnings.append("No LLM judge scores provided")
        
        # Apply deterministic override if enabled
        if self.policy.deterministic_override and deterministic_score is not None:
            if deterministic_score < 0.5:
                return AggregatedScore(
                    overall_score=deterministic_score,
                    confidence=deterministic_confidence,
                    policy_used=self.policy.name,
                    component_scores=component_scores,
                    breakdown={"deterministic_override": True},
                    warnings=warnings + ["Deterministic failure overrode all other scores"],
                    metadata={"override_applied": True}
                )
        
        # Adjust weights based on confidence if adaptive weighting enabled
        if self.policy.adaptive_weighting:
            component_scores = self._apply_adaptive_weighting(component_scores)
        
        # Filter by confidence threshold
        valid_components = [
            cs for cs in component_scores
            if cs.confidence >= self.policy.confidence_threshold
        ]
        
        if not valid_components:
            warnings.append(
                f"No components met confidence threshold {self.policy.confidence_threshold}"
            )
            # Use all components anyway but with warning
            valid_components = component_scores
        
        # Aggregate scores based on policy method
        overall_score = self._aggregate_scores(valid_components)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_confidence(valid_components)
        
        # Apply LLM override if enabled
        if self.policy.llm_override_enabled:
            overall_score = self._apply_llm_override(
                overall_score, component_scores, warnings
            )
        
        # Create breakdown
        breakdown = {
            cs.component_name: cs.score * cs.weight_used
            for cs in component_scores
        }
        breakdown["aggregation_method"] = self.policy.aggregation_method.value
        
        return AggregatedScore(
            overall_score=overall_score,
            confidence=overall_confidence,
            policy_used=self.policy.name,
            component_scores=component_scores,
            breakdown=breakdown,
            warnings=warnings,
            metadata={
                "adaptive_weighting_applied": self.policy.adaptive_weighting,
                "components_evaluated": len(component_scores),
                "components_used": len(valid_components)
            }
        )
    
    def _apply_adaptive_weighting(self, components: List[ComponentScore]) -> List[ComponentScore]:
        """Adjust weights based on component confidence."""
        if not components:
            return components
        
        # Calculate confidence-adjusted weights
        total_weighted_confidence = sum(cs.confidence * cs.weight_used for cs in components)
        
        if total_weighted_confidence == 0:
            return components
        
        # Redistribute weights proportionally to confidence
        for cs in components:
            adjusted_weight = (cs.confidence * cs.weight_used) / total_weighted_confidence
            cs.weight_used = adjusted_weight
        
        self.logger.debug("Applied adaptive weighting based on confidence")
        return components
    
    def _aggregate_scores(self, components: List[ComponentScore]) -> float:
        """Aggregate component scores using policy method."""
        if not components:
            return 0.0
        
        scores = [cs.score for cs in components]
        
        if self.policy.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
            total_weight = sum(cs.weight_used for cs in components)
            if total_weight == 0:
                return sum(scores) / len(scores)
            weighted_sum = sum(cs.score * cs.weight_used for cs in components)
            return weighted_sum / total_weight
        
        elif self.policy.aggregation_method == AggregationMethod.MINIMUM:
            return min(scores)
        
        elif self.policy.aggregation_method == AggregationMethod.MAXIMUM:
            return max(scores)
        
        elif self.policy.aggregation_method == AggregationMethod.MEDIAN:
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            if n % 2 == 0:
                return (sorted_scores[n//2-1] + sorted_scores[n//2]) / 2
            else:
                return sorted_scores[n//2]
        
        elif self.policy.aggregation_method == AggregationMethod.PRODUCT:
            result = 1.0
            for score in scores:
                result *= score
            return result
        
        else:
            # Default to weighted average
            total_weight = sum(cs.weight_used for cs in components)
            if total_weight == 0:
                return sum(scores) / len(scores)
            return sum(cs.score * cs.weight_used for cs in components) / total_weight
    
    def _calculate_confidence(self, components: List[ComponentScore]) -> float:
        """Calculate overall confidence from component confidences."""
        if not components:
            return 0.0
        
        # Weighted average of confidences
        total_weight = sum(cs.weight_used for cs in components)
        if total_weight == 0:
            return sum(cs.confidence for cs in components) / len(components)
        
        weighted_conf = sum(cs.confidence * cs.weight_used for cs in components)
        return weighted_conf / total_weight
    
    def _apply_llm_override(self, current_score: float, 
                           components: List[ComponentScore],
                           warnings: List[str]) -> float:
        """Apply LLM override if LLM judges scored significantly higher."""
        llm_components = [cs for cs in components if cs.component_name == "llm_judge"]
        
        if not llm_components:
            return current_score
        
        llm_score = llm_components[0].score
        llm_confidence = llm_components[0].confidence
        
        # If LLM score is significantly higher and has high confidence, allow partial override
        if llm_score > current_score + 0.2 and llm_confidence > 0.7:
            # Blend current score with LLM score (30% LLM influence)
            override_score = current_score * 0.7 + llm_score * 0.3
            warnings.append(
                f"LLM override applied: boosted score from {current_score:.3f} "
                f"to {override_score:.3f}"
            )
            return override_score
        
        return current_score
    
    def compare_policies(self,
                        policies: List[ScoringPolicy],
                        deterministic_score: Optional[float] = None,
                        heuristic_score: Optional[float] = None,
                        llm_scores: Optional[Dict[str, float]] = None,
                        **kwargs) -> Dict[str, AggregatedScore]:
        """
        Compare different policies on the same evaluation data.
        
        Args:
            policies: List of policies to compare
            deterministic_score: Deterministic evaluation score
            heuristic_score: Heuristic evaluation score
            llm_scores: LLM judge scores
            **kwargs: Additional arguments for aggregate_evaluation
            
        Returns:
            Dictionary mapping policy names to aggregated scores
        """
        comparison = {}
        original_policy = self.policy
        
        for policy in policies:
            self.policy = policy
            aggregated = self.aggregate_evaluation(
                deterministic_score=deterministic_score,
                heuristic_score=heuristic_score,
                llm_scores=llm_scores,
                **kwargs
            )
            comparison[policy.name] = aggregated
        
        # Restore original policy
        self.policy = original_policy
        
        self.logger.info(f"Compared {len(policies)} policies")
        return comparison

