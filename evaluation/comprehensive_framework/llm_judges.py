"""
LLM Judge Engine - AI-powered evaluation with G-Eval, DeepEval, and Rubric-based scoring

Implements multiple LLM judging approaches for sophisticated evaluation.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import re
import time

logger = logging.getLogger(__name__)

# Try to import OpenAI, but make it optional
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with: pip install openai")


class JudgmentType(Enum):
    """Types of LLM judgment approaches."""
    G_EVAL = "g_eval"              # G-Eval with chain-of-thought
    DEEPEVAL = "deepeval"          # DeepEval style with specific metrics
    RUBRIC_BASED = "rubric_based"  # Custom rubric evaluation
    HOLISTIC = "holistic"          # Overall holistic assessment


@dataclass
class RubricCriterion:
    """Individual criterion in an evaluation rubric."""
    name: str
    description: str
    weight: float = 1.0
    levels: Dict[int, str] = field(default_factory=dict)  # score -> description
    examples: List[str] = field(default_factory=list)


@dataclass
class EvaluationRubric:
    """Complete evaluation rubric with multiple criteria."""
    name: str
    description: str
    criteria: List[RubricCriterion] = field(default_factory=list)
    scale: Tuple[int, int] = (1, 5)  # min, max score
    
    def add_criterion(self, criterion: RubricCriterion) -> None:
        """Add a criterion to the rubric."""
        self.criteria.append(criterion)
    
    def total_weight(self) -> float:
        """Calculate total weight of all criteria."""
        return sum(c.weight for c in self.criteria)


@dataclass
class LLMJudgmentResult:
    """Result of LLM-based judgment."""
    judgment_id: str
    judgment_type: JudgmentType
    overall_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    rationale: str
    criterion_scores: Dict[str, float] = field(default_factory=dict)
    raw_response: str = ""
    model_used: str = ""
    processing_time: float = 0.0
    chain_of_thought: str = ""  # For G-Eval
    
    def __post_init__(self):
        """Ensure scores are between 0 and 1."""
        self.overall_score = max(0.0, min(1.0, self.overall_score))
        self.confidence = max(0.0, min(1.0, self.confidence))


class BaseLLMJudge(ABC):
    """Base class for all LLM judges."""
    
    def __init__(self, judge_id: str, model_name: str = "gpt-4o", 
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.judge_id = judge_id
        self.model_name = model_name
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url
        self.client = None
        self._setup_client()
        
    def _setup_client(self):
        """Setup the OpenAI client."""
        if not OPENAI_AVAILABLE:
            logger.warning(f"OpenAI not available for judge {self.judge_id}")
            return
            
        try:
            client_kwargs = {}
            if self.api_key:
                client_kwargs['api_key'] = self.api_key
            if self.base_url:
                client_kwargs['base_url'] = self.base_url
                
            self.client = openai.OpenAI(**client_kwargs)
        except Exception as e:
            logger.warning(f"Failed to setup OpenAI client for {self.judge_id}: {e}")
            
    @abstractmethod
    def evaluate(self, task_data: Dict[str, Any], trajectory: Optional[List[Dict]], 
                result: Dict[str, Any]) -> LLMJudgmentResult:
        """Evaluate the task execution using this judge."""
        pass
        
    def _make_llm_call(self, prompt: str, system_prompt: str = None) -> str:
        """Make an LLM API call."""
        if not self.client:
            raise RuntimeError(f"OpenAI client not available for {self.judge_id}")
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=2000
            )
            processing_time = time.time() - start_time
            
            content = response.choices[0].message.content
            logger.info(f"LLM call for {self.judge_id} completed in {processing_time:.2f}s")
            return content
            
        except Exception as e:
            logger.error(f"LLM API call failed for {self.judge_id}: {e}")
            raise


class GEvalJudge(BaseLLMJudge):
    """
    G-Eval style judge with chain-of-thought evaluation.
    
    G-Eval uses a multi-step chain-of-thought process:
    1. Generate evaluation criteria
    2. Step-by-step reasoning through the task
    3. Final scoring with detailed rationale
    """
    
    def __init__(self, judge_id: str = "g_eval", model_name: str = "gpt-4o",
                 criteria: Optional[List[str]] = None):
        super().__init__(judge_id, model_name)
        self.criteria = criteria or [
            "Task Completion: Was the task fully completed?",
            "Correctness: Are the results accurate and correct?",
            "Efficiency: Was the task completed efficiently?",
            "Process Quality: Was the approach logical and well-structured?"
        ]
        
    def evaluate(self, task_data: Dict[str, Any], trajectory: Optional[List[Dict]], 
                result: Dict[str, Any]) -> LLMJudgmentResult:
        """Evaluate using G-Eval chain-of-thought methodology."""
        start_time = time.time()
        
        try:
            # Step 1: Generate task-specific evaluation steps
            cot_prompt = self._create_cot_prompt(task_data, trajectory, result)
            
            system_prompt = """You are an expert evaluator using the G-Eval methodology.
Think step-by-step through your evaluation process. For each criterion, provide detailed reasoning before scoring.
Finally, provide an overall score from 0.0 to 1.0."""
            
            response = self._make_llm_call(cot_prompt, system_prompt)
            
            # Parse the response
            parsed_result = self._parse_geval_response(response)
            parsed_result.processing_time = time.time() - start_time
            parsed_result.model_used = self.model_name
            parsed_result.raw_response = response
            parsed_result.judgment_id = self.judge_id
            parsed_result.judgment_type = JudgmentType.G_EVAL
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"G-Eval judgment failed: {e}")
            return LLMJudgmentResult(
                judgment_id=self.judge_id,
                judgment_type=JudgmentType.G_EVAL,
                overall_score=0.0,
                confidence=0.3,
                rationale=f"Evaluation failed: {e}",
                processing_time=time.time() - start_time
            )
    
    def _create_cot_prompt(self, task_data: Dict[str, Any], 
                          trajectory: Optional[List[Dict]], 
                          result: Dict[str, Any]) -> str:
        """Create chain-of-thought prompt for G-Eval."""
        prompt_parts = [
            "# Task Evaluation using G-Eval Methodology\n",
            "## Task Information",
            f"**Task ID:** {task_data.get('id', 'unknown')}",
            f"**Task Instruction:** {task_data.get('instruction', 'No instruction provided')}\n",
            "## Execution Results",
            f"**Success Status:** {result.get('success', False)}",
            f"**Score:** {result.get('score', 0.0)}",
        ]
        
        if trajectory and len(trajectory) > 0:
            prompt_parts.extend([
                f"**Actions Taken:** {len(trajectory)} steps",
                "\n### Sample Actions (first 5):"
            ])
            for i, step in enumerate(trajectory[:5]):
                action_str = json.dumps(step.get('action', {}), indent=2)
                prompt_parts.append(f"{i+1}. {action_str}")
        
        if result.get('details'):
            prompt_parts.append(f"\n**Result Details:** {json.dumps(result['details'], indent=2)}")
        
        prompt_parts.extend([
            "\n## Evaluation Criteria",
            "Please evaluate based on the following criteria:\n"
        ])
        
        for i, criterion in enumerate(self.criteria, 1):
            prompt_parts.append(f"{i}. {criterion}")
        
        prompt_parts.extend([
            "\n## Instructions",
            "1. For EACH criterion above, provide:",
            "   - Your chain-of-thought reasoning (2-3 sentences)",
            "   - A score from 0.0 to 1.0",
            "",
            "2. Then provide:",
            "   - An overall aggregated score (0.0 to 1.0)",
            "   - Your confidence in this evaluation (0.0 to 1.0)",
            "   - A final summary rationale",
            "",
            "Format your response as:",
            "```",
            "CRITERION 1:",
            "Reasoning: [your reasoning]",
            "Score: [0.0-1.0]",
            "",
            "CRITERION 2:",
            "Reasoning: [your reasoning]",
            "Score: [0.0-1.0]",
            "",
            "...",
            "",
            "OVERALL SCORE: [0.0-1.0]",
            "CONFIDENCE: [0.0-1.0]",
            "RATIONALE: [final summary]",
            "```"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_geval_response(self, response: str) -> LLMJudgmentResult:
        """Parse G-Eval response into structured result."""
        criterion_scores = {}
        overall_score = 0.5
        confidence = 0.7
        rationale = ""
        chain_of_thought = response  # Store full response as COT
        
        # Extract criterion scores
        criterion_pattern = r"CRITERION\s+\d+:.*?Reasoning:\s*(.*?)(?:Score:|$)"
        score_pattern = r"Score:\s*([\d.]+)"
        
        criteria_matches = re.findall(criterion_pattern, response, re.DOTALL | re.IGNORECASE)
        score_matches = re.findall(score_pattern, response, re.IGNORECASE)
        
        for i, score_str in enumerate(score_matches[:-1] if len(score_matches) > 1 else []):
            try:
                score = float(score_str)
                criterion_scores[f"criterion_{i+1}"] = min(1.0, max(0.0, score))
            except ValueError:
                pass
        
        # Extract overall score
        overall_match = re.search(r"OVERALL\s+SCORE:\s*([\d.]+)", response, re.IGNORECASE)
        if overall_match:
            try:
                overall_score = float(overall_match.group(1))
            except ValueError:
                pass
        
        # Extract confidence
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                pass
        
        # Extract rationale
        rat_match = re.search(r"RATIONALE:\s*(.+?)(?:\n\n|$)", response, re.DOTALL | re.IGNORECASE)
        if rat_match:
            rationale = rat_match.group(1).strip()
        else:
            rationale = "See chain-of-thought reasoning above"
        
        return LLMJudgmentResult(
            judgment_id=self.judge_id,
            judgment_type=JudgmentType.G_EVAL,
            overall_score=overall_score,
            confidence=confidence,
            rationale=rationale,
            criterion_scores=criterion_scores,
            chain_of_thought=chain_of_thought
        )


class DeepEvalJudge(BaseLLMJudge):
    """
    DeepEval style judge with specific evaluation metrics.
    
    Evaluates on predefined metrics like correctness, relevance, 
    coherence, efficiency, and completeness.
    """
    
    def __init__(self, judge_id: str = "deepeval", model_name: str = "gpt-4o"):
        super().__init__(judge_id, model_name)
        self.evaluation_metrics = {
            "correctness": "How correct and accurate are the results?",
            "completeness": "How complete is the solution?",
            "efficiency": "How efficient was the execution?",
            "coherence": "How logical and coherent was the approach?",
            "error_handling": "How well were errors handled?"
        }
        
    def evaluate(self, task_data: Dict[str, Any], trajectory: Optional[List[Dict]], 
                result: Dict[str, Any]) -> LLMJudgmentResult:
        """Evaluate using DeepEval methodology."""
        start_time = time.time()
        
        try:
            prompt = self._create_deepeval_prompt(task_data, trajectory, result)
            
            system_prompt = """You are an expert evaluator using the DeepEval methodology.
Analyze the task execution and provide scores for each metric on a scale of 0.0 to 1.0.
Be objective, thorough, and provide clear reasoning for your scores."""
            
            response = self._make_llm_call(prompt, system_prompt)
            parsed_result = self._parse_deepeval_response(response)
            parsed_result.processing_time = time.time() - start_time
            parsed_result.model_used = self.model_name
            parsed_result.raw_response = response
            parsed_result.judgment_id = self.judge_id
            parsed_result.judgment_type = JudgmentType.DEEPEVAL
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"DeepEval judgment failed: {e}")
            return LLMJudgmentResult(
                judgment_id=self.judge_id,
                judgment_type=JudgmentType.DEEPEVAL,
                overall_score=0.0,
                confidence=0.3,
                rationale=f"Evaluation failed: {e}",
                processing_time=time.time() - start_time
            )
    
    def _create_deepeval_prompt(self, task_data: Dict[str, Any], 
                               trajectory: Optional[List[Dict]], 
                               result: Dict[str, Any]) -> str:
        """Create DeepEval prompt."""
        prompt_parts = [
            "# Task Evaluation using DeepEval Metrics\n",
            "## Task",
            f"**ID:** {task_data.get('id', 'unknown')}",
            f"**Instruction:** {task_data.get('instruction', 'No instruction')}\n",
            "## Results",
            f"**Success:** {result.get('success', False)}",
            f"**Score:** {result.get('score', 0.0)}",
        ]
        
        if trajectory:
            prompt_parts.append(f"**Steps:** {len(trajectory)}")
        
        prompt_parts.extend([
            "\n## Evaluation Metrics",
            "Rate each metric from 0.0 (poor) to 1.0 (excellent):\n"
        ])
        
        for metric, description in self.evaluation_metrics.items():
            prompt_parts.append(f"**{metric.upper()}:** {description}")
        
        prompt_parts.extend([
            "\nProvide your evaluation in this format:",
            "```",
            "CORRECTNESS: [score] - [brief justification]",
            "COMPLETENESS: [score] - [brief justification]",
            "EFFICIENCY: [score] - [brief justification]",
            "COHERENCE: [score] - [brief justification]",
            "ERROR_HANDLING: [score] - [brief justification]",
            "",
            "OVERALL: [weighted average score]",
            "CONFIDENCE: [your confidence 0.0-1.0]",
            "SUMMARY: [2-3 sentence summary]",
            "```"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_deepeval_response(self, response: str) -> LLMJudgmentResult:
        """Parse DeepEval response."""
        criterion_scores = {}
        overall_score = 0.5
        confidence = 0.7
        rationale = ""
        
        # Extract metric scores
        for metric in self.evaluation_metrics.keys():
            pattern = rf"{metric.upper()}:\s*([\d.]+)"
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    criterion_scores[metric] = min(1.0, max(0.0, score))
                except ValueError:
                    pass
        
        # Extract overall score
        overall_match = re.search(r"OVERALL:\s*([\d.]+)", response, re.IGNORECASE)
        if overall_match:
            try:
                overall_score = float(overall_match.group(1))
            except ValueError:
                pass
        elif criterion_scores:
            # Calculate average if not provided
            overall_score = sum(criterion_scores.values()) / len(criterion_scores)
        
        # Extract confidence
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                pass
        
        # Extract summary
        sum_match = re.search(r"SUMMARY:\s*(.+?)(?:\n```|$)", response, re.DOTALL | re.IGNORECASE)
        if sum_match:
            rationale = sum_match.group(1).strip()
        else:
            rationale = "See metric scores above"
        
        return LLMJudgmentResult(
            judgment_id=self.judge_id,
            judgment_type=JudgmentType.DEEPEVAL,
            overall_score=overall_score,
            confidence=confidence,
            rationale=rationale,
            criterion_scores=criterion_scores
        )


class RubricBasedJudge(BaseLLMJudge):
    """
    Rubric-based judge with custom evaluation rubrics.
    
    Allows defining custom rubrics with multiple criteria and scoring levels.
    """
    
    def __init__(self, judge_id: str, rubric: EvaluationRubric, model_name: str = "gpt-4o"):
        super().__init__(judge_id, model_name)
        self.rubric = rubric
        
    def evaluate(self, task_data: Dict[str, Any], trajectory: Optional[List[Dict]], 
                result: Dict[str, Any]) -> LLMJudgmentResult:
        """Evaluate using custom rubric."""
        start_time = time.time()
        
        try:
            prompt = self._create_rubric_prompt(task_data, trajectory, result)
            
            system_prompt = f"""You are an expert evaluator using a custom rubric: {self.rubric.name}.
{self.rubric.description}

Evaluate each criterion carefully according to the rubric levels provided."""
            
            response = self._make_llm_call(prompt, system_prompt)
            parsed_result = self._parse_rubric_response(response)
            parsed_result.processing_time = time.time() - start_time
            parsed_result.model_used = self.model_name
            parsed_result.raw_response = response
            parsed_result.judgment_id = self.judge_id
            parsed_result.judgment_type = JudgmentType.RUBRIC_BASED
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Rubric-based judgment failed: {e}")
            return LLMJudgmentResult(
                judgment_id=self.judge_id,
                judgment_type=JudgmentType.RUBRIC_BASED,
                overall_score=0.0,
                confidence=0.3,
                rationale=f"Evaluation failed: {e}",
                processing_time=time.time() - start_time
            )
    
    def _create_rubric_prompt(self, task_data: Dict[str, Any], 
                             trajectory: Optional[List[Dict]], 
                             result: Dict[str, Any]) -> str:
        """Create rubric-based evaluation prompt."""
        prompt_parts = [
            f"# {self.rubric.name}\n",
            f"{self.rubric.description}\n",
            "## Task Information",
            f"**Task:** {task_data.get('instruction', 'N/A')}",
            f"**Result:** Success={result.get('success', False)}, Score={result.get('score', 0.0)}\n",
            "## Evaluation Criteria\n"
        ]
        
        for criterion in self.rubric.criteria:
            prompt_parts.extend([
                f"### {criterion.name} (Weight: {criterion.weight})",
                f"{criterion.description}",
                f"Score range: {self.rubric.scale[0]} to {self.rubric.scale[1]}\n",
                "Scoring levels:"
            ])
            
            for level, desc in sorted(criterion.levels.items()):
                prompt_parts.append(f"- {level}: {desc}")
            
            prompt_parts.append("")
        
        prompt_parts.extend([
            "## Instructions",
            "For each criterion, provide:",
            "1. The score based on the rubric levels",
            "2. Brief justification",
            "",
            "Format:",
            "```"
        ])
        
        for criterion in self.rubric.criteria:
            prompt_parts.append(f"{criterion.name.upper()}: [score] - [justification]")
        
        prompt_parts.extend([
            "",
            "OVERALL: [weighted score 0.0-1.0]",
            "CONFIDENCE: [0.0-1.0]",
            "RATIONALE: [summary]",
            "```"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_rubric_response(self, response: str) -> LLMJudgmentResult:
        """Parse rubric-based response."""
        criterion_scores = {}
        overall_score = 0.5
        confidence = 0.7
        rationale = ""
        
        # Extract criterion scores and normalize them
        for criterion in self.rubric.criteria:
            pattern = rf"{re.escape(criterion.name.upper())}:\s*([\d.]+)"
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    raw_score = float(match.group(1))
                    # Normalize to 0-1 scale
                    min_scale, max_scale = self.rubric.scale
                    normalized = (raw_score - min_scale) / (max_scale - min_scale)
                    criterion_scores[criterion.name] = min(1.0, max(0.0, normalized))
                except (ValueError, ZeroDivisionError):
                    pass
        
        # Calculate weighted average
        if criterion_scores:
            total_weight = self.rubric.total_weight()
            if total_weight > 0:
                weighted_sum = sum(
                    criterion_scores.get(c.name, 0.0) * c.weight
                    for c in self.rubric.criteria
                )
                overall_score = weighted_sum / total_weight
        
        # Extract overall score if provided
        overall_match = re.search(r"OVERALL:\s*([\d.]+)", response, re.IGNORECASE)
        if overall_match:
            try:
                overall_score = float(overall_match.group(1))
            except ValueError:
                pass
        
        # Extract confidence
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                pass
        
        # Extract rationale
        rat_match = re.search(r"RATIONALE:\s*(.+?)(?:\n```|$)", response, re.DOTALL | re.IGNORECASE)
        if rat_match:
            rationale = rat_match.group(1).strip()
        else:
            rationale = "See criterion evaluations above"
        
        return LLMJudgmentResult(
            judgment_id=self.judge_id,
            judgment_type=JudgmentType.RUBRIC_BASED,
            overall_score=overall_score,
            confidence=confidence,
            rationale=rationale,
            criterion_scores=criterion_scores
        )


class LLMJudgeEngine:
    """Engine for managing and running multiple LLM judges."""
    
    def __init__(self):
        self.judges: Dict[str, BaseLLMJudge] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def add_judge(self, judge: BaseLLMJudge):
        """Add a judge to the engine."""
        self.judges[judge.judge_id] = judge
        self.logger.info(f"Added judge: {judge.judge_id}")
        
    def remove_judge(self, judge_id: str):
        """Remove a judge from the engine."""
        if judge_id in self.judges:
            del self.judges[judge_id]
            self.logger.info(f"Removed judge: {judge_id}")
            
    def evaluate_all(self, task_data: Dict[str, Any], trajectory: Optional[List[Dict]], 
                    result: Dict[str, Any],
                    judge_ids: Optional[List[str]] = None) -> Dict[str, LLMJudgmentResult]:
        """
        Evaluate using all (or specified) judges.
        
        Args:
            task_data: Task information
            trajectory: Optional execution trajectory
            result: Evaluation result
            judge_ids: Optional list of specific judge IDs to use
            
        Returns:
            Dictionary mapping judge IDs to their results
        """
        if not OPENAI_AVAILABLE:
            self.logger.warning("OpenAI not available, skipping LLM judges")
            return {}
            
        judges_to_run = judge_ids or list(self.judges.keys())
        results = {}
        
        for judge_id in judges_to_run:
            if judge_id not in self.judges:
                self.logger.warning(f"Judge {judge_id} not found, skipping")
                continue
                
            try:
                self.logger.info(f"Running judge: {judge_id}")
                result_obj = self.judges[judge_id].evaluate(task_data, trajectory, result)
                results[judge_id] = result_obj
                self.logger.info(
                    f"Judge {judge_id} completed: score={result_obj.overall_score:.3f}, "
                    f"confidence={result_obj.confidence:.3f}"
                )
            except Exception as e:
                self.logger.error(f"Judge {judge_id} failed: {e}")
                results[judge_id] = LLMJudgmentResult(
                    judgment_id=judge_id,
                    judgment_type=JudgmentType.HOLISTIC,
                    overall_score=0.0,
                    confidence=0.0,
                    rationale=f"Judge execution failed: {e}"
                )
        
        return results

