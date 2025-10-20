"""
TheAgentCompany Comprehensive Evaluation Framework

A multi-layered evaluation system that provides sophisticated assessment through:
- Deterministic assertions (binary checks)
- Heuristic metrics (behavioral analysis) 
- LLM judges (G-Eval, DeepEval, Rubric-based)
- Flexible scoring policies
- Comprehensive reporting

Adapted from OSWorld evaluation framework for TheAgentCompany tasks.
"""

import sys
from pathlib import Path

# Add current directory to path to allow relative imports
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    from core import (
        ComprehensiveEvaluationFramework,
        evaluate_task,
        batch_evaluate_tasks
    )

    from llm_judges import (
        GEvalJudge,
        DeepEvalJudge,
        RubricBasedJudge,
        LLMJudgmentResult,
        JudgmentType
    )

    from scoring import (
        ScoringPolicy,
        ComponentWeight,
        AggregatedScore
    )
except ImportError:
    # Fallback for relative imports
    from .core import (
        ComprehensiveEvaluationFramework,
        evaluate_task,
        batch_evaluate_tasks
    )

    from .llm_judges import (
        GEvalJudge,
        DeepEvalJudge,
        RubricBasedJudge,
        LLMJudgmentResult,
        JudgmentType
    )

    from .scoring import (
        ScoringPolicy,
        ComponentWeight,
        AggregatedScore
    )

__version__ = "1.0.0"

__all__ = [
    'ComprehensiveEvaluationFramework',
    'evaluate_task',
    'batch_evaluate_tasks',
    'GEvalJudge',
    'DeepEvalJudge',
    'RubricBasedJudge',
    'LLMJudgmentResult',
    'JudgmentType',
    'ScoringPolicy',
    'ComponentWeight',
    'AggregatedScore'
]

