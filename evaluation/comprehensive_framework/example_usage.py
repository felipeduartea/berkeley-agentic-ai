"""
Examples of using the evaluation framework
"""

import sys
from pathlib import Path

# Handle imports
try:
    from core import ComprehensiveEvaluationFramework, evaluate_task
    from tac_integration import TheAgentCompanyEvaluator
    from scoring import ScoringPolicy
    from llm_judges import EvaluationRubric, RubricCriterion, RubricBasedJudge
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from core import ComprehensiveEvaluationFramework, evaluate_task
    from tac_integration import TheAgentCompanyEvaluator
    from scoring import ScoringPolicy
    from llm_judges import EvaluationRubric, RubricCriterion, RubricBasedJudge


# Example 1: Basic evaluation
def example_1_simple_evaluation():
    print("Example 1: Simple Evaluation\n" + "="*50)
    
    task = "Find the API server and start it"
    result = {"success": True, "score": 0.8}
    trajectory = [
        {"timestamp": 0, "action": {"action_type": "read"}, "observation": "Read task"},
        {"timestamp": 1, "action": {"action_type": "search"}, "observation": "Found code"},
        {"timestamp": 2, "action": {"action_type": "run"}, "observation": "Started server"}
    ]
    
    framework = ComprehensiveEvaluationFramework(
        enable_llm_judges=True
    )
    
    score, report = framework.evaluate_task(
        task_id="example-1",
        task_instruction=task,
        result_data=result,
        trajectory=trajectory
    )
    
    print(f"Score: {score.overall_score:.3f}")
    print(f"Confidence: {score.confidence:.3f}\n")
    
    for comp in score.component_scores:
        print(f"  {comp.component_name:13s}: {comp.score:.3f}")
    
    print()


# Example 2: Different policies
def example_2_policy_comparison():
    print("Example 2: Policy Comparison\n" + "="*50)
    
    task = "Complete the task"
    result = {"success": True, "score": 0.75}
    trajectory = [{"timestamp": i, "action": {"action_type": "act"}, "observation": "ok"} 
                  for i in range(5)]
    
    policies = {
        "balanced": ScoringPolicy.create_balanced_policy(),
        "conservative": ScoringPolicy.create_conservative_policy(),
        "ai_focused": ScoringPolicy.create_ai_focused_policy(),
    }
    
    for name, policy in policies.items():
        framework = ComprehensiveEvaluationFramework(
            scoring_policy=policy,
            enable_llm_judges=True
        )
        
        score, _ = framework.evaluate_task(
            task_id=f"example-2-{name}",
            task_instruction=task,
            result_data=result,
            trajectory=trajectory
        )
        
        print(f"{name:15s}: {score.overall_score:.3f} (conf: {score.confidence:.3f})")
    
    print()


# Example 3: Custom rubric
def example_3_custom_rubric():
    print("Example 3: Custom Rubric\n" + "="*50)
    
    rubric = EvaluationRubric(
        rubric_id="custom",
        name="Custom Eval",
        criteria=[
            RubricCriterion(
                criterion_id="speed",
                description="Was it fast?",
                weight=0.5,
                scale_min=0,
                scale_max=1
            ),
            RubricCriterion(
                criterion_id="accuracy",
                description="Was it correct?",
                weight=0.5,
                scale_min=0,
                scale_max=1
            )
        ]
    )
    
    framework = ComprehensiveEvaluationFramework(
        enable_llm_judges=True,
        custom_rubrics=[rubric]
    )
    
    score, _ = framework.evaluate_task(
        task_id="example-3",
        task_instruction="Do the thing quickly and correctly",
        result_data={"success": True, "score": 0.9},
        trajectory=[{"timestamp": i, "action": {"action_type": "act"}, "observation": "ok"} 
                   for i in range(3)]
    )
    
    print(f"Score: {score.overall_score:.3f}")
    print(f"Confidence: {score.confidence:.3f}\n")


# Example 4: TheAgentCompany integration
def example_4_tac_integration():
    print("Example 4: TAC Integration\n" + "="*50)
    
    evaluator = TheAgentCompanyEvaluator(
        policy=ScoringPolicy.create_balanced_policy(),
        enable_llm_judges=True
    )
    
    task_data = {
        "task_id": "example-task",
        "instruction": "Find and start API server",
        "result": {"success": True, "score": 0.8},
        "trajectory": [
            {"timestamp": 0, "action": {"action_type": "read"}, "observation": "Read"},
            {"timestamp": 1, "action": {"action_type": "execute"}, "observation": "Done"}
        ]
    }
    
    score, report = evaluator.evaluate_tac_task(task_data)
    
    print(f"Score: {score.overall_score:.3f}")
    print(f"Confidence: {score.confidence:.3f}\n")


# Example 5: Batch evaluation
def example_5_batch_evaluation():
    print("Example 5: Batch Evaluation\n" + "="*50)
    
    tasks = [
        {
            "task_id": "task-1",
            "instruction": "Task 1",
            "result": {"success": True, "score": 0.7},
            "trajectory": [{"timestamp": i, "action": {"action_type": "act"}, "observation": "ok"} 
                          for i in range(3)]
        },
        {
            "task_id": "task-2",
            "instruction": "Task 2",
            "result": {"success": True, "score": 0.9},
            "trajectory": [{"timestamp": i, "action": {"action_type": "act"}, "observation": "ok"} 
                          for i in range(5)]
        }
    ]
    
    evaluator = TheAgentCompanyEvaluator(enable_llm_judges=False)  # Faster
    results = evaluator.batch_evaluate_tasks(tasks)
    
    for task_id, (score, _) in results.items():
        print(f"{task_id:10s}: {score.overall_score:.3f}")
    
    print()


if __name__ == "__main__":
    example_1_simple_evaluation()
    # example_2_policy_comparison()
    # example_3_custom_rubric()
    # example_4_tac_integration()
    # example_5_batch_evaluation()
