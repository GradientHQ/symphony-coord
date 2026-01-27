"""Prompt generation utilities for mathematical problem solving.

This module provides utilities for generating prompts for mathematical problem
solving, including collaborative problem-solving techniques and prompt formatting.
"""

import sys
import os
import argparse
from typing import List, Optional, Any, Dict

# Get absolute path of project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add project root directory to system path
sys.path.append(project_root)


def divide_and_collaborate(problem: str, model: Optional[Any] = None, 
                          tokenizer: Optional[Any] = None, 
                          args: Optional[Any] = None) -> str:
    """Divide complex problem into subproblems for collaborative solving.
    
    This function implements a divide-and-conquer approach to problem solving
    by breaking down complex problems into manageable subproblems that can be
    solved sequentially or in parallel.
    
    Args:
        problem (str): The complex problem to be divided and solved
        model (Optional[Any]): The language model to use for generation.
                              If None, returns a placeholder result.
        tokenizer (Optional[Any]): The tokenizer to use with the model.
                                  If None, returns a placeholder result.
        args (Optional[Any]): Arguments containing model configuration such as
                             architecture type and generation parameters.
        
    Returns:
        str: Final answer after collaborative solving, or placeholder if
             model/tokenizer not provided
             
    Example:
        >>> problem = "Calculate the area of a circle with radius 5"
        >>> result = divide_and_collaborate(problem)
        >>> print(result)
        "Collaborative solution for: Calculate the area of a circle with radius 5"
    """
    # Define subproblems for collaborative solving
    sub_problems = [
        "Step 1: Calculate some intermediate result of the problem",
        "Step 2: Calculate final answer based on intermediate result"
    ]
    
    # If no model/tokenizer provided, return a placeholder result
    if model is None or tokenizer is None:
        return f"Collaborative solution for: {problem}"
    
    intermediate_result = None
    
    for sub_problem in sub_problems:
        # Combine original problem and subproblem
        input_problem = f"{problem} {sub_problem}"
        
        try:
            input_ids = tokenizer.encode(
                input_problem, return_tensors='pt'
            ).to(model.device)
            
            # Set max length based on model architecture
            max_length = (
                384 if (args and args.arch == 'gpt2-xl') else 1024
            )
            
            output_ids = model.generate(
                input_ids,
                num_beams=5,
                early_stopping=True,
                temperature=1.0,
                max_length=max_length
            )
            
            output_str = tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            
            if intermediate_result is None:
                intermediate_result = output_str
            else:
                final_answer = output_str
                return final_answer
                
        except Exception as e:
            print(f"Error processing subproblem: {e}")
            continue
    
    return intermediate_result or f"Unable to solve: {problem}"


def prepare_math_prompt(question: str, context: str = "") -> str:
    """Prepare a mathematical problem prompt.
    
    Formats a mathematical question with optional context into a standardized
    prompt format suitable for mathematical problem-solving models.
    
    Args:
        question (str): The mathematical question to be formatted
        context (str, optional): Additional context or background information.
                               Defaults to empty string.
        
    Returns:
        str: Formatted prompt ready for model input
        
    Example:
        >>> question = "What is 2 + 2?"
        >>> context = "Basic arithmetic"
        >>> prompt = prepare_math_prompt(question, context)
        >>> print(prompt)
        "Context: Basic arithmetic\nQuestion: What is 2 + 2?\nAnswer:"
    """
    if context:
        return f"Context: {context}\nQuestion: {question}\nAnswer:"
    else:
        return f"Question: {question}\nAnswer:"


def create_collaborative_prompt(base_problem: str, 
                               sub_tasks: List[str]) -> List[str]:
    """Create collaborative prompts for distributed problem solving.
    
    Generates a list of prompts that can be distributed to different agents
    for collaborative problem solving.
    
    Args:
        base_problem (str): The main problem to be solved
        sub_tasks (List[str]): List of subtasks derived from the base problem
        
    Returns:
        List[str]: List of formatted prompts for each subtask
    """
    prompts = []
    
    for i, sub_task in enumerate(sub_tasks):
        prompt = (
            f"Base Problem: {base_problem}\n"
            f"Subtask {i+1}: {sub_task}\n"
            f"Please solve this subtask and provide your answer."
        )
        prompts.append(prompt)
    
    return prompts


def format_solution_response(solution: str, confidence: float = 1.0) -> Dict[str, Any]:
    """Format a solution response with metadata.
    
    Args:
        solution (str): The solution text
        confidence (float): Confidence score between 0.0 and 1.0
        
    Returns:
        Dict[str, Any]: Formatted response with solution and metadata
    """
    return {
        "solution": solution,
        "confidence": max(0.0, min(1.0, confidence)),
        "timestamp": None,  # Can be populated by caller
        "method": "collaborative"
    }


def main() -> None:
    """Main function for command-line usage.
    
    Provides a command-line interface for testing the prompt generation
    utilities with example problems.
    """
    parser = argparse.ArgumentParser(
        description="Math problem solving utilities"
    )
    parser.add_argument(
        '--problem', 
        type=str, 
        help='Mathematical problem to solve'
    )
    parser.add_argument(
        '--arch', 
        default='gpt2', 
        type=str, 
        help='Model architecture'
    )
    
    args = parser.parse_args()
    
    if args.problem:
        # Process user-provided problem
        result = divide_and_collaborate(args.problem, args=args)
        print(f"Solution: {result}")
    else:
        # Example complex problem
        complex_problem = (
            "Calculate the area of a circle with radius 5, "
            "then multiply the result by 2"
        )
        answer = divide_and_collaborate(complex_problem, args=args)
        print(f"Answer to complex problem: {answer}")


if __name__ == "__main__":
    main()
