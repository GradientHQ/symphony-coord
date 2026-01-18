#!/usr/bin/env python3
"""
Symphony 2.0 - Data Generator & Difficulty Scorer
================================================

This module provides a unified framework for:
1. Loading and preprocessing all benchmarks (HumanEval, GSM8K, BBH, AMC, Medical QA)
2. Computing difficulty scores for each task using scientifically-grounded metrics
3. Validating difficulty definitions using weak models (GPT-3.5, DeepSeek)
4. Building task streams with custom difficulty distributions
5. Supporting flexible benchmark mixing with exclusion rules

Author: Symphony Team
Date: December 29, 2025
"""

import os
import json
import re
import yaml
import copy
import hashlib
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
from collections import defaultdict

# Optional: For running validation with weak models
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Task:
    """Unified task representation across all benchmarks"""
    task_id: str
    benchmark: str
    difficulty_score: float
    difficulty_bin: str  # 'easy' or 'hard'
    raw_data: Dict[str, Any]
    scorer_metadata: Dict[str, Any]  # For debugging/analysis
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# DIFFICULTY SCORERS (Base + Implementations)
# ============================================================================

class BaseDifficultyScorer(ABC):
    """Abstract base class for difficulty scorers"""
    
    def __init__(self, benchmark_name: str):
        self.benchmark_name = benchmark_name
        self.metadata = {}
    
    @abstractmethod
    def score(self, task: Dict[str, Any], norm_constants: Optional[Dict[str, float]] = None) -> float:
        """
        Score a single task's difficulty.
        
        Args:
            task: Raw task dictionary from the benchmark
            norm_constants: Optional normalization constants (95th percentile from full dataset)
            
        Returns:
            difficulty_score: Float in [0, 1], where 0=easy, 1=hard
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata for debugging/visualization purposes.
        
        Returns:
            Dictionary with scoring breakdown (e.g., n_asserts, prompt_len, etc.)
        """
        pass


class HumanEvalDifficultyScorer(BaseDifficultyScorer):
    """
    HumanEval Difficulty Scorer
    
    Definition: difficulty = 0.6 * (n_asserts / 20) + 0.4 * (prompt_len / 100)
    
    Rationale:
    - n_asserts: Number of assertions in test code → represents code path complexity
    - prompt_len: Word count of problem description → represents problem statement complexity
    
    References:
    - "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them" (2023)
      Shows multi-step reasoning increases difficulty
    - HumanEval papers: test complexity correlates with problem difficulty
    """
    
    def score(self, task: Dict[str, Any], norm_constants: Optional[Dict[str, float]] = None) -> float:
        metadata = self.extract_metadata(task)
        
        n_asserts = metadata['n_asserts']
        prompt_len = metadata['prompt_len']
        
        # Use data-driven normalization constants if provided, else fallback to defaults
        max_asserts = norm_constants.get('max_asserts', 20.0) if norm_constants else 20.0
        max_prompt_len = norm_constants.get('max_prompt_len', 100.0) if norm_constants else 100.0
        
        # Normalize to [0, 1]
        norm_asserts = min(n_asserts / max_asserts, 1.0)
        norm_prompt = min(prompt_len / max_prompt_len, 1.0)
        
        # Weighted combination
        difficulty = 0.6 * norm_asserts + 0.4 * norm_prompt
        
        return min(difficulty, 1.0)
    
    def extract_metadata(self, task: Dict[str, Any]) -> Dict[str, Any]:
        test_code = task.get('test', '')
        prompt = task.get('prompt', '')
        
        # Count assertions
        n_asserts = test_code.count('assert ') + test_code.count('check(')
        
        # Count words
        prompt_len = len(prompt.split())
        
        return {
            'n_asserts': n_asserts,
            'prompt_len': prompt_len,
        }


class GSM8KDifficultyScorer(BaseDifficultyScorer):
    """
    GSM8K Difficulty Scorer
    
    Definition: difficulty = reasoning_steps / 10
    
    Rationale:
    - GSM8K answers contain explicit reasoning steps (e.g., "First...", "Then...", "Finally...")
    - Reasoning depth directly indicates problem difficulty
    - Reference: "Using Depth of Thought as a Difficulty Signal for Tuning LLMs" (2024)
      "The depth of reasoning required to solve a problem directly corresponds to its difficulty"
    
    References:
    - GRADE: Generating multi-hop QA and fine-gRAined Difficulty (2025)
      "reasoning_depth = hop count, error_rate ∝ reasoning_depth"
    """
    
    def score(self, task: Dict[str, Any], norm_constants: Optional[Dict[str, float]] = None) -> float:
        metadata = self.extract_metadata(task)
        step_count = metadata['reasoning_steps']
        
        # Use data-driven normalization constant if provided, else fallback to default
        max_steps = norm_constants.get('max_reasoning_steps', 10.0) if norm_constants else 10.0
        
        # Normalize: GSM8K typically has 1-10 steps
        normalized_score = min(step_count / max_steps, 1.0)
        
        return normalized_score
    
    def extract_metadata(self, task: Dict[str, Any]) -> Dict[str, Any]:
        answer = task.get('answer', '')
        
        # Method 1: Count by newlines
        step_count_a = len(answer.strip().split('\n'))
        
        # Method 2: Count by numbered items (1., 2., etc.)
        steps_b = re.findall(r'^\d+\.|^[A-Za-z]\)', answer, re.MULTILINE)
        step_count_b = len(steps_b) if steps_b else 1
        
        # Method 3: Count by logical connectives
        logic_words = ['therefore', 'next', 'so', 'then', 'thus', 'thus,', 'finally']
        step_count_c = sum(1 for word in logic_words if word.lower() in answer.lower())
        
        # Take average for robustness
        step_count = (step_count_a + step_count_b + step_count_c) / 3.0
        
        return {
            'reasoning_steps': step_count,
            'answer_length': len(answer),
            'answer_words': len(answer.split()),
        }


class BBHDifficultyScorer(BaseDifficultyScorer):
    """
    BBH (Big-Bench Hard) Difficulty Scorer
    
    Definition: difficulty = base_complexity + input_factor + example_factor
    
    Rationale:
    - BBH has 23 tasks covering 10+ reasoning skills (deduction, causal, spatial, etc.)
    - Since no official difficulty labels exist, we use:
      1. Task-specific complexity (from design of BBH)
      2. Input length (more context = harder)
      3. Number of examples (more examples = potentially harder task)
    
    References:
    - BIG-Bench Extra Hard (BBEH, 2025)
      "Tasks require many-hop reasoning, long-range dependency, dealing with distractors"
    - "Challenging BIG-Bench Tasks" (2023)
      Analyzes 10 core reasoning abilities across BBH
    
    Implementation Notes:
    - task_complexity_map should be validated empirically:
      Run a weak model on all 23 BBH tasks, use 1 - accuracy as ground truth difficulty
    """
    
    # Task complexity map (based on BBH task design)
    # Should be verified by running weak model baseline
    TASK_COMPLEXITY_MAP = {
        # Relatively simple tasks (< 0.4)
        'sports_understanding': 0.25,
        'logical_fallacy_identification': 0.35,
        'movie_recommendation': 0.30,
        
        # Medium difficulty tasks (0.4 - 0.6)
        'date_understanding': 0.45,
        'disambiguation_qa': 0.50,
        'logical_deduction': 0.50,
        'reasoning_about_colored_objects': 0.50,
        'tracking_shuffled_objects': 0.55,
        
        # High difficulty tasks (> 0.6)
        'causal_reasoning': 0.70,
        'navigate': 0.70,
        'web_of_lies': 0.75,
        'formal_fallacies_syllogistic_logic': 0.80,
        'multi_step_arithmetic': 0.85,
    }
    
    def score(self, task: Dict[str, Any], norm_constants: Optional[Dict[str, float]] = None) -> float:
        metadata = self.extract_metadata(task)
        
        # Use data-driven normalization constant if provided, else fallback to default
        max_input_len = norm_constants.get('max_input_len', 150.0) if norm_constants else 150.0
        
        base_complexity = metadata['base_complexity']
        input_len = metadata['input_len']
        input_factor = min(input_len / max_input_len, 1.0) * 0.3
        
        difficulty = base_complexity + input_factor
        
        return min(difficulty, 1.0)
    
    def extract_metadata(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_name = task.get('task_name', '')
        input_text = task.get('input', '')
        
        # Base complexity from task design
        base_complexity = self.TASK_COMPLEXITY_MAP.get(task_name, 0.5)
        
        # Input length
        input_len = len(input_text.split())
        
        return {
            'task_name': task_name,
            'base_complexity': base_complexity,
            'input_len': input_len,
        }


class AMCDifficultyScorer(BaseDifficultyScorer):
    """
    AMC (American Mathematical Competition) Difficulty Scorer
    
    Definition: difficulty = 0.6 * (problem_length / 400) + 0.4 * (has_fraction_indicator)
    
    Rationale:
    - Longer problems typically have more complex reasoning
    - Problems with LaTeX/fractions are often more advanced
    - Since we don't have official problem ordering, use complexity heuristics
    """
    
    def score(self, task: Dict[str, Any], norm_constants: Optional[Dict[str, float]] = None) -> float:
        metadata = self.extract_metadata(task)
        
        problem_len = metadata['problem_len']
        has_latex = metadata['has_latex']
        
        # Use data-driven normalization constant if provided, else fallback to default
        max_problem_len = norm_constants.get('max_problem_len', 400.0) if norm_constants else 400.0
        
        # Normalize problem length
        norm_length = min(problem_len / max_problem_len, 1.0)
        
        # LaTeX presence indicates more advanced math
        latex_score = 0.3 if has_latex else 0.0
        
        difficulty = 0.7 * norm_length + 0.3 + latex_score * 0.4
        
        return min(difficulty, 1.0)
    
    def extract_metadata(self, task: Dict[str, Any]) -> Dict[str, Any]:
        problem = task.get('problem', '')
        
        problem_len = len(problem)
        has_latex = ('\\' in problem or '$' in problem)
        
        return {
            'problem_len': problem_len,
            'has_latex': has_latex,
            'problem_id': task.get('id', 0),
        }


class MedicalQADifficultyScorer(BaseDifficultyScorer):
    """
    Medical QA Difficulty Scorer
    
    Definition: difficulty = 0.4*steps + 0.3*keywords + 0.2*answer_len + 0.1*question_len
    
    Rationale:
    - Medical QA difficulty comes from multiple dimensions:
      1. Reasoning depth (# of clinical reasoning steps)
      2. Knowledge coverage (# of medical concepts referenced)
      3. Answer complexity (length and detail)
    
    References:
    - MedReason-Dx (2025)
      "Medical QA requires 6.4 avg reasoning steps + 27.1 fine-grained clinical points"
      "Difficulty = reasoning_steps + knowledge_coverage"
    
    - Medical QA Systems Survey
      Difficulty factors: multi-hop reasoning, contradictions, domain expertise
    
    - GRADE Framework
      In RAG + Medical: difficulty = retrieval_difficulty + reasoning_difficulty
    """
    
    # Medical keywords for complexity assessment
    MEDICAL_KEYWORDS = [
        'diagnosis', 'treatment', 'syndrome', 'pathology', 'etiology',
        'differential', 'contraindication', 'adverse', 'comorbidity',
        'prognosis', 'staging', 'protocol', 'guideline', 'evidence',
        'therapeutic', 'pharmacology', 'mechanism', 'clinical', 'patient',
    ]
    
    def score(self, task: Dict[str, Any], norm_constants: Optional[Dict[str, float]] = None) -> float:
        metadata = self.extract_metadata(task)
        
        # Use data-driven normalization constants if provided, else fallback to defaults
        max_question_words = norm_constants.get('max_question_words', 200.0) if norm_constants else 200.0
        max_keywords = norm_constants.get('max_keywords', 8.0) if norm_constants else 8.0
        max_option_len = norm_constants.get('max_option_len', 20.0) if norm_constants else 20.0
        
        question_words = metadata['question_words']
        n_keywords = metadata['n_keywords']
        avg_option_len = metadata['avg_option_len']
        is_clinical_case = metadata['is_clinical_case']
        
        # Normalize with data-driven constants
        norm_question_len = min(question_words / max_question_words, 1.0)
        norm_keywords = min(n_keywords / max_keywords, 1.0)
        norm_option_complexity = min(avg_option_len / max_option_len, 1.0)
        case_bonus = 0.2 if is_clinical_case else 0.0
        
        # Weighted combination: question length and medical terminology
        difficulty = (
            0.4 * norm_question_len +
            0.3 * norm_keywords +
            0.2 * norm_option_complexity +
            case_bonus
        )
        
        return min(difficulty, 1.0)
    
    def extract_metadata(self, task: Dict[str, Any]) -> Dict[str, Any]:
        question = task.get('question', '')
        answer = task.get('answer', '')
        options = task.get('options', {})
        
        # Feature 1: Question complexity (longer = harder)
        question_words = len(question.split())
        
        # Feature 2: Medical keywords count
        n_keywords = sum(
            1 for kw in self.MEDICAL_KEYWORDS 
            if kw.lower() in question.lower()
        )
        
        # Feature 3: Option complexity (average option length)
        avg_option_len = 0
        if options:
            option_texts = [str(v) for v in options.values() if v]
            if option_texts:
                avg_option_len = sum(len(opt.split()) for opt in option_texts) / len(option_texts)
        
        # Feature 4: Has clinical scenario (longer questions are usually clinical cases)
        is_clinical_case = question_words > 100
        
        return {
            'question_words': question_words,
            'n_keywords': n_keywords,
            'avg_option_len': avg_option_len,
            'is_clinical_case': is_clinical_case,
        }


# ============================================================================
# DIFFICULTY VALIDATOR (using weak models)
# ============================================================================

class DifficultyValidator:
    """
    Validates difficulty definitions by checking:
    1. Difficulty scores correlate with weak model error rate
    2. Easy tasks have high accuracy, hard tasks have low accuracy
    3. Distribution of scores makes sense
    
    Usage:
        validator = DifficultyValidator(
            weak_model='gpt-3.5-turbo',
            api_key='sk-...'
        )
        stats = validator.validate(tasks, benchmark='humaneval', n_samples=20)
    """
    
    def __init__(
        self,
        weak_model: str = 'gpt-3.5-turbo',
        api_key: Optional[str] = None,
        budget_limit: float = 5.0
    ):
        """
        Args:
            weak_model: Model to use for validation (gpt-3.5-turbo, etc.)
            api_key: OpenAI API key (uses env var if not provided)
            budget_limit: Maximum budget for validation in USD
        """
        self.weak_model = weak_model
        self.budget_limit = budget_limit
        
        if not HAS_OPENAI:
            logger.warning("OpenAI not installed. Validation requires: pip install openai")
            self.client = None
            return
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
    
    def validate(
        self,
        tasks: List[Task],
        benchmark: str,
        n_samples: int = 20,
        random_seed: int = 2025
    ) -> Dict[str, Any]:
        """
        Validate difficulty scores by running weak model on sample tasks.
        
        Returns:
            {
                'correlation': float,  # Pearson correlation (difficulty vs 1-accuracy)
                'easy_accuracy': float,
                'hard_accuracy': float,
                'mean_difficulty': float,
                'std_difficulty': float,
                'results_df': DataFrame,
            }
        """
        
        if self.client is None:
            logger.warning("Cannot validate without OpenAI API key. Skipping validation.")
            return {'error': 'No OpenAI client'}
        
        logger.info(f"Starting validation for {benchmark} ({n_samples} samples)...")
        
        np.random.seed(random_seed)
        sampled_tasks = np.random.choice(tasks, min(n_samples, len(tasks)), replace=False)
        
        results = []
        total_cost = 0.0
        
        for i, task in enumerate(sampled_tasks):
            # Skip if budget exceeded
            if total_cost > self.budget_limit:
                logger.warning(f"Budget limit ${self.budget_limit} reached. Stopping validation.")
                break
            
            logger.info(f"[{i+1}/{len(sampled_tasks)}] Validating {task.task_id}...")
            
            try:
                accuracy, cost = self._evaluate_task(task, benchmark)
                total_cost += cost
                
                results.append({
                    'task_id': task.task_id,
                    'benchmark': benchmark,
                    'difficulty_score': task.difficulty_score,
                    'difficulty_bin': task.difficulty_bin,
                    'accuracy': accuracy,
                    'error_rate': 1.0 - accuracy,
                    'cost': cost,
                })
            except Exception as e:
                logger.error(f"Error evaluating {task.task_id}: {e}")
        
        # Analyze results
        df = pd.DataFrame(results)
        
        if len(df) == 0:
            logger.warning("No validation results collected.")
            return {'error': 'No results'}
        
        # Compute correlation (difficulty should correlate with error rate)
        correlation = df['difficulty_score'].corr(df['error_rate'])
        
        # Split by difficulty bin
        easy_tasks = df[df['difficulty_bin'] == 'easy']
        hard_tasks = df[df['difficulty_bin'] == 'hard']
        
        easy_accuracy = easy_tasks['accuracy'].mean() if len(easy_tasks) > 0 else 0.0
        hard_accuracy = hard_tasks['accuracy'].mean() if len(hard_tasks) > 0 else 0.0
        
        stats = {
            'benchmark': benchmark,
            'n_samples': len(df),
            'total_cost': total_cost,
            'correlation': correlation,  # Should be > 0.7 (higher difficulty → lower accuracy)
            'easy_accuracy': easy_accuracy,
            'hard_accuracy': hard_accuracy,
            'mean_difficulty': df['difficulty_score'].mean(),
            'std_difficulty': df['difficulty_score'].std(),
            'accuracy_gap': easy_accuracy - hard_accuracy,  # Should be large
            'results_df': df,
        }
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"VALIDATION SUMMARY - {benchmark}")
        logger.info(f"{'='*60}")
        logger.info(f"Correlation (difficulty vs error_rate): {correlation:.3f} {'✓' if correlation > 0.6 else '✗'}")
        logger.info(f"Easy task accuracy: {easy_accuracy:.1%}")
        logger.info(f"Hard task accuracy: {hard_accuracy:.1%}")
        logger.info(f"Accuracy gap: {stats['accuracy_gap']:.1%}")
        logger.info(f"Total cost: ${total_cost:.2f}")
        logger.info(f"{'='*60}\n")
        
        return stats
    
    def _evaluate_task(self, task: Task, benchmark: str) -> Tuple[float, float]:
        """
        Run weak model on a single task and compute accuracy.
        
        Returns:
            (accuracy, cost_in_usd)
        """
        
        if benchmark == 'humaneval':
            return self._eval_humaneval(task)
        elif benchmark == 'gsm8k':
            return self._eval_gsm8k(task)
        elif benchmark == 'bbh':
            return self._eval_bbh(task)
        elif benchmark == 'amc':
            return self._eval_amc(task)
        elif benchmark == 'medical_qa':
            return self._eval_medical_qa(task)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
    
    def _eval_humaneval(self, task: Task) -> Tuple[float, float]:
        """Evaluate HumanEval task"""
        raw = task.raw_data
        prompt = raw.get('prompt', '')
        test = raw.get('test', '')
        
        # Call weak model
        response = self.client.chat.completions.create(
            model=self.weak_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )
        
        code = response.choices[0].message.content
        tokens = response.usage.total_tokens
        cost = tokens * (0.5 / 1e6)  # gpt-3.5-turbo: $0.5/1M tokens (approximate)
        
        # Check if code passes test
        try:
            exec(code + "\n" + test)
            accuracy = 1.0
        except Exception:
            accuracy = 0.0
        
        return accuracy, cost
    
    def _eval_gsm8k(self, task: Task) -> Tuple[float, float]:
        """Evaluate GSM8K task"""
        raw = task.raw_data
        question = raw.get('question', '')
        answer = raw.get('answer', '')
        
        response = self.client.chat.completions.create(
            model=self.weak_model,
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        generated = response.choices[0].message.content
        tokens = response.usage.total_tokens
        cost = tokens * (0.5 / 1e6)
        
        # Simple check: does answer contain correct final number?
        correct_answer = str(answer).split()[-1]
        accuracy = 1.0 if correct_answer in generated else 0.0
        
        return accuracy, cost
    
    def _eval_bbh(self, task: Task) -> Tuple[float, float]:
        """Evaluate BBH task"""
        raw = task.raw_data
        input_text = raw.get('input', '')
        target = raw.get('target', '')
        
        response = self.client.chat.completions.create(
            model=self.weak_model,
            messages=[
                {"role": "user", "content": f"Answer: {input_text}"}
            ],
            temperature=0.0,
            max_tokens=100
        )
        
        generated = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens
        cost = tokens * (0.5 / 1e6)
        
        # Simple string matching
        accuracy = 1.0 if target.strip() in generated else 0.0
        
        return accuracy, cost
    
    def _eval_amc(self, task: Task) -> Tuple[float, float]:
        """Evaluate AMC task"""
        raw = task.raw_data
        problem_text = raw.get('problem', '')
        answer = raw.get('answer', '')
        
        response = self.client.chat.completions.create(
            model=self.weak_model,
            messages=[
                {"role": "user", "content": problem_text}
            ],
            temperature=0.0,
            max_tokens=200
        )
        
        generated = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens
        cost = tokens * (0.5 / 1e6)
        
        # Check if answer is in response
        accuracy = 1.0 if str(answer) in generated else 0.0
        
        return accuracy, cost
    
    def _eval_medical_qa(self, task: Task) -> Tuple[float, float]:
        """Evaluate Medical QA task"""
        raw = task.raw_data
        question = raw.get('question', '')
        answer = raw.get('answer', '')
        
        response = self.client.chat.completions.create(
            model=self.weak_model,
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        generated = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens
        cost = tokens * (0.5 / 1e6)
        
        # Simple heuristic: check if key medical term is in answer
        key_term = answer.split('\n')[0].split()[-1] if answer else ''
        accuracy = 1.0 if key_term.lower() in generated.lower() else 0.0
        
        return accuracy, cost


# ============================================================================
# DATASET BUILDER (Main orchestrator)
# ============================================================================

class DatasetBuilder:
    """
    Main orchestrator for building task streams with flexible difficulty control.
    
    Usage:
        builder = DatasetBuilder('config/data_config.yaml')
        tasks = builder.build_task_stream(
            benchmarks=['humaneval', 'gsm8k'],
            difficulty_split='80:20',
            n_tasks=1000
        )
        builder.save_task_pool(tasks, 'data/task_pool.jsonl')
    """
    
    def __init__(self, config_path: str = 'config/data_config.yaml'):
        """
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.stream_cfg = self.config.get('stream_generation', {}) or {}
        self.scorers = self._init_scorers()
        self.validator = None
        
        logger.info(f"Initialized DatasetBuilder with config: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration"""
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._default_config()

        with open(config_path, 'r', encoding='utf-8-sig') as f:
            return yaml.safe_load(f)

    def _default_config(self) -> Dict[str, Any]:
        """Fallback default configuration"""
        return {
            'benchmarks': {
                'humaneval': {
                    'path': 'datasets:openai_humaneval',
                    'enabled': True,
                    'task_type': 'code',
                    'n_problems': 164,
                },
                'gsm8k': {
                    'path': 'datasets:gsm8k/main',
                    'enabled': True,
                    'task_type': 'math',
                    'n_problems': 200,
                },
                'bbh': {
                    'path': 'datasets:big_bench',
                    'enabled': False,
                    'task_type': 'reasoning',
                    'n_problems': 100,
                },
                'amc': {
                    'path': 'custom:amc_dataset',
                    'enabled': False,
                    'task_type': 'math_competition',
                    'n_problems': 50,
                },
                'medical_qa': {
                    'path': 'custom:medical_qa',
                    'enabled': False,
                    'task_type': 'domain_qa',
                    'n_problems': 100,
                },
            },
            'mixing_rules': {
                'exclude_combinations': [],
                'task_type_balance': {
                    'code': 0.5,
                    'math': 0.3,
                    'reasoning': 0.2,
                },
            },
        }
    
    def _init_scorers(self) -> Dict[str, BaseDifficultyScorer]:
        """Initialize difficulty scorers for all benchmarks"""
        return {
            'humaneval': HumanEvalDifficultyScorer('humaneval'),
            'gsm8k': GSM8KDifficultyScorer('gsm8k'),
            'bbh': BBHDifficultyScorer('bbh'),
            'amc': AMCDifficultyScorer('amc'),
            'medical_qa': MedicalQADifficultyScorer('medical_qa'),
        }
    
    def enable_validation(self, api_key: Optional[str] = None, budget: float = 5.0):
        """
        Enable difficulty validation using weak models.
        
        Args:
            api_key: OpenAI API key (or use env var OPENAI_API_KEY)
            budget: Maximum budget in USD for validation
        """
        self.validator = DifficultyValidator(
            weak_model='gpt-3.5-turbo',
            api_key=api_key,
            budget_limit=budget
        )
        logger.info("Difficulty validation enabled")
    
    def validate_difficulties(
        self,
        tasks: List[Task],
        benchmark: str,
        n_samples: int = 20
    ) -> Dict[str, Any]:
        """
        Validate difficulty scores for a benchmark.
        
        Returns statistics including correlation coefficient.
        """
        if self.validator is None:
            logger.warning("Validation not enabled. Call enable_validation() first.")
            return {}
        
        return self.validator.validate(tasks, benchmark, n_samples)
    
    def validate_all_benchmarks(
        self,
        tasks: List[Task],
        n_samples: int = 20,
        budget_limit: float = 2.0
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate difficulty scores for all benchmarks in the task list.
        
        Args:
            tasks: List of Task objects to validate
            n_samples: Number of samples to validate per benchmark
            budget_limit: Maximum budget in USD for all validations
            
        Returns:
            Dictionary mapping benchmark names to validation statistics
        """
        if self.validator is None:
            logger.warning("Validation not enabled. Call enable_validation() first.")
            return {}
        
        # Group tasks by benchmark
        tasks_by_benchmark = defaultdict(list)
        for task in tasks:
            tasks_by_benchmark[task.benchmark].append(task)
        
        logger.info(f"Validating {len(tasks_by_benchmark)} benchmarks...")
        logger.info(f"Benchmarks to validate: {', '.join(tasks_by_benchmark.keys())}")
        
        results = {}
        total_cost = 0.0
        
        for benchmark, benchmark_tasks in tasks_by_benchmark.items():
            # Check if budget exceeded
            if total_cost >= budget_limit:
                logger.warning(f"Budget limit ${budget_limit:.2f} reached. Stopping validation.")
                results[benchmark] = {'error': 'Budget exceeded'}
                continue
            
            logger.info(f"\nValidating {benchmark}...")
            
            try:
                # Validate this benchmark
                stats = self.validate_difficulties(
                    benchmark_tasks,
                    benchmark=benchmark,
                    n_samples=min(n_samples, len(benchmark_tasks))
                )
                
                if 'error' not in stats:
                    total_cost += stats.get('total_cost', 0.0)
                    results[benchmark] = stats
                else:
                    results[benchmark] = stats
                    
            except Exception as e:
                logger.error(f"Error validating {benchmark}: {e}")
                results[benchmark] = {'error': str(e)}
        
        logger.info(f"\nTotal validation cost: ${total_cost:.2f}")
        
        return results
    
    def load_benchmark(self, benchmark_name: str) -> List[Task]:
        """
        DEPRECATED: Use preprocess + load_preprocessed_benchmark instead.
        
        Load a single benchmark and compute difficulty scores.
        
        Returns:
            List of Task objects with difficulty scores
        """
        cfg = self.config['benchmarks'].get(benchmark_name)
        if cfg is None or not cfg.get('enabled', False):
            logger.info(f"Benchmark {benchmark_name} is disabled or not found.")
            return []
        
        logger.info(f"Loading benchmark: {benchmark_name}")
        
        # Load raw data (implementation depends on benchmark)
        raw_data = self._load_raw_data(benchmark_name, cfg)
        
        # Score each task
        tasks = []
        scorer = self.scorers[benchmark_name]
        
        for idx, raw_task in enumerate(raw_data):
            task_id = f"{benchmark_name}_{idx}"
            
            difficulty_score = scorer.score(raw_task)
            metadata = scorer.extract_metadata(raw_task)
            
            task = Task(
                task_id=task_id,
                benchmark=benchmark_name,
                difficulty_score=difficulty_score,
                difficulty_bin='',  # Will be set later
                raw_data=raw_task,
                scorer_metadata=metadata,
            )
            
            tasks.append(task)
        
        logger.info(f"Loaded {len(tasks)} tasks from {benchmark_name}")
        return tasks
    
    def _benchmark_config_fingerprint(self, benchmark_name: str) -> str:
        """
        Return a stable fingerprint for the benchmark config (used to warn on stale caches).
        """
        cfg = (self.config.get('benchmarks') or {}).get(benchmark_name, {})
        blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode('utf-8')).hexdigest()

    def _load_tasks_from_jsonl(self, jsonl_path: str) -> List[Task]:
        """
        Load Task objects from a JSONL file produced by preprocess_* methods.
        """
        tasks: List[Task] = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {jsonl_path} at line {line_no}: {e}") from e

                try:
                    tasks.append(Task(**obj))
                except TypeError as e:
                    raise ValueError(
                        f"Invalid Task schema in {jsonl_path} at line {line_no}: {e}"
                    ) from e
        return tasks

    def load_preprocessed_benchmark(
        self,
        benchmark_name: str,
        preprocessed_dir: str = 'data/benchmarks/sampled',
        auto_reprocess_on_error: bool = False,
    ) -> List[Task]:
        """
        Load a sampled benchmark from JSONL.
        
        By default loads from 'data/benchmarks/sampled/' which contains sampled subsets.
        If the JSONL is missing, raises FileNotFoundError.
        """
        pre_dir = Path(preprocessed_dir)
        jsonl_path = pre_dir / f"{benchmark_name}_sampled.jsonl"
        meta_path = pre_dir / f"{benchmark_name}_sampled_meta.json"

        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Sampled benchmark file not found: {jsonl_path}\n"
                f"Please run preprocess_all_benchmarks() and sample_benchmarks() first."
            )

        try:
            tasks = self._load_tasks_from_jsonl(str(jsonl_path))
        except Exception as e:
            raise ValueError(f"Failed to load sampled benchmark {benchmark_name} from {jsonl_path}: {e}")

        # Warn if config changed since preprocessing (best-effort)
        try:
            if meta_path.exists():
                with open(meta_path, 'r') as mf:
                    meta = json.load(mf)
                old_fp = meta.get('benchmark_config_fingerprint')
                new_fp = self._benchmark_config_fingerprint(benchmark_name)
                if old_fp and old_fp != new_fp:
                    logger.warning(
                        f"Benchmark config changed since preprocessing for {benchmark_name}. "
                        f"Consider re-running preprocess (force_reprocess=True)."
                    )
        except Exception:
            # Never fail load due to meta issues
            pass

        if len(tasks) == 0:
            logger.warning(f"Preprocessed task list is empty for {benchmark_name}: {jsonl_path}")

        return tasks

    def _compute_normalization_constants(
        self,
        benchmark_name: str,
        all_metadata: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute data-driven normalization constants using 95th percentile.
        
        Args:
            benchmark_name: Name of the benchmark
            all_metadata: List of metadata dicts from all tasks
        
        Returns:
            Dictionary of normalization constants for this benchmark
        """
        if len(all_metadata) == 0:
            return {}
        
        constants = {}
        
        if benchmark_name == 'humaneval':
            # n_asserts and prompt_len
            n_asserts_list = [m['n_asserts'] for m in all_metadata]
            prompt_len_list = [m['prompt_len'] for m in all_metadata]
            constants['max_asserts'] = float(np.percentile(n_asserts_list, 95))
            constants['max_prompt_len'] = float(np.percentile(prompt_len_list, 95))
        
        elif benchmark_name == 'gsm8k':
            # reasoning_steps
            steps_list = [m['reasoning_steps'] for m in all_metadata]
            constants['max_reasoning_steps'] = float(np.percentile(steps_list, 95))
        
        elif benchmark_name == 'bbh':
            # input_len
            input_len_list = [m['input_len'] for m in all_metadata]
            constants['max_input_len'] = float(np.percentile(input_len_list, 95))
        
        elif benchmark_name == 'amc':
            # problem_len
            problem_len_list = [m['problem_len'] for m in all_metadata]
            constants['max_problem_len'] = float(np.percentile(problem_len_list, 95))
        
        elif benchmark_name == 'medical_qa':
            # question_words, n_keywords, avg_option_len
            question_words_list = [m['question_words'] for m in all_metadata]
            n_keywords_list = [m['n_keywords'] for m in all_metadata]
            option_len_list = [m['avg_option_len'] for m in all_metadata]
            constants['max_question_words'] = float(np.percentile(question_words_list, 95))
            constants['max_keywords'] = float(np.percentile(n_keywords_list, 95))
            constants['max_option_len'] = float(np.percentile(option_len_list, 95))
        
        return constants

    def preprocess_all_benchmarks(
        self,
        output_dir: str = 'data/benchmarks/full',
        force_reprocess: bool = False,
    ) -> Dict[str, List[Task]]:
        """
        One-time preprocessing step - saves FULL preprocessed benchmarks:
        - download/load ALL raw benchmark data
        - compute difficulty scores for ALL tasks (with data-driven normalization)
        - normalize within each benchmark
        - write FULL JSONL to output_dir/{benchmark}_full.jsonl
        
        Note: This saves ALL tasks. Use sample_benchmarks() to create sampled subsets.
        """
        benchmarks_cfg = self.config.get('benchmarks') or {}
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, List[Task]] = {}

        for benchmark_name, cfg in benchmarks_cfg.items():
            if not cfg or not cfg.get('enabled', False):
                continue

            jsonl_path = out_dir / f"{benchmark_name}_full.jsonl"
            meta_path = out_dir / f"{benchmark_name}_full_meta.json"

            if jsonl_path.exists() and not force_reprocess:
                try:
                    results[benchmark_name] = self._load_tasks_from_jsonl(str(jsonl_path))
                    continue
                except Exception as e:
                    logger.warning(
                        f"Existing preprocessed file is unreadable for {benchmark_name} ({e}). "
                        f"Reprocessing..."
                    )

            logger.info(f"Preprocessing benchmark: {benchmark_name}")
            # STEP 1: Load FULL dataset (no sampling yet)
            raw_data = self._load_raw_data(benchmark_name, cfg, full_dataset=True)

            if len(raw_data) == 0:
                logger.warning(
                    f"No raw data loaded for {benchmark_name}. Skipping preprocessing."
                )
                continue

            scorer = self.scorers.get(benchmark_name)
            if scorer is None:
                logger.warning(f"No scorer found for {benchmark_name}. Skipping preprocessing.")
                continue

            logger.info(f"Computing difficulty scores for {len(raw_data)} tasks...")
            
            # STEP 2a: Extract all metadata first (to compute normalization constants)
            all_metadata = []
            for raw_task in raw_data:
                metadata = scorer.extract_metadata(raw_task)
                all_metadata.append(metadata)
            
            # STEP 2b: Compute data-driven normalization constants (95th percentile)
            norm_constants = self._compute_normalization_constants(benchmark_name, all_metadata)
            logger.info(f"Computed normalization constants: {norm_constants}")
            
            # STEP 2c: Compute raw scores using data-driven constants
            tasks: List[Task] = []
            raw_scores = []
            
            for idx, (raw_task, metadata) in enumerate(zip(raw_data, all_metadata)):
                task_id = f"{benchmark_name}_{idx}"
                raw_score = float(scorer.score(raw_task, norm_constants))
                raw_scores.append(raw_score)
                
                tasks.append(
                    Task(
                        task_id=task_id,
                        benchmark=benchmark_name,
                        difficulty_score=raw_score,  # Will normalize below
                        difficulty_bin='',
                        raw_data=raw_task,
                        scorer_metadata=metadata,
                    )
                )
            
            # STEP 3: Normalize scores within this benchmark (min-max normalization)
            if len(raw_scores) > 0:
                min_score = min(raw_scores)
                max_score = max(raw_scores)
                score_range = max_score - min_score
                
                if score_range > 0:
                    logger.info(
                        f"Normalizing scores: raw range [{min_score:.3f}, {max_score:.3f}] -> [0.0, 1.0]"
                    )
                    # Min-max normalization: (x - min) / (max - min)
                    for task, raw_score in zip(tasks, raw_scores):
                        task.difficulty_score = (raw_score - min_score) / score_range
                else:
                    # All scores are the same - set to 0.5 (middle)
                    logger.warning(f"All raw scores are identical ({min_score:.3f}), setting to 0.5")
                    for task in tasks:
                        task.difficulty_score = 0.5
            
            # Save ALL tasks (no sampling at this stage)
            logger.info(f"Saving {len(tasks)} full preprocessed tasks...")
            
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for task in tasks:
                    f.write(json.dumps(task.to_dict(), ensure_ascii=False) + '\n')

            # Save metadata
            try:
                scores = [t.difficulty_score for t in tasks]
                meta = {
                    'benchmark': benchmark_name,
                    'n_tasks_full': len(tasks),
                    'benchmark_config_fingerprint': self._benchmark_config_fingerprint(benchmark_name),
                    'normalization': {
                        'method': 'min-max',
                        'raw_min': float(min(raw_scores)) if raw_scores else 0.0,
                        'raw_max': float(max(raw_scores)) if raw_scores else 1.0,
                        'normalized_min': float(min(scores)) if scores else 0.0,
                        'normalized_max': float(max(scores)) if scores else 1.0,
                        'normalized_mean': float(np.mean(scores)) if scores else 0.5,
                    },
                    'normalization_constants': norm_constants,  # Data-driven constants (95th percentile)
                    'note': 'This is the FULL preprocessed dataset. Use sample_benchmarks() to create sampled subsets.'
                }
                with open(meta_path, 'w', encoding='utf-8') as mf:
                    json.dump(meta, mf, indent=2, ensure_ascii=False)
            except Exception:
                pass

            logger.info(f"✓ Saved full preprocessed benchmark ({len(tasks)} tasks) to {jsonl_path}")
            results[benchmark_name] = tasks

        return results

    def sample_benchmarks(
        self,
        full_dir: str = 'data/benchmarks/full',
        sampled_dir: str = 'data/benchmarks/sampled',
        force_resample: bool = False,
    ) -> Dict[str, List[Task]]:
        """
        Sample subsets from full preprocessed benchmarks.
        
        Args:
            full_dir: Directory containing full preprocessed benchmarks
            sampled_dir: Directory to save sampled subsets
            force_resample: If True, resample even if sampled files exist
        
        Returns:
            Dict mapping benchmark name to sampled tasks
        """
        benchmarks_cfg = self.config.get('benchmarks') or {}
        full_path = Path(full_dir)
        sampled_path = Path(sampled_dir)
        sampled_path.mkdir(parents=True, exist_ok=True)

        results: Dict[str, List[Task]] = {}

        for benchmark_name, cfg in benchmarks_cfg.items():
            if not cfg or not cfg.get('enabled', False):
                continue

            full_jsonl = full_path / f"{benchmark_name}_full.jsonl"
            sampled_jsonl = sampled_path / f"{benchmark_name}_sampled.jsonl"
            sampled_meta = sampled_path / f"{benchmark_name}_sampled_meta.json"

            # Check if sampled file already exists
            if sampled_jsonl.exists() and not force_resample:
                try:
                    results[benchmark_name] = self._load_tasks_from_jsonl(str(sampled_jsonl))
                    logger.info(f"✓ Loaded existing sampled benchmark: {benchmark_name}")
                    continue
                except Exception as e:
                    logger.warning(
                        f"Existing sampled file is unreadable for {benchmark_name} ({e}). "
                        f"Resampling..."
                    )

            # Load full preprocessed data
            if not full_jsonl.exists():
                logger.warning(
                    f"Full preprocessed data not found for {benchmark_name} at {full_jsonl}. "
                    f"Run preprocess_all_benchmarks() first. Skipping."
                )
                continue

            logger.info(f"Sampling benchmark: {benchmark_name}")
            try:
                full_tasks = self._load_tasks_from_jsonl(str(full_jsonl))
            except Exception as e:
                logger.error(f"Failed to load full data for {benchmark_name}: {e}")
                continue

            if len(full_tasks) == 0:
                logger.warning(f"No tasks in full data for {benchmark_name}. Skipping.")
                continue

            # Sample according to config
            n_problems = cfg.get('n_problems', None)
            if n_problems and n_problems < len(full_tasks):
                logger.info(f"Sampling {n_problems} tasks from {len(full_tasks)} total...")
                # Sample randomly with fixed seed for reproducibility
                np.random.seed(2025)
                sampled_indices = np.random.choice(
                    len(full_tasks), 
                    size=n_problems, 
                    replace=False
                )
                sampled_indices = sorted(sampled_indices)  # Keep original order
                sampled_tasks = [full_tasks[i] for i in sampled_indices]
            else:
                logger.info(f"Using all {len(full_tasks)} tasks (no sampling needed)")
                sampled_tasks = full_tasks

            # Save sampled data
            with open(sampled_jsonl, 'w', encoding='utf-8') as f:
                for task in sampled_tasks:
                    f.write(json.dumps(task.to_dict(), ensure_ascii=False) + '\n')

            # Save metadata
            try:
                scores = [t.difficulty_score for t in sampled_tasks]
                meta = {
                    'benchmark': benchmark_name,
                    'n_tasks_sampled': len(sampled_tasks),
                    'n_tasks_full': len(full_tasks),
                    'sampling_config': {
                        'n_problems': n_problems,
                        'seed': 2025,
                    },
                    'difficulty_stats': {
                        'min': float(min(scores)) if scores else 0.0,
                        'max': float(max(scores)) if scores else 1.0,
                        'mean': float(np.mean(scores)) if scores else 0.5,
                    },
                    'note': 'This is a sampled subset. Full data in benchmarks/full/'
                }
                with open(sampled_meta, 'w', encoding='utf-8') as mf:
                    json.dump(meta, mf, indent=2, ensure_ascii=False)
            except Exception:
                pass

            logger.info(f"✓ Saved sampled benchmark ({len(sampled_tasks)} tasks) to {sampled_jsonl}")
            results[benchmark_name] = sampled_tasks

        return results

    def _clone_task_for_stream(self, task: Task, new_task_id: str) -> Task:
        """
        Clone a Task for use in a generated stream.
        
        Important: sampling uses replacement, so we must not mutate shared Task objects.
        """
        return Task(
            task_id=new_task_id,
            benchmark=task.benchmark,
            difficulty_score=task.difficulty_score,
            difficulty_bin=task.difficulty_bin,
            raw_data=copy.deepcopy(task.raw_data),
            scorer_metadata=copy.deepcopy(task.scorer_metadata),
        )
    
    def _load_raw_data(self, benchmark_name: str, cfg: Dict, full_dataset: bool = False) -> List[Dict]:
        """
        Load raw data from dataset source.
        
        Args:
            benchmark_name: Name of the benchmark
            cfg: Configuration dict for this benchmark
            full_dataset: If True, load entire dataset; if False, sample according to n_problems
        """
        cache_dir = Path("data/hf_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        if benchmark_name == 'humaneval':
            try:
                from datasets import load_dataset
                ds = load_dataset(
                    'openai_humaneval',
                    cache_dir=str(cache_dir),
                    download_mode='force_redownload',
                )
                return list(ds['test'])
            except Exception as e:
                logger.error(f"Failed to load HumanEval: {e}")
                return []
        
        elif benchmark_name == 'gsm8k':
            try:
                from datasets import load_dataset
                ds = load_dataset(
                    'gsm8k',
                    'main',
                    cache_dir=str(cache_dir),
                    download_mode='force_redownload',
                )
                # Load full test set if full_dataset=True, else sample
                if full_dataset:
                    return [dict(item) for item in ds['test']]
                else:
                    n_samples = cfg.get('n_problems', 200)
                    test_data = ds['test'].select(range(min(n_samples, len(ds['test']))))
                    return [dict(item) for item in test_data]
            except Exception as e:
                logger.error(f"Failed to load GSM8K: {e}")
                return []
        
        elif benchmark_name == 'bbh':
            try:
                from datasets import load_dataset
                # BBH has multiple task configs - load a subset of them
                configs = cfg.get('configs', [
                    'boolean_expressions', 'causal_judgement', 'date_understanding',
                    'disambiguation_qa', 'logical_deduction_three_objects',
                    'movie_recommendation', 'navigate', 'sports_understanding',
                    'tracking_shuffled_objects_three_objects', 'web_of_lies'
                ])
                
                all_data = []
                
                for config in configs:
                    try:
                        ds = load_dataset(
                            'lukaemon/bbh',
                            config,
                            cache_dir=str(cache_dir),
                            download_mode='force_redownload',
                        )
                        
                        # Load full or sample
                        if full_dataset:
                            config_data = ds['test']
                        else:
                            n_per_config = cfg.get('n_problems', 100) // len(configs)
                            config_data = ds['test'].select(range(min(n_per_config, len(ds['test']))))
                        
                        # Add task_name to each item
                        for item in config_data:
                            item['task_name'] = config
                        all_data.extend([dict(item) for item in config_data])
                    except Exception as e:
                        logger.warning(f"Failed to load BBH config {config}: {e}")
                        continue
                
                return all_data
            except Exception as e:
                logger.error(f"Failed to load BBH: {e}")
                return []
        
        elif benchmark_name == 'amc':
            try:
                from datasets import load_dataset
                ds = load_dataset(
                    'AI-MO/aimo-validation-amc',
                    cache_dir=str(cache_dir),
                    download_mode='force_redownload',
                )
                # Load full or sample
                if full_dataset:
                    return [dict(item) for item in ds['train']]
                else:
                    n_samples = cfg.get('n_problems', 50)
                    data = ds['train'].select(range(min(n_samples, len(ds['train']))))
                    return [dict(item) for item in data]
            except Exception as e:
                logger.error(f"Failed to load AMC: {e}")
                return []
        
        elif benchmark_name == 'medical_qa':
            try:
                from datasets import load_dataset
                ds = load_dataset(
                    'GBaker/MedQA-USMLE-4-options',
                    cache_dir=str(cache_dir),
                    download_mode='force_redownload',
                )
                # Load full test set or sample
                if full_dataset:
                    return [dict(item) for item in ds['test']]
                else:
                    n_samples = cfg.get('n_problems', 100)
                    test_data = ds['test'].select(range(min(n_samples, len(ds['test']))))
                    return [dict(item) for item in test_data]
            except Exception as e:
                logger.error(f"Failed to load Medical QA: {e}")
                return []
        
        else:
            logger.warning(f"Data loading not implemented for {benchmark_name}")
            return []
    
    def build_task_stream(
        self,
        benchmarks_to_include: List[str],
        difficulty_split: str = '50:50',
        n_total_tasks: int = 1000,
        random_seed: int = 2025,
        exclude_benchmarks: List[str] = None,
        # NEW: control difficulty binning and sampling
        difficulty_percentiles: Optional[Dict[str, Tuple[float, float]]] = None,
        benchmark_ratios: Optional[Dict[str, float]] = None,
        sample_with_replacement: Optional[bool] = None,
    ) -> List[Task]:
        """
        Build a task stream directly from full preprocessed datasets.
        
        New streamlined approach:
        1. Load full preprocessed datasets (all tasks)
        2. Compute difficulty thresholds from full datasets (stable, accurate)
        3. Assign difficulty bins using full thresholds
        4. Sample tasks directly from full datasets for experiment
        
        Args:
            benchmarks_to_include: Which benchmarks to include
            difficulty_split: "easy_pct:hard_pct" (e.g., "80:20", "50:50")
            n_total_tasks: Total number of tasks in experiment stream
            random_seed: Random seed for reproducibility
            exclude_benchmarks: Benchmarks to exclude (for mixing rules)
            difficulty_percentiles: Dict {benchmark: (low_pct, high_pct)} for custom bins (default: [20, 80])
            benchmark_ratios: Dict {benchmark: ratio} summing to 1.0 (default: equal split)
            sample_with_replacement: Whether to sample with replacement (default: from config or False)
            
        Returns:
            List of Task objects ready for experiment
        """
        # Load stream-generation defaults from config if parameters not provided
        cfg_stream = self.stream_cfg
        if difficulty_percentiles is None:
            difficulty_percentiles = cfg_stream.get('difficulty_percentiles', {})
        if benchmark_ratios is None:
            benchmark_ratios = cfg_stream.get('benchmark_ratios', {})
        if sample_with_replacement is None:
            sample_with_replacement = cfg_stream.get('sample_with_replacement', False)
        
        logger.info(f"Building task stream: {', '.join(benchmarks_to_include)}")
        logger.info(f"Difficulty split: {difficulty_split}, Total tasks: {n_total_tasks}")
        logger.info(f"Using full-dataset thresholds for accurate binning")

        if exclude_benchmarks:
            benchmarks_to_include = [b for b in benchmarks_to_include if b not in set(exclude_benchmarks)]
            logger.info(f"After exclusions: {', '.join(benchmarks_to_include)}")
        
        # Validate mixing rules
        self._validate_mixing_rules(benchmarks_to_include)
        
        # STEP 1: Load full preprocessed benchmarks
        logger.info("Step 1: Loading full preprocessed datasets...")
        benchmark_tasks: Dict[str, List[Task]] = {}
        full_dir = Path('data/benchmarks/full')
        
        for bn in benchmarks_to_include:
            cfg = (self.config.get('benchmarks') or {}).get(bn)
            if cfg is None or not cfg.get('enabled', False):
                raise ValueError(
                    f"Benchmark {bn} is disabled or not found in config. "
                    f"Enable it in config/data_config.yaml before generating streams."
                )

            # Load from full/ directory
            full_jsonl = full_dir / f"{bn}_full.jsonl"
            if not full_jsonl.exists():
                logger.warning(f"Full preprocessed data not found for {bn}. Running preprocessing...")
                self.preprocess_all_benchmarks(
                    output_dir='data/benchmarks/full',
                    force_reprocess=False
                )
            
            try:
                tasks = self._load_tasks_from_jsonl(str(full_jsonl))
                logger.info(f"  Loaded {len(tasks)} tasks from {bn}")
                benchmark_tasks[bn] = tasks
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load full preprocessed data for {bn}: {e}. "
                    f"Run preprocess_all_benchmarks() first."
                ) from e
        
        # STEP 2: Compute difficulty thresholds from FULL datasets
        logger.info("Step 2: Computing difficulty thresholds from full datasets...")
        thresholds: Dict[str, Dict[str, float]] = {}
        
        for bn, tasks in benchmark_tasks.items():
            if len(tasks) == 0:
                continue
            
            scores = [t.difficulty_score for t in tasks]
            pct_cfg = difficulty_percentiles.get(bn, [20, 80])  # Default: bottom 20%, top 20%
            
            if isinstance(pct_cfg, (list, tuple)) and len(pct_cfg) == 2:
                low_pct, high_pct = pct_cfg
                threshold_low = float(np.percentile(scores, low_pct))
                threshold_high = float(np.percentile(scores, high_pct))
            else:
                # Fallback: 50th percentile
                threshold_low = float(np.percentile(scores, 50))
                threshold_high = float(np.percentile(scores, 50))
            
            thresholds[bn] = {
                'easy': threshold_low,
                'hard': threshold_high,
            }
            logger.info(f"  {bn}: easy ≤ {threshold_low:.3f}, hard ≥ {threshold_high:.3f}")
        
        # STEP 3: Assign difficulty bins using full thresholds
        logger.info("Step 3: Assigning difficulty bins to tasks...")
        for bn, tasks in benchmark_tasks.items():
            threshold_low = thresholds[bn]['easy']
            threshold_high = thresholds[bn]['hard']
            
            for task in tasks:
                if task.difficulty_score <= threshold_low:
                    task.difficulty_bin = 'easy'
                elif task.difficulty_score >= threshold_high:
                    task.difficulty_bin = 'hard'
                else:
                    task.difficulty_bin = 'medium'
            
            easy_count = sum(1 for t in tasks if t.difficulty_bin == 'easy')
            hard_count = sum(1 for t in tasks if t.difficulty_bin == 'hard')
            medium_count = sum(1 for t in tasks if t.difficulty_bin == 'medium')
            logger.info(f"  {bn}: {easy_count} easy, {hard_count} hard, {medium_count} medium")

        # STEP 4: Sample tasks directly from full datasets for experiment
        logger.info("Step 4: Sampling tasks for experiment stream...")
        easy_pct, hard_pct = map(int, difficulty_split.split(':'))
        
        # Normalize benchmark ratios if provided
        if benchmark_ratios:
            ratio_sum = sum(benchmark_ratios.values())
            if not math.isclose(ratio_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3):
                logger.warning(f"benchmark_ratios sum to {ratio_sum:.3f}, normalizing to 1.0")
                benchmark_ratios = {k: v / ratio_sum for k, v in benchmark_ratios.items()}
        
        # Calculate tasks per benchmark
        def tasks_for_benchmark(bn: str) -> int:
            if benchmark_ratios and bn in benchmark_ratios:
                return max(0, int(n_total_tasks * benchmark_ratios[bn]))
            return max(0, n_total_tasks // len(benchmarks_to_include))
        
        np.random.seed(random_seed)
        sampled_all: List[Task] = []
        
        # Sample per benchmark (keep separate)
        for bn in benchmarks_to_include:
            tasks = benchmark_tasks.get(bn, [])
            if len(tasks) == 0:
                logger.warning(f"No tasks for benchmark {bn}; skipping.")
                continue
            
            n_for_bn = tasks_for_benchmark(bn)
            if n_for_bn == 0:
                continue
            
            # Separate by difficulty bin
            bn_easy = [t for t in tasks if t.difficulty_bin == 'easy']
            bn_hard = [t for t in tasks if t.difficulty_bin == 'hard']
            
            # Calculate split
            n_easy = int(n_for_bn * easy_pct / (easy_pct + hard_pct))
            n_hard = n_for_bn - n_easy
            
            # Sample from bins
            def sample_pool(pool: List[Task], n: int, pool_name: str) -> List[Task]:
                if n <= 0:
                    return []
                if len(pool) == 0:
                    logger.warning(f"  {bn}: {pool_name} pool is empty, cannot sample {n} tasks")
                    return []
                
                replace_flag = sample_with_replacement or len(pool) < n
                if replace_flag and len(pool) < n:
                    logger.info(f"  {bn}: sampling {n} from {len(pool)} {pool_name} tasks (with replacement)")
                
                idx = np.random.choice(len(pool), n, replace=replace_flag).tolist()
                return [pool[i] for i in idx]
            
            sampled_easy = sample_pool(bn_easy, n_easy, 'easy')
            sampled_hard = sample_pool(bn_hard, n_hard, 'hard')
            sampled_all.extend(sampled_easy + sampled_hard)
            
            logger.info(f"  {bn}: sampled {len(sampled_easy)} easy + {len(sampled_hard)} hard = {len(sampled_easy) + len(sampled_hard)} tasks")

        if len(sampled_all) == 0:
            logger.error("Sampling produced zero tasks. Check configuration.")
            return []

        # STEP 5: Shuffle and clone sampled tasks
        logger.info("Step 5: Finalizing experiment stream...")
        np.random.shuffle(sampled_all)

        # Clone tasks to avoid mutation of cached data
        task_stream: List[Task] = []
        for i, task in enumerate(sampled_all):
            task_stream.append(self._clone_task_for_stream(task, f"exp_{i:05d}_{task.benchmark}"))
        
        easy_count = sum(1 for t in task_stream if t.difficulty_bin == 'easy')
        hard_count = sum(1 for t in task_stream if t.difficulty_bin == 'hard')
        
        logger.info(
            f"✓ Generated experiment stream: {len(task_stream)} tasks total "
            f"({easy_count} easy, {hard_count} hard)"
        )
        
        # Log per-benchmark breakdown
        for bn in benchmarks_to_include:
            bn_tasks = [t for t in task_stream if t.benchmark == bn]
            if bn_tasks:
                logger.info(f"  {bn}: {len(bn_tasks)} tasks")
        
        return task_stream
    
    def _validate_mixing_rules(self, benchmarks: List[str]):
        """Check if benchmark combination violates exclusion rules"""
        rules = self.config.get('mixing_rules', {})
        exclude_combos = rules.get('exclude_combinations', [])
        
        for excluded_pair in exclude_combos:
            if set(excluded_pair).issubset(set(benchmarks)):
                raise ValueError(
                    f"Cannot mix {excluded_pair}: incompatible benchmarks per config"
                )
    
    def save_task_pool(
        self,
        task_stream: List[Task],
        output_path: str = 'data/task_pool.jsonl'
    ):
        """Save task stream as JSONL for team use"""
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for task in task_stream:
                f.write(json.dumps(task.to_dict(), ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(task_stream)} tasks to {output_path}")
    
    def save_statistics(
        self,
        task_stream: List[Task],
        output_path: str = 'data/task_statistics.json'
    ):
        """Save difficulty statistics for analysis"""
        
        # Handle empty task stream
        if len(task_stream) == 0:
            logger.warning("Task stream is empty. Saving empty statistics.")
            stats = {
                'n_total': 0,
                'benchmarks': [],
                'difficulty_distribution': {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                },
                'difficulty_bins': {
                    'easy': 0,
                    'hard': 0,
                },
                'benchmark_breakdown': {}
            }
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved empty statistics to {output_path}")
            return
        
        stats = {
            'n_total': len(task_stream),
            'benchmarks': list(set(t.benchmark for t in task_stream)),
            'difficulty_distribution': {
                'mean': float(np.mean([t.difficulty_score for t in task_stream])),
                'std': float(np.std([t.difficulty_score for t in task_stream])),
                'min': float(np.min([t.difficulty_score for t in task_stream])),
                'max': float(np.max([t.difficulty_score for t in task_stream])),
            },
            'difficulty_bins': {
                'easy': sum(1 for t in task_stream if t.difficulty_bin == 'easy'),
                'hard': sum(1 for t in task_stream if t.difficulty_bin == 'hard'),
            },
            'benchmark_breakdown': {}
        }
        
        # Count by benchmark
        for benchmark in stats['benchmarks']:
            tasks_in_bn = [t for t in task_stream if t.benchmark == benchmark]
            stats['benchmark_breakdown'][benchmark] = {
                'count': len(tasks_in_bn),
                'mean_difficulty': float(np.mean([t.difficulty_score for t in tasks_in_bn])),
                'easy': sum(1 for t in tasks_in_bn if t.difficulty_bin == 'easy'),
                'hard': sum(1 for t in tasks_in_bn if t.difficulty_bin == 'hard'),
            }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved statistics to {output_path}")
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"TASK POOL STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total tasks: {stats['n_total']}")
        logger.info(f"Benchmarks: {', '.join(stats['benchmarks'])}")
        logger.info(f"Mean difficulty: {stats['difficulty_distribution']['mean']:.3f}")
        logger.info(f"Std difficulty: {stats['difficulty_distribution']['std']:.3f}")
        logger.info(f"Easy tasks: {stats['difficulty_bins']['easy']}")
        logger.info(f"Hard tasks: {stats['difficulty_bins']['hard']}")
        logger.info(f"\nBreakdown by benchmark:")
        for bn, info in stats['benchmark_breakdown'].items():
            logger.info(f"  {bn}: {info['count']} tasks (mean_diff={info['mean_difficulty']:.3f})")
        logger.info(f"{'='*60}\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Example 1: Build task stream for Exp 1
    logger.info("=" * 70)
    logger.info("EXAMPLE 1: Build task stream for Exp 1 (80% easy, 20% hard)")
    logger.info("=" * 70)
    
    builder = DatasetBuilder('config/data_config.yaml')
    
    try:
        exp1_tasks = builder.build_task_stream(
            benchmarks_to_include=['humaneval', 'gsm8k'],
            difficulty_split='80:20',
            n_total_tasks=1000,
            random_seed=2025
        )
        
        builder.save_task_pool(exp1_tasks, 'data/exp1_task_pool.jsonl')
        builder.save_statistics(exp1_tasks, 'data/exp1_statistics.json')
        
    except Exception as e:
        logger.error(f"Error building Exp 1 task stream: {e}")
    
    # Example 2: Build task stream for Exp 5 (all benchmarks)
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 2: Build task stream for Exp 5 (all benchmarks, 50/50)")
    logger.info("=" * 70)
    
    try:
        exp5_tasks = builder.build_task_stream(
            benchmarks_to_include=['humaneval', 'gsm8k', 'bbh', 'amc', 'medical_qa'],
            difficulty_split='50:50',
            n_total_tasks=2000,
            random_seed=2025
        )
        
        builder.save_task_pool(exp5_tasks, 'data/exp5_task_pool.jsonl')
        builder.save_statistics(exp5_tasks, 'data/exp5_statistics.json')
        
    except Exception as e:
        logger.error(f"Error building Exp 5 task stream: {e}")
    
    # Example 3: Validate difficulty definitions (optional, requires API key)
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 3: Validate difficulty definitions (requires OpenAI API)")
    logger.info("=" * 70)
    
    if HAS_OPENAI and os.getenv('OPENAI_API_KEY'):
        builder.enable_validation(budget=2.0)  # $2 budget for validation
        
        try:
            # Validate HumanEval
            humaneval_validation = builder.validate_difficulties(
                exp1_tasks,
                benchmark='humaneval',
                n_samples=5  # Small sample for quick testing
            )
            
            # Validate GSM8K
            gsm8k_validation = builder.validate_difficulties(
                exp1_tasks,
                benchmark='gsm8k',
                n_samples=5
            )
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
    else:
        logger.warning("Skipping validation: OPENAI_API_KEY not set or openai not installed")
