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
    
    Definition: difficulty = 0.6 * (n_asserts / max_asserts) + 0.4 * (prompt_len / max_prompt_len)
    
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
    
    Definition: difficulty = reasoning_steps / max_reasoning_steps
    
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
    
    Definition: difficulty = base_complexity + 0.3 * (input_len / max_input_len)
    
    Rationale:
    - BBH has 23 tasks covering 10+ reasoning skills (deduction, causal, spatial, etc.)
    - Since no official difficulty labels exist, we use:
      1. Task-specific complexity (from design of BBH)
      2. Input length (more context = harder)
    
    References:
    - BIG-Bench Extra Hard (BBEH, 2025)
      "Tasks require many-hop reasoning, long-range dependency, dealing with distractors"
    - "Challenging BIG-Bench Tasks" (2023)
      Analyzes 10 core reasoning abilities across BBH
    """
    
    # Task complexity map (based on BBH task design)
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
    
    Definition: difficulty = 0.7 * (problem_len / max_problem_len) + 0.3 + 0.12 * has_latex
    
    Rationale:
    - Longer problems typically have more complex reasoning
    - Problems with LaTeX/fractions are often more advanced
    - Base difficulty of 0.3 reflects inherent competition problem difficulty
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
    
    Definition: difficulty = 0.4*question_len + 0.3*keywords + 0.2*option_len + 0.1*clinical_case
    
    Rationale:
    - Medical QA difficulty comes from multiple dimensions:
      1. Question length (longer = more complex scenarios)
      2. Medical keyword density (domain complexity)
      3. Option complexity (longer options = harder discrimination)
      4. Clinical case indicator (vignettes are harder)
    
    References:
    - MedReason-Dx (2025)
    - Medical QA Systems Survey
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
        
        # Weighted combination
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
    """
    
    def __init__(
        self,
        weak_model: str = 'gpt-3.5-turbo',
        api_key: Optional[str] = None,
        budget_limit: float = 5.0
    ):
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
        """Validate difficulty scores by running weak model on sample tasks."""
        
        if self.client is None:
            logger.warning("Cannot validate without OpenAI API key. Skipping validation.")
            return {'error': 'No OpenAI client'}
        
        logger.info(f"Starting validation for {benchmark} ({n_samples} samples)...")
        
        np.random.seed(random_seed)
        sampled_tasks = np.random.choice(tasks, min(n_samples, len(tasks)), replace=False)
        
        results = []
        total_cost = 0.0
        
        for i, task in enumerate(sampled_tasks):
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
        
        df = pd.DataFrame(results)
        
        if len(df) == 0:
            logger.warning("No validation results collected.")
            return {'error': 'No results'}
        
        correlation = df['difficulty_score'].corr(df['error_rate'])
        
        easy_tasks = df[df['difficulty_bin'] == 'easy']
        hard_tasks = df[df['difficulty_bin'] == 'hard']
        
        easy_accuracy = easy_tasks['accuracy'].mean() if len(easy_tasks) > 0 else 0.0
        hard_accuracy = hard_tasks['accuracy'].mean() if len(hard_tasks) > 0 else 0.0
        
        stats = {
            'benchmark': benchmark,
            'n_samples': len(df),
            'total_cost': total_cost,
            'correlation': correlation,
            'easy_accuracy': easy_accuracy,
            'hard_accuracy': hard_accuracy,
            'mean_difficulty': df['difficulty_score'].mean(),
            'std_difficulty': df['difficulty_score'].std(),
            'accuracy_gap': easy_accuracy - hard_accuracy,
            'results_df': df,
        }
        
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
        """Run weak model on a single task and compute accuracy."""
        
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
        raw = task.raw_data
        prompt = raw.get('prompt', '')
        test = raw.get('test', '')
        
        response = self.client.chat.completions.create(
            model=self.weak_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000
        )
        
        code = response.choices[0].message.content
        tokens = response.usage.total_tokens
        cost = tokens * (0.5 / 1e6)
        
        try:
            exec(code + "\n" + test)
            accuracy = 1.0
        except Exception:
            accuracy = 0.0
        
        return accuracy, cost
    
    def _eval_gsm8k(self, task: Task) -> Tuple[float, float]:
        raw = task.raw_data
        question = raw.get('question', '')
        answer = raw.get('answer', '')
        
        response = self.client.chat.completions.create(
            model=self.weak_model,
            messages=[{"role": "user", "content": question}],
            temperature=0.0,
            max_tokens=500
        )
        
        generated = response.choices[0].message.content
        tokens = response.usage.total_tokens
        cost = tokens * (0.5 / 1e6)
        
        correct_answer = str(answer).split()[-1]
        accuracy = 1.0 if correct_answer in generated else 0.0
        
        return accuracy, cost
    
    def _eval_bbh(self, task: Task) -> Tuple[float, float]:
        raw = task.raw_data
        input_text = raw.get('input', '')
        target = raw.get('target', '')
        
        response = self.client.chat.completions.create(
            model=self.weak_model,
            messages=[{"role": "user", "content": f"Answer: {input_text}"}],
            temperature=0.0,
            max_tokens=100
        )
        
        generated = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens
        cost = tokens * (0.5 / 1e6)
        
        accuracy = 1.0 if target.strip() in generated else 0.0
        
        return accuracy, cost
    
    def _eval_amc(self, task: Task) -> Tuple[float, float]:
        raw = task.raw_data
        problem_text = raw.get('problem', '')
        answer = raw.get('answer', '')
        
        response = self.client.chat.completions.create(
            model=self.weak_model,
            messages=[{"role": "user", "content": problem_text}],
            temperature=0.0,
            max_tokens=200
        )
        
        generated = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens
        cost = tokens * (0.5 / 1e6)
        
        accuracy = 1.0 if str(answer) in generated else 0.0
        
        return accuracy, cost
    
    def _eval_medical_qa(self, task: Task) -> Tuple[float, float]:
        raw = task.raw_data
        question = raw.get('question', '')
        answer = raw.get('answer', '')
        
        response = self.client.chat.completions.create(
            model=self.weak_model,
            messages=[{"role": "user", "content": question}],
            temperature=0.0,
            max_tokens=500
        )
        
        generated = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens
        cost = tokens * (0.5 / 1e6)
        
        key_term = answer.split('\n')[0].split()[-1] if answer else ''
        accuracy = 1.0 if key_term.lower() in generated.lower() else 0.0
        
        return accuracy, cost


# ============================================================================
# DATASET BUILDER (Main orchestrator)
# ============================================================================

class DatasetBuilder:
    """
    Main orchestrator for building task streams with flexible difficulty control.
    """
    
    def __init__(self, config_path: str = 'config/data_config.yaml'):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.stream_cfg = self.config.get('stream_generation', {}) or {}
        self.scorers = self._init_scorers()
        self.validator = None
        
        logger.info(f"Initialized DatasetBuilder with config: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._default_config()

        with open(config_path, 'r', encoding='utf-8-sig') as f:
            return yaml.safe_load(f)

    def _default_config(self) -> Dict[str, Any]:
        return {
            'benchmarks': {
                'humaneval': {'path': 'datasets:openai_humaneval', 'enabled': True},
                'gsm8k': {'path': 'datasets:gsm8k/main', 'enabled': True},
                'bbh': {'path': 'datasets:big_bench', 'enabled': False},
                'amc': {'path': 'custom:amc_dataset', 'enabled': False},
                'medical_qa': {'path': 'custom:medical_qa', 'enabled': False},
            },
            'mixing_rules': {'exclude_combinations': []},
        }
    
    def _init_scorers(self) -> Dict[str, BaseDifficultyScorer]:
        return {
            'humaneval': HumanEvalDifficultyScorer('humaneval'),
            'gsm8k': GSM8KDifficultyScorer('gsm8k'),
            'bbh': BBHDifficultyScorer('bbh'),
            'amc': AMCDifficultyScorer('amc'),
            'medical_qa': MedicalQADifficultyScorer('medical_qa'),
        }
    
    def enable_validation(self, api_key: Optional[str] = None, budget: float = 5.0):
        self.validator = DifficultyValidator(
            weak_model='gpt-3.5-turbo',
            api_key=api_key,
            budget_limit=budget
        )
        logger.info("Difficulty validation enabled")
    
    def validate_difficulties(self, tasks: List[Task], benchmark: str, n_samples: int = 20) -> Dict[str, Any]:
        if self.validator is None:
            logger.warning("Validation not enabled. Call enable_validation() first.")
            return {}
        return self.validator.validate(tasks, benchmark, n_samples)
    
    def validate_all_benchmarks(self, tasks: List[Task], n_samples: int = 20, budget_limit: float = 2.0) -> Dict[str, Dict[str, Any]]:
        if self.validator is None:
            logger.warning("Validation not enabled. Call enable_validation() first.")
            return {}
        
        tasks_by_benchmark = defaultdict(list)
        for task in tasks:
            tasks_by_benchmark[task.benchmark].append(task)
        
        results = {}
        total_cost = 0.0
        
        for benchmark, benchmark_tasks in tasks_by_benchmark.items():
            if total_cost >= budget_limit:
                results[benchmark] = {'error': 'Budget exceeded'}
                continue
            
            try:
                stats = self.validate_difficulties(benchmark_tasks, benchmark=benchmark, n_samples=min(n_samples, len(benchmark_tasks)))
                if 'error' not in stats:
                    total_cost += stats.get('total_cost', 0.0)
                results[benchmark] = stats
            except Exception as e:
                results[benchmark] = {'error': str(e)}
        
        return results

    def _benchmark_config_fingerprint(self, benchmark_name: str) -> str:
        cfg = (self.config.get('benchmarks') or {}).get(benchmark_name, {})
        blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode('utf-8')).hexdigest()

    def _load_tasks_from_jsonl(self, jsonl_path: str) -> List[Task]:
        tasks: List[Task] = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tasks.append(Task(**obj))
        return tasks

    def _compute_normalization_constants(self, benchmark_name: str, all_metadata: List[Dict[str, Any]]) -> Dict[str, float]:
        if len(all_metadata) == 0:
            return {}
        
        constants = {}
        
        if benchmark_name == 'humaneval':
            constants['max_asserts'] = float(np.percentile([m['n_asserts'] for m in all_metadata], 95))
            constants['max_prompt_len'] = float(np.percentile([m['prompt_len'] for m in all_metadata], 95))
        elif benchmark_name == 'gsm8k':
            constants['max_reasoning_steps'] = float(np.percentile([m['reasoning_steps'] for m in all_metadata], 95))
        elif benchmark_name == 'bbh':
            constants['max_input_len'] = float(np.percentile([m['input_len'] for m in all_metadata], 95))
        elif benchmark_name == 'amc':
            constants['max_problem_len'] = float(np.percentile([m['problem_len'] for m in all_metadata], 95))
        elif benchmark_name == 'medical_qa':
            constants['max_question_words'] = float(np.percentile([m['question_words'] for m in all_metadata], 95))
            constants['max_keywords'] = float(np.percentile([m['n_keywords'] for m in all_metadata], 95))
            constants['max_option_len'] = float(np.percentile([m['avg_option_len'] for m in all_metadata], 95))
        
        return constants

    def preprocess_all_benchmarks(self, output_dir: str = 'data/benchmarks/full', force_reprocess: bool = False) -> Dict[str, List[Task]]:
        """One-time preprocessing step - saves FULL preprocessed benchmarks."""
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
                    logger.warning(f"Existing file unreadable for {benchmark_name} ({e}). Reprocessing...")

            logger.info(f"Preprocessing benchmark: {benchmark_name}")
            raw_data = self._load_raw_data(benchmark_name, cfg, full_dataset=True)

            if len(raw_data) == 0:
                logger.warning(f"No raw data loaded for {benchmark_name}. Skipping.")
                continue

            scorer = self.scorers.get(benchmark_name)
            if scorer is None:
                continue

            # Extract metadata and compute normalization constants
            all_metadata = [scorer.extract_metadata(raw_task) for raw_task in raw_data]
            norm_constants = self._compute_normalization_constants(benchmark_name, all_metadata)
            logger.info(f"Computed normalization constants: {norm_constants}")
            
            # Compute raw scores
            tasks: List[Task] = []
            raw_scores = []
            
            for idx, (raw_task, metadata) in enumerate(zip(raw_data, all_metadata)):
                task_id = f"{benchmark_name}_{idx}"
                raw_score = float(scorer.score(raw_task, norm_constants))
                raw_scores.append(raw_score)
                
                tasks.append(Task(
                    task_id=task_id,
                    benchmark=benchmark_name,
                    difficulty_score=raw_score,
                    difficulty_bin='',
                    raw_data=raw_task,
                    scorer_metadata=metadata,
                ))
            
            # Min-max normalization
            if len(raw_scores) > 0:
                min_score, max_score = min(raw_scores), max(raw_scores)
                score_range = max_score - min_score
                
                if score_range > 0:
                    for task, raw_score in zip(tasks, raw_scores):
                        task.difficulty_score = (raw_score - min_score) / score_range
                else:
                    for task in tasks:
                        task.difficulty_score = 0.5
            
            # Save
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for task in tasks:
                    f.write(json.dumps(task.to_dict(), ensure_ascii=False) + '\n')

            logger.info(f"✓ Saved full preprocessed benchmark ({len(tasks)} tasks) to {jsonl_path}")
            results[benchmark_name] = tasks

        return results

    def _clone_task_for_stream(self, task: Task, new_task_id: str) -> Task:
        return Task(
            task_id=new_task_id,
            benchmark=task.benchmark,
            difficulty_score=task.difficulty_score,
            difficulty_bin=task.difficulty_bin,
            raw_data=copy.deepcopy(task.raw_data),
            scorer_metadata=copy.deepcopy(task.scorer_metadata),
        )
    
    def _load_raw_data(self, benchmark_name: str, cfg: Dict, full_dataset: bool = False) -> List[Dict]:
        cache_dir = Path("data/hf_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        if benchmark_name == 'humaneval':
            try:
                from datasets import load_dataset
                ds = load_dataset('openai_humaneval', cache_dir=str(cache_dir))
                return list(ds['test'])
            except Exception as e:
                logger.error(f"Failed to load HumanEval: {e}")
                return []
        
        elif benchmark_name == 'gsm8k':
            try:
                from datasets import load_dataset
                ds = load_dataset('gsm8k', 'main', cache_dir=str(cache_dir))
                return [dict(item) for item in ds['test']]
            except Exception as e:
                logger.error(f"Failed to load GSM8K: {e}")
                return []
        
        elif benchmark_name == 'bbh':
            try:
                from datasets import load_dataset
                configs = cfg.get('configs', [
                    'boolean_expressions', 'causal_judgement', 'date_understanding',
                    'disambiguation_qa', 'logical_deduction_three_objects',
                    'movie_recommendation', 'navigate', 'sports_understanding',
                    'tracking_shuffled_objects_three_objects', 'web_of_lies'
                ])
                
                all_data = []
                for config in configs:
                    try:
                        ds = load_dataset('lukaemon/bbh', config, cache_dir=str(cache_dir))
                        for item in ds['test']:
                            item['task_name'] = config
                        all_data.extend([dict(item) for item in ds['test']])
                    except Exception as e:
                        logger.warning(f"Failed to load BBH config {config}: {e}")
                
                return all_data
            except Exception as e:
                logger.error(f"Failed to load BBH: {e}")
                return []
        
        elif benchmark_name == 'amc':
            try:
                from datasets import load_dataset
                ds = load_dataset('AI-MO/aimo-validation-amc', cache_dir=str(cache_dir))
                return [dict(item) for item in ds['train']]
            except Exception as e:
                logger.error(f"Failed to load AMC: {e}")
                return []
        
        elif benchmark_name == 'medical_qa':
            try:
                from datasets import load_dataset
                ds = load_dataset('GBaker/MedQA-USMLE-4-options', cache_dir=str(cache_dir))
                return [dict(item) for item in ds['test']]
            except Exception as e:
                logger.error(f"Failed to load Medical QA: {e}")
                return []
        
        return []
    
    def build_task_stream(
        self,
        benchmarks_to_include: List[str],
        difficulty_split: str = '50:50',
        n_total_tasks: int = 1000,
        random_seed: int = 2025,
        exclude_benchmarks: List[str] = None,
        difficulty_percentiles: Optional[Dict[str, Tuple[float, float]]] = None,
        benchmark_ratios: Optional[Dict[str, float]] = None,
        sample_with_replacement: Optional[bool] = None,
    ) -> List[Task]:
        """Build a task stream directly from full preprocessed datasets."""
        
        cfg_stream = self.stream_cfg
        if difficulty_percentiles is None:
            difficulty_percentiles = cfg_stream.get('difficulty_percentiles', {})
        if sample_with_replacement is None:
            sample_with_replacement = cfg_stream.get('sample_with_replacement', False)
        
        logger.info(f"Building task stream: {', '.join(benchmarks_to_include)}")
        logger.info(f"Difficulty split: {difficulty_split}, Total tasks: {n_total_tasks}")

        if exclude_benchmarks:
            benchmarks_to_include = [b for b in benchmarks_to_include if b not in set(exclude_benchmarks)]
        
        # Load full preprocessed benchmarks
        benchmark_tasks: Dict[str, List[Task]] = {}
        full_dir = Path('data/benchmarks/full')
        
        for bn in benchmarks_to_include:
            full_jsonl = full_dir / f"{bn}_full.jsonl"
            if not full_jsonl.exists():
                self.preprocess_all_benchmarks(output_dir='data/benchmarks/full', force_reprocess=False)
            
            tasks = self._load_tasks_from_jsonl(str(full_jsonl))
            logger.info(f"  Loaded {len(tasks)} tasks from {bn}")
            benchmark_tasks[bn] = tasks
        
        # Compute thresholds from FULL datasets
        thresholds: Dict[str, Dict[str, float]] = {}
        for bn, tasks in benchmark_tasks.items():
            scores = [t.difficulty_score for t in tasks]
            pct_cfg = difficulty_percentiles.get(bn, [20, 80])
            
            low_pct, high_pct = pct_cfg if isinstance(pct_cfg, (list, tuple)) else (50, 50)
            threshold_low = float(np.percentile(scores, low_pct))
            threshold_high = float(np.percentile(scores, high_pct))
            
            thresholds[bn] = {'easy': threshold_low, 'hard': threshold_high}
            logger.info(f"  {bn}: easy ≤ {threshold_low:.3f}, hard ≥ {threshold_high:.3f}")
        
        # Assign difficulty bins
        for bn, tasks in benchmark_tasks.items():
            for task in tasks:
                if task.difficulty_score <= thresholds[bn]['easy']:
                    task.difficulty_bin = 'easy'
                elif task.difficulty_score >= thresholds[bn]['hard']:
                    task.difficulty_bin = 'hard'
                else:
                    task.difficulty_bin = 'medium'

        # Sample tasks
        easy_pct, hard_pct = map(int, difficulty_split.split(':'))
        
        np.random.seed(random_seed)
        sampled_all: List[Task] = []
        
        for bn in benchmarks_to_include:
            tasks = benchmark_tasks.get(bn, [])
            if not tasks:
                continue
            
            n_for_bn = n_total_tasks // len(benchmarks_to_include)
            if benchmark_ratios and bn in benchmark_ratios:
                n_for_bn = int(n_total_tasks * benchmark_ratios[bn])
            
            bn_easy = [t for t in tasks if t.difficulty_bin == 'easy']
            bn_hard = [t for t in tasks if t.difficulty_bin == 'hard']
            
            n_easy = int(n_for_bn * easy_pct / (easy_pct + hard_pct))
            n_hard = n_for_bn - n_easy
            
            def sample_pool(pool, n):
                if n <= 0 or not pool:
                    return []
                replace = sample_with_replacement or len(pool) < n
                idx = np.random.choice(len(pool), n, replace=replace).tolist()
                return [pool[i] for i in idx]
            
            sampled_all.extend(sample_pool(bn_easy, n_easy))
            sampled_all.extend(sample_pool(bn_hard, n_hard))
        
        np.random.shuffle(sampled_all)
        
        task_stream = [self._clone_task_for_stream(task, f"exp_{i:05d}_{task.benchmark}") for i, task in enumerate(sampled_all)]
        
        logger.info(f"✓ Generated {len(task_stream)} tasks")
        return task_stream

    def save_task_pool(self, task_stream: List[Task], output_path: str = 'data/task_pool.jsonl'):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for task in task_stream:
                f.write(json.dumps(task.to_dict(), ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(task_stream)} tasks to {output_path}")
    
    def save_statistics(self, task_stream: List[Task], output_path: str = 'data/task_statistics.json'):
        if not task_stream:
            return
        
        stats = {
            'n_total': len(task_stream),
            'benchmarks': list(set(t.benchmark for t in task_stream)),
            'difficulty_distribution': {
                'mean': float(np.mean([t.difficulty_score for t in task_stream])),
                'std': float(np.std([t.difficulty_score for t in task_stream])),
            },
            'difficulty_bins': {
                'easy': sum(1 for t in task_stream if t.difficulty_bin == 'easy'),
                'hard': sum(1 for t in task_stream if t.difficulty_bin == 'hard'),
            },
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved statistics to {output_path}")


if __name__ == '__main__':
    builder = DatasetBuilder('config/data_config.yaml')
    
    try:
        tasks = builder.build_task_stream(
            benchmarks_to_include=['humaneval', 'gsm8k'],
            difficulty_split='80:20',
            n_total_tasks=1000,
            random_seed=2025
        )
        builder.save_task_pool(tasks, 'data/exp1_task_pool.jsonl')
        builder.save_statistics(tasks, 'data/exp1_statistics.json')
    except Exception as e:
        logger.error(f"Error: {e}")
