#!/usr/bin/env python3
"""
User Registration and Task Management Module

This module provides the UserRunner class for managing Symphony Network users.
It handles task distribution, result collection, and voting mechanisms for
distributed task execution across the network.

The UserRunner manages:
- Task assignment to network agents
- Result collection and aggregation
- Majority voting on multiple Chain-of-Thought (CoT) responses
- Batch task processing with persistence
- Timeout management for long-running tasks

Author: Symphony Network Team
License: MIT
"""

import time
import threading
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

from runtime.config import load_config_from_file
from agents.user import User


class UserRunner:
    """
    User runner for managing task execution and result collection.
    
    This class orchestrates the distribution of tasks across the Symphony Network
    and collects results from multiple agents. It implements a voting mechanism
    to determine the best answer from multiple Chain-of-Thought executions.
    
    Attributes:
        config (dict): Configuration loaded from the config file
        user (User): The Symphony user instance
        cot_num (int): Number of CoT executions for voting
        running (bool): Flag indicating if the runner is active
        answers (List[str]): Results of all completed tasks
        current_answers (List[str]): Results from all CoTs of the current task
        full_answers (List[Dict]): Detailed results including previous steps
        result_event (threading.Event): Synchronization for result collection
        lock (threading.Lock): Thread-safe result manipulation
        final_result (str): The voted final result of current task
    """
    
    def __init__(self, config_path: str, cot_num: int) -> None:
        """
        Initialize user runner with configuration and CoT settings.
        
        Args:
            config_path (str): Path to the YAML configuration file
            cot_num (int): Number of Chain of Thought executions for voting
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If cot_num is less than 1
        """
        if cot_num < 1:
            raise ValueError("Chain of Thought number must be at least 1")
            
        # Load configuration and initialize User
        self.config = load_config_from_file(config_path)
        self.user = User(self.config, cot_num)
        self.cot_num = cot_num

        self.running = False  # Running state flag

        self.answers: List[str] = []  # Results of all tasks
        self.current_answers: List[str] = []  # Results from all CoTs of the current task
        self.full_answers: List[Dict[str, Any]] = []  # Detailed results with previous steps
        self.result_event = threading.Event()
        self.lock = threading.Lock()  # Thread-safe result addition
        self.final_result: str = ""
    
    def start(self) -> None:
        """
        Start the user runner with result listening loop.
        
        This method initiates the result collection process by starting
        a dedicated thread that listens for task results from network agents.
        The method is non-blocking and returns immediately after starting
        the listener thread.
        """
        self.running = True
        print(f"[STARTUP] User {self.user.user_id} started, listening for task results...")

        # Start result listening thread
        result_thread = threading.Thread(target=self._listen_results, daemon=True)
        result_thread.start()
    
    def _listen_results(self) -> None:
        """
        Continuously listen for task results and process them.
        
        This method runs in a separate thread and handles result collection
        from network agents. When enough results are collected (based on cot_num),
        it triggers the voting mechanism to determine the final answer.
        
        The method implements thread-safe result aggregation and notifies
        waiting threads when the voting process is complete.
        """
        while self.running:
            try:
                sender_id, msg_type, result = self.user.isep_client.receive_result(timeout=1)
                if msg_type == "task_result":
                    print(f"[RESULT] Received result from agent: {sender_id}")
                    answer = result["result"]
                    self.current_answers.append(answer)
                    
                    with self.lock:
                        self.full_answers.append(result['previous_results'])
                        print(f"[PROGRESS] Received {len(self.current_answers)}/{self.cot_num} answers")

                        if len(self.current_answers) >= self.cot_num:
                            self.final_result = self._vote_results()
                            self.answers.append(self.final_result)
                            print(f"[VOTING] Final result determined: {self.final_result}")
                            print("[DETAILED_RESULTS] All Chain-of-Thought execution results:")
                            for idx, detailed_result in enumerate(self.full_answers, 1):
                                print(f"  CoT {idx}: {detailed_result}")
                            self.result_event.set()

            except Exception as e:
                if self.running:
                    print(f"[RESULT_ERROR] Failed to process result: {str(e)}")
            time.sleep(0.2)

    def _vote_results(self) -> str:
        """
        Perform majority voting on the current answers.
        
        This method implements a simple majority voting mechanism to determine
        the best answer from multiple Chain-of-Thought executions. In case of
        ties, it returns the first most common answer.
        
        Returns:
            str: The answer that received the most votes
            
        Note:
            If all answers are unique, returns the first answer received.
        """
        if not self.current_answers:
            return "[NO_ANSWERS]"
            
        count = Counter(self.current_answers)
        most_common = count.most_common(1)[0]  # Returns format like ('result', 3)
        
        # Log voting details for transparency
        if len(count) > 1:
            print(f"[VOTING_DETAILS] Answer distribution: {dict(count)}")
        else:
            print("[VOTING_DETAILS] All answers were identical")
            
        return most_common[0]
    
    def process_task(self, task_description: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a single task and return the voted result.
        
        This method orchestrates the execution of a single task by:
        1. Clearing previous state
        2. Assigning the task to the network
        3. Waiting for results with timeout
        4. Returning the voted result and detailed answers
        
        Args:
            task_description (str): The task to be executed
            
        Returns:
            Tuple[str, List[Dict[str, Any]]]: Final result and detailed execution logs
            
        Note:
            This method blocks until all CoT executions complete or timeout occurs.
        """
        self.result_event.clear()
        self.user.initiate_task(task_description)

        def timeout_handler() -> None:
            """Handle task timeout by setting a timeout result."""
            if not self.result_event.is_set():
                print("[TIMEOUT] Task execution exceeded 5 minutes, marking as timeout")
                self.final_result = "[TIMEOUT]"
                self.result_event.set()
        
        # Set up 5-minute timeout
        timer = threading.Timer(300, timeout_handler)
        timer.start()

        print("[WAITING] Waiting for task results to return...")
        self.result_event.wait()
        timer.cancel()

        return self.final_result, self.full_answers
    
    def process_task_batch(self, tasks: List[Dict[str, Any]], output_path: str, 
                          max_tasks: Optional[int] = None) -> None:
        """
        Process a batch of tasks with persistent result saving.
        
        This method handles batch processing of multiple tasks with:
        - Progressive result saving every 5 tasks
        - Individual task state management
        - Error recovery and logging
        - Optional task limiting for testing
        
        Args:
            tasks (List[Dict[str, Any]]): List of tasks with 'problem' and 'unique_id' fields
            output_path (str): Path to save results in JSONL format
            max_tasks (Optional[int]): Maximum number of tasks to process (for testing)
            
        Note:
            Results are saved incrementally to prevent data loss on interruption.
        """
        tasks_to_save: List[Dict[str, Any]] = []
        processed_count = 0
        max_process = max_tasks or len(tasks)
        
        print(f"[BATCH_START] Processing {min(max_process, len(tasks))} tasks...")
        
        for index, task_item in enumerate(tasks):
            if processed_count >= max_process:
                break
                
            # Reset state for new task
            self.current_answers = []
            self.full_answers = []
            self.final_result = ""
            
            task_description = task_item['problem']
            print(f"[TASK_{index + 1}] Processing: {task_description[:100]}...")
            
            try:
                final_answer, full_answers = self.process_task(task_description)
                
                result_item = {
                    "unique_id": task_item['unique_id'],
                    "final_answer": final_answer,
                    "full_answers": full_answers
                }
                tasks_to_save.append(result_item)
                
                print(f"[TASK_{index + 1}] Completed with result: {final_answer}")
                processed_count += 1
                
                # Save results every 5 tasks to prevent data loss
                if processed_count % 5 == 0:
                    self._save_batch_results(tasks_to_save, output_path)
                    print(f"[CHECKPOINT] Saved {processed_count} completed tasks")
                    tasks_to_save = []
                    
            except Exception as e:
                print(f"[TASK_ERROR] Failed to process task {index + 1}: {str(e)}")
                # Add error result to maintain consistency
                error_result = {
                    "unique_id": task_item['unique_id'],
                    "final_answer": f"[ERROR] {str(e)}",
                    "full_answers": []
                }
                tasks_to_save.append(error_result)
                processed_count += 1
        
        # Save any remaining results
        if tasks_to_save:
            self._save_batch_results(tasks_to_save, output_path)
            
        print(f"[BATCH_COMPLETE] All {processed_count} tasks completed. Results saved to {output_path}")
    
    def _save_batch_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save batch results to file in JSONL format.
        
        Args:
            results (List[Dict[str, Any]]): Results to save
            output_path (str): Path to the output file
            
        Raises:
            IOError: If file writing fails
        """
        try:
            with open(output_path, 'a', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except IOError as e:
            print(f"[SAVE_ERROR] Failed to save results: {str(e)}")
            raise
    
    def stop(self) -> None:
        """
        Gracefully stop the user runner.
        
        This method sets the running flag to False, which causes all
        background threads to terminate gracefully.
        """
        self.running = False
        print("[SHUTDOWN] User runner stopping...")


def load_tasks_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load tasks from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        List[Dict[str, Any]]: List of loaded tasks
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file format is invalid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        print(f"[LOADER] Successfully loaded {len(tasks)} tasks from {file_path}")
        return tasks
    except FileNotFoundError:
        print(f"[LOADER_ERROR] Task file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"[LOADER_ERROR] Invalid JSON format in {file_path}: {str(e)}")
        raise


def main() -> None:
    """
    Main entry point for the user runner application.
    
    Parses command line arguments and starts the user runner with
    task processing capabilities. Supports configurable Chain-of-Thought
    execution numbers for voting mechanisms.
    
    Command line arguments:
        --cot_num: Number of CoT answers needed for voting (default: 2)
    
    Example:
        python user_register.py --cot_num 3
    """
    parser = argparse.ArgumentParser(
        description="User Runner for Symphony Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python user_register.py --cot_num 2    Use 2 CoT executions for voting
  python user_register.py --cot_num 5    Use 5 CoT executions for voting
        """
    )
    parser.add_argument(
        "--cot_num", 
        type=int, 
        default=2, 
        help="Number of Chain-of-Thought answers needed for voting (minimum: 1)"
    )
    args = parser.parse_args()
    
    if args.cot_num < 1:
        print("[ERROR] CoT number must be at least 1")
        return
    
    cot_num = args.cot_num

    # Configuration file path (adjust according to actual project)
    config_path = "./runtime/config_user.yaml"
    
    try:
        # Initialize and start runner
        runner = UserRunner(config_path, cot_num)
        runner.start()
        
        # Load and process test tasks
        test_file = "./test.jsonl"
        output_file = "./symphony_results.jsonl"
        
        try:
            tasks = load_tasks_from_jsonl(test_file)
            runner.process_task_batch(tasks=tasks, output_path=output_file, max_tasks=10)
        except FileNotFoundError:
            print(f"[WARNING] Test file {test_file} not found. Runner started in listening mode.")
            print("The runner will wait for tasks to be assigned programmatically.")
        
        # Keep the runner alive for programmatic task assignment
        try:
            print("[INFO] User runner is active. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Keyboard interrupt received, stopping user runner...")
            runner.stop()
            
    except FileNotFoundError:
        print(f"[ERROR] Configuration file not found: {config_path}")
        print("Please ensure the config file exists before starting the user.")
    except Exception as e:
        print(f"[ERROR] Failed to start user runner: {str(e)}")


if __name__ == "__main__":
    main()
