#!/usr/bin/env python3
"""
Agent Registration and Execution Module

This module provides the AgentRunner class for managing Symphony Network agents.
It handles beacon listening, task execution, and coordination between different
components of the distributed system.

The AgentRunner manages the lifecycle of an agent including:
- Beacon message processing for peer discovery
- Task execution and delegation
- Network communication management
- Resource cleanup

Author: Symphony Network Team
License: MIT
"""

import time
import threading
from typing import Optional
import argparse

from runtime.config import load_config_from_file
from agents.agent import Agent


class AgentRunner:
    """
    Agent runner for handling beacon listening and task execution.
    
    This class manages the full lifecycle of a Symphony Network agent,
    including network communication, task processing, and resource management.
    It uses threading to handle concurrent operations like beacon listening
    and task execution.
    
    Attributes:
        config (dict): Configuration loaded from the config file
        agent (Agent): The Symphony agent instance
        running (bool): Flag indicating if the runner is active
        beacon_enabled (threading.Event): Controls beacon listening state
        first_task_received (bool): Tracks if initial task has been received
    """
    
    def __init__(self, config_path: str) -> None:
        """
        Initialize agent runner with configuration.
        
        Args:
            config_path (str): Path to the YAML configuration file
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration is invalid
        """
        # Load configuration and initialize Agent
        self.config = load_config_from_file(config_path)
        self.agent = Agent(self.config)

        self.running = False  # Running state flag

        self.beacon_enabled = threading.Event()  # Control beacon listening
        self.beacon_enabled.set()  # Initially allow listening

        self.first_task_received = False
    
    def start(self) -> None:
        """
        Start the agent runner with listening and processing loops.
        
        This method initiates the main execution flow by starting:
        1. Beacon listening thread for peer discovery
        2. Task listening thread for work coordination
        3. Main thread waiting for stop commands
        
        The method blocks until a stop command is received.
        """
        self.running = True
        print(f"[STARTUP] Agent {self.agent.agent_id} started, listening for Beacon and Task messages...")

        # Start beacon listening thread
        beacon_thread = threading.Thread(target=self._listen_beacon, daemon=True)
        beacon_thread.start()

        # Start task listening thread
        task_thread = threading.Thread(target=self._listen_tasks, daemon=True)
        task_thread.start()

        # Main thread waits for user input to stop
        self._wait_for_stop()
    
    def _listen_beacon(self) -> None:
        """
        Continuously listen for beacon messages and process them.
        
        This method runs in a separate thread and handles peer discovery
        through beacon messages. It only processes beacons when beacon
        listening is enabled (controlled by beacon_enabled event).
        
        The method catches and logs exceptions to prevent thread termination
        during temporary network issues.
        """
        while self.running:
            if self.beacon_enabled.is_set():  # Only listen when allowed
                try:
                    # Call ISEPClient's receive_beacon function to receive Beacon
                    sender_id, msg_type, beacon = self.agent.isep_client.receive_beacon(timeout=1)
                    if msg_type == "beacon":
                        print(f"[BEACON] Received beacon request from agent: {sender_id}")
                        # Call original class's handle_beacon method to process
                        self.agent.handle_beacon(sender_id, beacon)
                except Exception as e:
                    if self.running:  # Only print errors when running
                        print(f"[BEACON_ERROR] Failed to process beacon: {str(e)}")
            time.sleep(0.5)  # Avoid CPU spinning
    
    def _listen_tasks(self) -> None:
        """
        Continuously listen for task messages and coordinate execution.
        
        This method handles the core task processing logic:
        1. Receives tasks from the network
        2. Manages task execution state
        3. Controls beacon listening during task processing
        4. Handles task delegation and result submission
        
        The method implements a state machine for task processing that
        ensures proper coordination between different system components.
        """
        while self.running:
            try:
                sender_id, msg_type, task = self.agent.isep_client.receive_task(timeout=1)
                if msg_type == "task":
                    print(f"[TASK] Received subtask from agent: {sender_id}")

                    if not self.first_task_received:
                        # First time receiving task, start execution
                        self.first_task_received = True

                        print("[CONTROL] Pausing beacon listening during task execution")
                        self.beacon_enabled.clear()

                        new_task = self.agent.execute_task(task)

                        if new_task.subtask_id == len(new_task.steps) + 1:  # Final result
                            final_result = new_task.final_result
                            previous_results = new_task.previous_results
                            print(f"[RESULT] Task completed: {final_result}")
                            print(f"[RESULT] Submitting result to user: {new_task.user_id}")
                            self.agent.isep_client.submit_result(new_task.user_id, final_result, previous_results)

                            # Current task ended, restore normal operation
                            print("[CONTROL] Resuming beacon listening")
                            self.beacon_enabled.set()
                            self.first_task_received = False
                        else:
                            print("[CONTROL] Resuming beacon listening")
                            self.beacon_enabled.set()
                            self.first_task_received = False
                            self.agent.assign_task(new_task)  # Intermediate steps

                    else:
                        # Task already executing, forward the new received task
                        print(f"[RELAY] Current task is executing, forwarding new received task")
                        self.agent.assign_task(task)

            except Exception as e:
                if self.running:
                    print(f"[TASK_ERROR] Failed to process task: {str(e)}")
                    print("[CONTROL] Resuming beacon listening after error")
                    self.beacon_enabled.set()
                    self.first_task_received = False
            time.sleep(0.5)
    
    def _wait_for_stop(self) -> None:
        """
        Wait for user input to stop the agent runner.
        
        This method blocks the main thread and waits for user input.
        When 'stop' is entered, it initiates a graceful shutdown:
        1. Sets running flag to False
        2. Closes network connections
        3. Cleans up resources
        
        The method ensures all threads are properly terminated and
        resources are released before the program exits.
        """
        while self.running:
            try:
                user_input = input()
                if user_input.strip().lower() == "stop":
                    self.running = False
                    print("\n[SHUTDOWN] Received stop command, initiating graceful shutdown...")
                    # Clean up resources (such as closing network connections)
                    self.agent.network.close()
                    print("[SHUTDOWN] Agent service stopped successfully")
                    break
            except KeyboardInterrupt:
                self.running = False
                print("\n[SHUTDOWN] Keyboard interrupt received, stopping agent...")
                self.agent.network.close()
                print("[SHUTDOWN] Agent service stopped")
                break
            except EOFError:
                # Handle case where input is not available (e.g., in some testing environments)
                print("[SHUTDOWN] Input not available, agent will run indefinitely. Use Ctrl+C to stop.")
                try:
                    while self.running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.running = False
                    print("\n[SHUTDOWN] Keyboard interrupt received, stopping agent...")
                    self.agent.network.close()
                    print("[SHUTDOWN] Agent service stopped")
                break


def main() -> None:
    """
    Main entry point for the agent runner application.
    
    Parses command line arguments and starts the agent runner with
    the specified configuration. Supports multiple agent configurations
    through the --config_num parameter.
    
    Command line arguments:
        --config_num: Integer specifying which agent config to use (default: 1)
    
    Example:
        python agent_register.py --config_num 2
    """
    parser = argparse.ArgumentParser(
        description="Agent Runner for Symphony Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent_register.py --config_num 1    Start agent with config_agent1.yaml
  python agent_register.py --config_num 3    Start agent with config_agent3.yaml
        """
    )
    parser.add_argument(
        "--config_num", 
        type=int, 
        default=1, 
        help="Configuration file number (corresponds to config_agent{N}.yaml)"
    )
    args = parser.parse_args()
    
    config_num = args.config_num

    # Configuration file path (adjust according to actual project)
    config_path = f"./runtime/config_agent{config_num}.yaml"
    
    try:
        # Start runner
        runner = AgentRunner(config_path)
        runner.start()
    except FileNotFoundError:
        print(f"[ERROR] Configuration file not found: {config_path}")
        print("Please ensure the config file exists before starting the agent.")
    except Exception as e:
        print(f"[ERROR] Failed to start agent: {str(e)}")


if __name__ == "__main__":
    main()