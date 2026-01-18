#!/usr/bin/env python3
"""Symphony High-Level API Usage Examples

This script demonstrates how to use Symphony's high-level API
for multi-agent task execution as shown in the README.
"""

import sys
import os

# Add parent directory to path for local development
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from agents.agent import Agent
from protocol.task_contract import Task
from symphony import execute_task, get_registered_agents


def example_1_basic_usage():
    """Example 1: Basic multi-agent task execution"""
    print("\n=== Example 1: Basic Multi-Agent Task Execution ===")
    
    # Initialize specialized agents
    math_agent = Agent(
        node_id="math_specialist",
        capabilities=["mathematical-reasoning", "calculus", "statistics"]
    )
    
    code_agent = Agent(
        node_id="code_specialist", 
        capabilities=["code-generation", "debugging", "optimization"]
    )
    
    research_agent = Agent(
        node_id="research_specialist",
        capabilities=["web-search", "data-analysis", "summarization"]
    )
    
    print(f"Registered {len(get_registered_agents())} agents")
    
    # Create a complex task requiring multiple specialties
    complex_task = Task(
        description="Build a machine learning model to predict stock prices using financial data",
        requirements=["data-collection", "mathematical-modeling", "code-implementation"],
        context={"domain": "finance", "complexity": "high"}
    )
    
    print(f"Task: {complex_task.description}")
    print(f"Requirements: {complex_task.requirements}")
    
    # Execute with automatic coordination
    result = execute_task(complex_task)
    
    print("\nExecution Result:")
    print("-" * 50)
    print(result)
    print("-" * 50)


def example_2_research_task():
    """Example 2: Research-focused task"""
    print("\n=== Example 2: Research-Focused Task ===")
    
    # Create research specialists
    web_agent = Agent(
        node_id="web_researcher",
        capabilities=["web-search", "information-extraction", "fact-checking"]
    )
    
    analysis_agent = Agent(
        node_id="data_analyst",
        capabilities=["data-analysis", "statistical-analysis", "visualization"]
    )
    
    writing_agent = Agent(
        node_id="technical_writer",
        capabilities=["technical-writing", "summarization", "documentation"]
    )
    
    # Create research task
    research_task = Task(
        description="Analyze the current trends in renewable energy adoption and create a comprehensive report",
        requirements=["information-gathering", "data-analysis", "report-writing"],
        context={"domain": "energy", "complexity": "medium", "output": "report"}
    )
    
    print(f"Research Task: {research_task.description}")
    
    # Execute research
    result = execute_task(research_task, cot_count=2)  # Use fewer CoT executions
    
    print("\nResearch Result:")
    print("-" * 50)
    print(result)
    print("-" * 50)


def example_3_software_development():
    """Example 3: Software development task"""
    print("\n=== Example 3: Software Development Task ===")
    
    # Create development team
    architect = Agent(
        node_id="software_architect",
        capabilities=["system-design", "architecture", "planning"]
    )
    
    backend_dev = Agent(
        node_id="backend_developer",
        capabilities=["backend-development", "api-design", "database-design"]
    )
    
    frontend_dev = Agent(
        node_id="frontend_developer",
        capabilities=["frontend-development", "ui-design", "user-experience"]
    )
    
    tester = Agent(
        node_id="qa_tester",
        capabilities=["testing", "quality-assurance", "debugging"]
    )
    
    # Create development task
    dev_task = Task(
        description="Design and implement a task management web application with user authentication",
        requirements=["system-design", "backend-development", "frontend-development", "testing"],
        context={"domain": "web-development", "complexity": "high", "stack": "full-stack"}
    )
    
    print(f"Development Task: {dev_task.description}")
    
    # Execute development
    result = execute_task(dev_task, cot_count=2)
    
    print("\nDevelopment Result:")
    print("-" * 50)
    print(result)
    print("-" * 50)


def main():
    """Run all examples"""
    print("🎼 Symphony High-Level API Examples")
    print("=" * 60)
    
    try:
        example_1_basic_usage()
        example_2_research_task()
        example_3_software_development()
        
        print(f"\n✅ All examples completed successfully!")
        print(f"Total agents registered: {len(get_registered_agents())}")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
