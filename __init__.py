"""Symphony - A Decentralized Multi-Agent Framework for Edge Devices

Symphony enables intelligent agents to collaborate seamlessly across 
heterogeneous edge devices through beacon-guided task routing and 
Chain-of-Thought (CoT) voting mechanisms.
"""

__version__ = "0.1.0"
__author__ = "Symphony Team"

# Core imports - using absolute imports for package mode
try:
    # Package mode imports
    from symphony.agents.agent import Agent
    from symphony.agents.user import User
    from symphony.core.identity import Identity
    from symphony.core.capability import CapabilityManager as Capability
    from symphony.core.memory import LocalMemory as Memory
    from symphony.protocol.task_contract import Task, TaskResult
    from symphony.protocol.beacon import Beacon
    from symphony.protocol.response import BeaconResponse
except ImportError:
    # Direct execution mode imports
    from agents.agent import Agent
    from agents.user import User
    from core.identity import Identity
    from core.capability import CapabilityManager as Capability
    from core.memory import LocalMemory as Memory
    from protocol.task_contract import Task, TaskResult
    from protocol.beacon import Beacon
    from protocol.response import BeaconResponse

# Import the main execution function
try:
    from .symphony import execute_task, register_agent, get_registered_agents
except ImportError:
    from symphony import execute_task, register_agent, get_registered_agents

__all__ = [
    "Agent",
    "User", 
    "Identity",
    "Capability",
    "Memory",
    "Task",
    "TaskResult",
    "Beacon",
    "BeaconResponse",
    "execute_task",
    "register_agent",
    "get_registered_agents",
]
