"""Test agent functionality."""

import pytest
import sys
import os

# Add the parent directory to the path so we can import symphony
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent import Agent


class TestAgent:
    """Test Agent class."""
    
    def test_agent_creation(self):
        """Test agent creation."""
        agent = Agent(agent_id="test_001", name="Test Agent")
        assert agent.agent_id == "test_001"
        assert agent.name == "Test Agent"
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = Agent(agent_id="test_002", name="Test Agent 2")
        assert hasattr(agent, 'identity')
        assert hasattr(agent, 'capabilities')
        assert hasattr(agent, 'memory')
