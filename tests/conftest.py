"""Test configuration."""

import pytest
import os


@pytest.fixture
def test_config():
    """Test configuration fixture."""
    return {
        "test_mode": True,
        "log_level": "DEBUG"
    }


@pytest.fixture
def mock_agent():
    """Mock agent fixture."""
    from symphony.agents.agent import Agent
    return Agent(agent_id="test_agent", name="Test Agent")
