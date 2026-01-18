"""Configuration management for Symphony Network nodes.

This module provides configuration loading and management functionality
for Symphony Network agents and users.
"""

import os
import yaml
from typing import Dict, Any, List, Tuple


# Default configuration template
DEFAULT_CONFIG: Dict[str, Any] = {
    # Basic configuration
    "debug": False,
    "role": "task_requester",
    "node_id": "default-agent",

    # Model loading configuration
    "base_model": "default_model",
    "sys_prompt": "",
    "load_quantized": False,
    "use_lora": False,

    # Capability registration (automatically matched with tools)
    "capabilities": [],

    # Network parameters
    "p2p": {
        "port": 7788,
        "enable_dht": False,
        "beacon_ttl": 2,
        "heartbeat_interval": 15
    },

    # LoRA storage configuration
    "lora": {
        "save_dir": "checkpoints/",
        "patch_interval": 300
    },

    # Logging configuration
    "log_level": "INFO",

    # Network adapter configuration
    "network": {
        "host": "213.192.2.94",
        "port": 8000
    },

    # GPU configuration
    "gpu_ids": [0],

    # Neighbour nodes configuration
    "neighbours": [["agent-002", "127.0.0.1", 8000]]
}


def load_config_from_file(filepath: str) -> Dict[str, Any]:
    """Load configuration from YAML file with fallback to default config.
    
    Attempts to load configuration from the specified YAML file. If the file
    doesn't exist or contains invalid YAML, falls back to default configuration.
    The loaded configuration is merged with the default configuration to ensure
    all required fields are present.
    
    Args:
        filepath (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Merged configuration dictionary containing both
                       default and file-based configuration values
                       
    Note:
        File configuration values override default values for matching keys.
        Missing keys in the file will use default values.
    """
    if not os.path.exists(filepath):
        print(f"[Config] Config file not found: {filepath}, "
              f"using default config.")
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"[Config] Error loading YAML file: {e}, "
              f"using default config.")
        return DEFAULT_CONFIG.copy()
    except FileNotFoundError:
        print(f"[Config] Config file not found: {filepath}, "
              f"using default config.")
        return DEFAULT_CONFIG.copy()
    
    # Merge file config with default config
    merged_config = DEFAULT_CONFIG.copy()
    if file_config:
        merged_config.update(file_config)
    
    return merged_config


def get_config() -> Dict[str, Any]:
    """Get the default configuration.
    
    Returns a copy of the default configuration dictionary.
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return DEFAULT_CONFIG.copy()


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters.
    
    Checks if the provided configuration contains all required fields
    and has valid values for critical parameters.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_fields = ["node_id", "role", "network", "p2p"]
    
    for field in required_fields:
        if field not in config:
            print(f"[Config] Missing required field: {field}")
            return False
    
    # Validate network configuration
    network_config = config.get("network", {})
    if not isinstance(network_config.get("port"), int):
        print("[Config] Invalid network port configuration")
        return False
    
    return True
