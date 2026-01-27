"""Command-line interface for running Symphony Network agents.

This module provides a command-line interface for initializing and running
different types of agents in the Symphony Network.
"""

import argparse
from runtime.config import load_config_from_file, get_config
# Note: These imports may not exist yet - they are placeholder references
# from agents.task_requester import TaskRequester
# from agents.compute_provider import ComputeProvider
# from agents.nap_gateway import NAPGateway


def run_agent(agent_type: str, config: dict) -> None:
    """Run an agent of the specified type.
    
    Initializes and starts an agent based on the provided type and configuration.
    Supports multiple GPU deployment for compute-intensive agents.
    
    Args:
        agent_type (str): Type of agent to run. Options: TaskRequester, 
                         ComputeProvider, NAP
        config (dict): Agent configuration dictionary containing network,
                      GPU, and model settings
                      
    Raises:
        ValueError: If agent_type is not supported
    """
    gpu_ids = config.get("gpu_ids", [0])
    
    for i, gpu_id in enumerate(gpu_ids):
        new_config = config.copy()
        new_config["gpu_id"] = gpu_id
        # Assign unique node_id for each agent instance
        new_config["node_id"] = f"{config['node_id']}-{i}"

        try:
            if agent_type == "TaskRequester":
                # agent = TaskRequester(new_config)
                print(f"[CLI] TaskRequester agent type not yet implemented")
                print(f"[CLI] Would run TaskRequester Agent: "
                      f"{new_config['node_id']} on GPU {gpu_id}")
            elif agent_type == "ComputeProvider":
                # agent = ComputeProvider(new_config)
                print(f"[CLI] ComputeProvider agent type not yet implemented")
                print(f"[CLI] Would run ComputeProvider Agent: "
                      f"{new_config['node_id']} on GPU {gpu_id}")
            elif agent_type == "NAP":
                # agent = NAPGateway(new_config["node_id"], api_interface=None)
                print(f"[CLI] NAP Gateway agent type not yet implemented")
                print(f"[CLI] Would run NAP Gateway: "
                      f"{new_config['node_id']} on GPU {gpu_id}")
            else:
                raise ValueError(
                    "Unknown agent type. Choose from: TaskRequester, "
                    "ComputeProvider, NAP"
                )

            print("[CLI] Agent configuration ready. "
                  "Would be live and ready to receive tasks...")
        except Exception as e:
            print(f"[CLI] Error initializing agent: {e}")


def main() -> None:
    """Main CLI function.
    
    Parses command line arguments and starts the specified agent type
    with the provided or default configuration.
    """
    parser = argparse.ArgumentParser(description="Run Symphony agent")
    parser.add_argument(
        "--agent",
        type=str,
        default="TaskRequester",
        help="Agent type: TaskRequester | ComputeProvider | NAP"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config file path"
    )

    args = parser.parse_args()
    config = (
        load_config_from_file(args.config) if args.config else get_config()
    )
    run_agent(args.agent, config)


if __name__ == "__main__":
    main()
