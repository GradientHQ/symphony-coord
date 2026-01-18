"""User agent for initiating and managing distributed tasks.

This module implements the User class which serves as the entry point
for task initiation in the Symphony Network. It handles task distribution,
result collection, and consensus mechanisms.
"""

from typing import Dict, List, Tuple, Any

from protocol.beacon import Beacon
from protocol.task_contract import Task
from infra.ISEP import ISEPClient
from infra.network_adapter import NetworkAdapter


class User:
    """User agent for task initiation and result collection.

    This class represents a user agent that can initiate computational
    tasks, distribute them across the Symphony Network, and collect
    results using consensus mechanisms.

    Attributes:
        user_id (str): Unique identifier for this user agent
        network (NetworkAdapter): Network communication adapter
        isep_client (ISEPClient): ISEP protocol client for communication
        consensus_count (int): Number of executions for consensus building
    """

    def __init__(self, config: Dict[str, Any], consensus_count: int) -> None:
        """Initialize User agent with configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - node_id: Unique user identifier
                - neighbours: List of neighbor node configurations
            consensus_count (int): Number of Chain of Thought executions
                                 for consensus building

        Raises:
            KeyError: If required configuration keys are missing
            ValueError: If consensus_count is invalid
        """
        if consensus_count <= 0:
            raise ValueError("Consensus count must be positive")

        self.user_id = config["node_id"]
        self.network = NetworkAdapter(self.user_id, config)
        self.isep_client = ISEPClient(self.user_id, self.network)
        self.consensus_count = consensus_count

        # Initialize neighbor connections
        self._initialize_neighbors(config["neighbours"])

    def _initialize_neighbors(self, neighbors: List[Tuple[str, str, str]]) -> None:
        """Initialize neighbor nodes in the network.

        Args:
            neighbors: List of tuples containing (node_id, address, port)

        Raises:
            ValueError: If neighbor configuration is invalid
        """
        for neighbor in neighbors:
            try:
                node_id, address, port = neighbor
                self.network.add_neighbor(node_id, address, int(port))
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid neighbor configuration: {neighbor}") from e

    def initiate_task(self, user_input: str) -> None:
        """Initiate a new computational task in the network.

        This method creates a new task from user input, finds suitable
        executor agents through beacon broadcasting, and delegates the
        task for execution.

        Args:
            user_input (str): Original problem description from user

        Raises:
            ValueError: If user_input is empty or invalid
            RuntimeError: If no suitable executors are found
        """
        if not user_input or not user_input.strip():
            raise ValueError("User input cannot be empty")

        # Create initial task object
        task = self._create_initial_task(user_input)

        # Find and select suitable executor agents
        executor_candidates = self._discover_executors(task)
        selected_executors = self._select_executors(
            executor_candidates,
            self.consensus_count
        )

        if not selected_executors:
            raise RuntimeError("No suitable executors found for the task")

        # Delegate task to selected executors
        self._delegate_to_executors(selected_executors, task)

    def _create_initial_task(self, user_input: str) -> Task:
        """Create an initial task object from user input.

        Args:
            user_input (str): Original problem description

        Returns:
            Task: Initial task object ready for execution
        """
        return Task(
            subtask_id=0,
            steps={},
            previous_results=[],
            original_problem=user_input,
            final_result="",
            user_id=self.user_id
        )

    def _discover_executors(self, task: Task) -> List[Tuple[str, float]]:
        """Discover suitable executor agents for the task.

        This method broadcasts a beacon to find agents capable of
        handling the planning phase of the task.

        Args:
            task (Task): Task requiring executor discovery

        Returns:
            List[Tuple[str, float]]: List of (agent_id, capability_score) pairs
        """
        # Create beacon for planning capability discovery
        discovery_beacon = Beacon(
            sender=self.user_id,
            task_id=str(task.subtask_id),
            requirement="Plan",
            ttl=2  # Time-to-live for beacon propagation
        )

        # Broadcast beacon and collect responses
        return self.isep_client.broadcast_and_collect(discovery_beacon)

    def _select_executors(
            self,
            candidates: List[Tuple[str, float]],
            num_executors: int
    ) -> List[Tuple[str, float]]:
        """Select executor agents from discovered candidates.

        This method implements a simple selection strategy by taking
        the first N candidates. More sophisticated selection strategies
        can be implemented by overriding this method.

        Args:
            candidates (List[Tuple[str, float]]): Available executor candidates
            num_executors (int): Number of executors to select

        Returns:
            List[Tuple[str, float]]: Selected executor agents
        """
        # Sort candidates by capability score (descending)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_candidates[:num_executors]

    def _delegate_to_executors(
            self,
            executors: List[Tuple[str, float]],
            task: Task
    ) -> None:
        """Delegate task execution to selected executor agents.

        Args:
            executors (List[Tuple[str, float]]): Selected executor agents
            task (Task): Task to delegate
        """
        for executor_id, capability_score in executors:
            print(f"Delegating task to executor {executor_id} "
                  f"(capability score: {capability_score:.3f})")

            self.isep_client.delegate_task(executor_id, task)

    def collect_results(self, task_id: str, timeout: int = 30) -> List[Any]:
        """Collect execution results from delegated tasks.

        This method waits for and collects results from executor agents
        that were assigned the specified task.

        Args:
            task_id (str): Identifier of the task to collect results for
            timeout (int): Maximum time to wait for results (seconds)

        Returns:
            List[Any]: Collected results from executor agents

        Raises:
            TimeoutError: If results are not received within timeout
        """
        # Implementation would depend on the ISEP client's result collection mechanism
        # This is a placeholder for the actual implementation
        return self.isep_client.collect_task_results(task_id, timeout)

    def build_consensus(
            self,
            results: List[Any],
            consensus_method: str = "majority"
    ) -> Any:
        """Build consensus from multiple execution results.

        This method implements consensus building from multiple
        execution results to improve reliability and accuracy.

        Args:
            results (List[Any]): Results from multiple executions
            consensus_method (str): Method for consensus building
                                  ("majority", "average", "best")

        Returns:
            Any: Consensus result

        Raises:
            ValueError: If consensus_method is not supported
            RuntimeError: If consensus cannot be built
        """
        if not results:
            raise RuntimeError("Cannot build consensus from empty results")

        if consensus_method == "majority":
            return self._majority_consensus(results)
        elif consensus_method == "average":
            return self._average_consensus(results)
        elif consensus_method == "best":
            return self._best_result_consensus(results)
        else:
            raise ValueError(f"Unsupported consensus method: {consensus_method}")

    def _majority_consensus(self, results: List[Any]) -> Any:
        """Build consensus using majority voting.

        Args:
            results (List[Any]): Results to build consensus from

        Returns:
            Any: Most frequent result
        """
        from collections import Counter
        result_counts = Counter(str(result) for result in results)
        most_common = result_counts.most_common(1)[0][0]

        # Return the original result object that matches the most common string
        for result in results:
            if str(result) == most_common:
                return result

        return results[0]  # Fallback

    def _average_consensus(self, results: List[Any]) -> Any:
        """Build consensus using averaging (for numeric results).

        Args:
            results (List[Any]): Results to build consensus from

        Returns:
            Any: Average result
        """
        # This is a simplified implementation
        # Real implementation would need proper numeric result handling
        return results[0] if results else None

    def _best_result_consensus(self, results: List[Any]) -> Any:
        """Build consensus by selecting the best result.

        Args:
            results (List[Any]): Results to build consensus from

        Returns:
            Any: Best result based on quality metrics
        """
        # This would require implementation of result quality assessment
        # For now, return the first result as a placeholder
        return results[0] if results else None

