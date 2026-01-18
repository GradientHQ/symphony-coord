# infra/network_adapter.py
"""
Network Adapter for Symphony Node Communication

Provides TCP-based network communication capabilities for Symphony distributed computing nodes.
Handles message serialization, connection management, and protocol implementation.

Symphony 2.0 changes:
- Support sending dict/list/native json types directly (no forced .to_dict()).
- Keep backward compatibility with objects that implement to_dict() and dataclasses.
"""

import socket
import json
import threading
from typing import Dict, Callable, List, Optional, Tuple, Any
from dataclasses import asdict, is_dataclass  # ✅ Symphony 2.0: add is_dataclass
import time


class NetworkAdapter:
    """Network adapter handling TCP communication between Symphony nodes."""

    def __init__(self, node_id: str, config: dict):
        self.node_id = node_id
        self.host = config["network"]["host"]
        self.port = config["network"]["port"]
        self.neighbors: Dict[str, Tuple[str, int]] = {}  # Node ID → (host, port)
        self.handlers: Dict[str, Callable] = {}
        self.server_socket = None
        self.receive_thread = None
        self._start_server()

    def register_handler(self, msg_type: str, handler: Callable):
        """Register message handler for specific message type."""
        self.handlers[msg_type] = handler

    def add_neighbor(self, node_id: str, host: str, port: int):
        """Add a neighbor node to the network topology."""
        self.neighbors[node_id] = (host, port)

    # -------------------------------------------------------------------------
    # ✅ Symphony 2.0: robust serialization (dict/list/native/object/to_dict/dataclass)
    # -------------------------------------------------------------------------
    def _serialize_data(self, data: Any):
        """
        Serialize data object for transmission.

        Supports:
          - dict / list / str / int / float / bool / None
          - objects with to_dict()
          - dataclass instances
          - generic python objects via __dict__

        Note: This method ensures Symphony 2.0 payload fields (e.g., dynamic_state)
              are not dropped by forced conversions.
        """
        if data is None:
            return None

        # Native JSON types (already serializable)
        if isinstance(data, (dict, list, str, int, float, bool)):
            return data

        # Explicit serializer
        if hasattr(data, "to_dict") and callable(getattr(data, "to_dict")):
            return data.to_dict()

        # Dataclass
        if is_dataclass(data):
            return asdict(data)

        # Generic Python object
        if hasattr(data, "__dict__"):
            return dict(data.__dict__)

        # Fallback best-effort
        return str(data)

    def send(self, target_id: str, msg_type: str, data) -> bool:
        """Send message to target node via TCP.

        Symphony 2.0 change:
        - Do NOT assume `data` has `.to_dict()`. Use `_serialize_data()`.
        """
        if target_id not in self.neighbors:
            print(f"Error: Unknown node ID {target_id}")
            return False

        host, port = self.neighbors[target_id]

        try:
            # Build message packet
            message = {
                "sender_id": self.node_id,
                "target_id": target_id,
                "msg_type": msg_type,
                "data": self._serialize_data(data),  # ✅ changed here
                "timestamp": time.time()
            }

            # Create TCP connection and send message
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((host, port))

                msg_json = json.dumps(message)
                msg_bytes = msg_json.encode('utf-8')

                s.sendall(len(msg_bytes).to_bytes(4, 'big'))
                s.sendall(msg_bytes)

                response = self._receive_response(s)
                return response.get("status") == "success"

        except (socket.error, json.JSONDecodeError) as e:
            print(f"Failed to send message to {target_id}: {e}")
            return False

    def broadcast(self, msg_type: str, data, exclude: List[str] = None):
        """Broadcast message to all neighbors with optional exclusions."""
        exclude = exclude or []
        for neighbor_id in list(self.neighbors.keys()):
            if neighbor_id not in exclude:
                self.send(neighbor_id, msg_type, data)

    def _start_server(self):
        """Start server thread to receive incoming messages."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
        print(f"Node {self.node_id} network adapter started, listening on {self.host}:{self.port}")

    def _receive_loop(self):
        """Message receiving loop running in background thread."""
        while True:
            try:
                conn, addr = self.server_socket.accept()
                with conn:
                    len_bytes = conn.recv(4)
                    if not len_bytes:
                        continue

                    msg_length = int.from_bytes(len_bytes, 'big')

                    msg_bytes = b""
                    while len(msg_bytes) < msg_length:
                        chunk = conn.recv(min(4096, msg_length - len(msg_bytes)))
                        if not chunk:
                            break
                        msg_bytes += chunk

                    if len(msg_bytes) != msg_length:
                        continue

                    msg_json = msg_bytes.decode('utf-8')
                    message = json.loads(msg_json)

                    # Send acknowledgment response
                    ack_response = {"status": "success", "message": "received"}
                    ack_json = json.dumps(ack_response)
                    ack_bytes = ack_json.encode('utf-8')
                    conn.sendall(len(ack_bytes).to_bytes(4, 'big'))
                    conn.sendall(ack_bytes)

                    self._handle_message(message)

            except Exception as e:
                print(f"Error receiving message: {e}")

    def _handle_message(self, message):
        """Handle received message by dispatching to appropriate handler."""
        msg_type = message.get("msg_type")
        sender_id = message.get("sender_id")
        data = message.get("data")

        if not msg_type or not sender_id or data is None:
            print(f"Invalid message format: {message}")
            return

        if msg_type in self.handlers:
            # Current implementation: data is already JSON-deserialized
            deserialized_data = self._deserialize_data(data)
            self.handlers[msg_type](sender_id, deserialized_data)
        else:
            print(f"Unknown message type: {msg_type}")

    def _receive_response(self, socket_obj) -> Dict:
        """Receive and parse response message from socket."""
        try:
            len_bytes = socket_obj.recv(4)
            if not len_bytes:
                return {"status": "error", "message": "No response"}

            resp_length = int.from_bytes(len_bytes, 'big')

            resp_bytes = b""
            while len(resp_bytes) < resp_length:
                chunk = socket_obj.recv(min(4096, resp_length - len(resp_bytes)))
                if not chunk:
                    break
                resp_bytes += chunk

            if len(resp_bytes) != resp_length:
                return {"status": "error", "message": "Incomplete response"}

            resp_json = resp_bytes.decode('utf-8')
            return json.loads(resp_json)

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _deserialize_data(self, data):
        """Deserialize data object after reception.

        Note: simplified; returns dict/list directly. Higher-level modules
              (ISEP/Agent) can call from_dict() if needed.
        """
        return data

    def close(self):
        """Close network adapter and clean up resources."""
        if self.server_socket:
            self.server_socket.close()
        print(f"Node {self.node_id} network adapter closed")
