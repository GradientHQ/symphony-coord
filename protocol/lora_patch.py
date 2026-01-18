# protocol/lora_patch.py
"""LoRA patch sharing implementation for distributed model updates.

This module provides the LoRAPatch class for sharing and managing
LoRA (Low-Rank Adaptation) patches across the symphony network,
enabling distributed fine-tuning and model updates.
"""

import os
import torch
import time
import uuid
from typing import Dict, List, Any, Optional


class LoRAPatch:
    """LoRA patch container for distributed model sharing.
    
    Manages LoRA patches that can be shared across nodes in the symphony
    network to enable collaborative model fine-tuning and updates.
    
    Attributes:
        patch_id (str): Unique identifier for this patch
        source_id (str): ID of the node that created this patch
        patch_path (str): Local storage path for the patch file
        layer_names (List[str]): Names of the LoRA layers being updated
        timestamp (int): Unix timestamp when patch was created
        is_sparse (bool): Whether the patch uses sparse representation
    """
    
    def __init__(
        self, 
        source_id: str, 
        patch_path: str, 
        layer_names: List[str], 
        is_sparse: bool = False
    ) -> None:
        """Initialize LoRAPatch instance.
        
        Args:
            source_id: ID of the node sharing this patch
            patch_path: Local storage path for the patch
            layer_names: List of LoRA layer names being updated
            is_sparse: Whether to use sparse representation for the patch
        """
        self.patch_id = str(uuid.uuid4())
        self.source_id = source_id
        self.patch_path = patch_path
        self.layer_names = layer_names
        self.timestamp = int(time.time())
        self.is_sparse = is_sparse

    def to_dict(self) -> Dict[str, Any]:
        """Convert patch to dictionary format for serialization.
        
        Returns:
            Dictionary representation of the LoRA patch
        """
        return {
            "patch_id": self.patch_id,
            "source_id": self.source_id,
            "patch_path": self.patch_path,
            "layer_names": self.layer_names,
            "timestamp": self.timestamp,
            "is_sparse": self.is_sparse
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'LoRAPatch':
        """Create LoRAPatch instance from dictionary data.
        
        Args:
            data: Dictionary containing patch data
            
        Returns:
            LoRAPatch instance created from the dictionary
        """
        patch = LoRAPatch(
            source_id=data.get("source_id", "unknown"),
            patch_path=data.get("patch_path", ""),
            layer_names=data.get("layer_names", []),
            is_sparse=data.get("is_sparse", False)
        )
        patch.patch_id = data.get("patch_id", str(uuid.uuid4()))
        patch.timestamp = data.get("timestamp", int(time.time()))
        return patch

    def save_patch(
        self, 
        state_dict: Dict[str, torch.Tensor], 
        save_dir: str = "shared_patches"
    ) -> str:
        """Save the LoRA patch state dictionary to disk.
        
        Args:
            state_dict: PyTorch state dictionary containing LoRA weights
            save_dir: Directory to save the patch file
            
        Returns:
            Full path to the saved patch file
            
        Raises:
            OSError: If unable to create save directory or write file
        """
        os.makedirs(save_dir, exist_ok=True)
        filename = f"patch_{self.source_id[:6]}_{self.patch_id[:6]}.pt"
        full_path = os.path.join(save_dir, filename)
        
        try:
            torch.save(state_dict, full_path)
            self.patch_path = full_path
            print(f"[LoRA Patch] Saved to {full_path}")
            return full_path
        except Exception as e:
            raise OSError(f"Failed to save LoRA patch: {e}")

    def load_patch(self) -> Dict[str, torch.Tensor]:
        """Load the LoRA patch from disk.
        
        Returns:
            PyTorch state dictionary containing LoRA weights
            
        Raises:
            FileNotFoundError: If patch file does not exist
            RuntimeError: If unable to load the patch file
        """
        if not os.path.exists(self.patch_path):
            raise FileNotFoundError(f"Patch file not found: {self.patch_path}")
        
        try:
            return torch.load(self.patch_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA patch: {e}")

    def __repr__(self) -> str:
        """Return string representation of the LoRA patch."""
        return (
            f"<LoRAPatch {self.patch_id[:6]} from {self.source_id} | "
            f"Layers: {self.layer_names}>"
        )
