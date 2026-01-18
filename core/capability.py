"""Core capability management for matching node skills with task requirements."""

from typing import List
import difflib

__all__ = ['CapabilityManager']


class CapabilityManager:
    """Manages and matches node capabilities against task requirements.
    
    This class provides functionality to manage a node's capabilities and
    match them against task requirements using similarity scoring.
    
    Attributes:
        capabilities (List[str]): List of normalized capability tags.
    """
    
    def __init__(self, capability_tags: List[str]):
        """Initialize the capability manager with a list of tags.
        
        Args:
            capability_tags (List[str]): Initial list of capability tags.
                Tags will be normalized to lowercase and deduplicated.
        """
        self.capabilities = list(set([tag.lower() for tag in capability_tags]))
    
    def add_capability(self, tag: str) -> None:
        """Add a new capability tag if it doesn't already exist.
        
        Args:
            tag (str): The capability tag to add.
        """
        tag = tag.lower()
        if tag not in self.capabilities:
            self.capabilities.append(tag)
    
    def remove_capability(self, tag: str) -> None:
        """Remove a capability tag if it exists.
        
        Args:
            tag (str): The capability tag to remove.
        """
        tag = tag.lower()
        if tag in self.capabilities:
            self.capabilities.remove(tag)
    
    def list_capabilities(self) -> List[str]:
        """Get the list of all current capabilities.
        
        Returns:
            List[str]: List of normalized capability tags.
        """
        return self.capabilities
    
    def match(self, requirement: str, threshold: float = 0.5) -> float:
        """Calculate the best similarity score for a requirement.
        
        Compares the requirement against all capabilities and returns
        the highest similarity score found.
        
        Args:
            requirement (str): The requirement to match against capabilities.
            threshold (float, optional): Minimum threshold for matching.
                Defaults to 0.5.
        
        Returns:
            float: Similarity score between 0.0 and 1.0, where 1.0 is
                an exact match.
        """
        requirement = requirement.lower()
        best_score = 0.0
        for tag in self.capabilities:
            ratio = difflib.SequenceMatcher(None, tag, requirement).ratio()
            best_score = max(best_score, ratio)
        return round(best_score, 3)
    
    def match_and_filter(self, requirement: str, threshold: float = 0.5) -> bool:
        """Check if any capability meets the requirement threshold.
        
        Args:
            requirement (str): The requirement to match against capabilities.
            threshold (float, optional): Minimum threshold for acceptance.
                Defaults to 0.5.
        
        Returns:
            bool: True if the best match meets or exceeds the threshold.
        """
        return self.match(requirement, threshold=threshold) >= threshold


if __name__ == "__main__":
    # Example usage
    manager = CapabilityManager(["image-to-text", "style-transfer", "translation"])
    print("Declared:", manager.list_capabilities())
    print("Match 'image translation':", manager.match("image translation"))
    print("Should accept?", manager.match_and_filter(
        "image translation", threshold=0.6))
