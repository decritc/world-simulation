"""House module for shelter."""

from typing import Tuple
import numpy as np


class House:
    """A house that provides shelter for NPCs."""
    
    def __init__(self, x: float, y: float, z: float, capacity: int = 2):
        """
        Initialize a house.
        
        Args:
            x: X position in world
            y: Y position in world
            z: Z position in world
            capacity: Maximum number of adult NPCs that can shelter here (default: 2)
        """
        self.x = x
        self.y = y
        self.z = z
        self.capacity = capacity  # Now defaults to 2 adults
        self.current_occupants = set()  # Set of NPC IDs or references
        self.is_built = True
        
    def can_shelter_adult(self) -> bool:
        """Check if the house has space for an adult NPC (max 2 adults)."""
        return len(self.current_occupants) < self.capacity
    
    def can_shelter(self) -> bool:
        """Check if the house has space for more NPCs."""
        return self.can_shelter_adult()  # Alias for backwards compatibility
    
    def add_occupant(self, npc_id) -> bool:
        """
        Add an NPC to the house.
        
        Args:
            npc_id: Identifier for the NPC
            
        Returns:
            True if successfully added, False if full
        """
        if self.can_shelter():
            self.current_occupants.add(npc_id)
            return True
        return False
    
    def remove_occupant(self, npc_id):
        """Remove an NPC from the house."""
        self.current_occupants.discard(npc_id)
    
    def get_position(self) -> Tuple[float, float, float]:
        """Get the house position."""
        return (self.x, self.y, self.z)
    
    def distance_to(self, x: float, z: float) -> float:
        """Calculate distance to a point."""
        dx = self.x - x
        dz = self.z - z
        return np.sqrt(dx**2 + dz**2)
