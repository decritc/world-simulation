"""House module for shelter."""

from typing import Tuple
import numpy as np
import random


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
        
        # Door location (on front face, centered)
        self.door_x = x  # Door is centered on front face
        self.door_z = z + 1.5  # Door is on front face (+z direction)
        self.door_y = y
        
        # House color theme (randomized for variety)
        self.wall_color = self._generate_house_color()
        self.roof_color = self._generate_roof_color()
    
    def _generate_house_color(self) -> Tuple[float, float, float]:
        """Generate a random house wall color."""
        # Generate warm, varied colors for houses
        colors = [
            (0.8, 0.6, 0.4),  # Beige
            (0.7, 0.5, 0.6),  # Light pink
            (0.6, 0.7, 0.5),  # Light green
            (0.5, 0.6, 0.8),  # Light blue
            (0.7, 0.7, 0.5),  # Light yellow
            (0.6, 0.8, 0.7),  # Mint
        ]
        return random.choice(colors)
    
    def _generate_roof_color(self) -> Tuple[float, float, float]:
        """Generate a random roof color."""
        # Generate darker roof colors
        colors = [
            (0.6, 0.3, 0.2),  # Brown
            (0.4, 0.3, 0.3),  # Dark gray
            (0.5, 0.2, 0.2),  # Dark red
            (0.3, 0.3, 0.4),  # Dark blue-gray
        ]
        return random.choice(colors)
    
    def get_door_position(self) -> Tuple[float, float, float]:
        """
        Get the door entry position.
        
        Returns:
            Tuple of (x, y, z) position for door entry
        """
        return (self.door_x, self.door_y, self.door_z)
    
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
    
    def distance_to_door(self, x: float, z: float) -> float:
        """Calculate distance to the door."""
        dx = self.door_x - x
        dz = self.door_z - z
        return np.sqrt(dx**2 + dz**2)
