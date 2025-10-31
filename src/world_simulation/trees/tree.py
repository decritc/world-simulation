"""Fruit tree implementation."""

from typing import List
import numpy as np


class FruitTree:
    """A fruit tree that grows and produces fruit."""
    
    def __init__(self, x: float, y: float, z: float, species: str = "apple"):
        """
        Initialize a fruit tree.
        
        Args:
            x: X position in world
            y: Y position in world
            z: Z position in world
            species: Type of fruit tree
        """
        self.x = x
        self.y = y
        self.z = z
        self.species = species
        
        self.age = 0.0
        self.max_age = 100.0  # days
        self.growth_stage = 0.0  # 0.0 to 1.0
        
        self.fruit_count = 0
        self.max_fruit = 10
        self.fruit_maturity = {}  # fruit_id -> maturity (0.0 to 1.0)
        
        self.is_alive = True
        
    def update(self, delta_time: float, world):
        """Update the tree's state."""
        if not self.is_alive:
            return
        
        # Age the tree
        self.age += delta_time
        
        # Grow the tree
        if self.growth_stage < 1.0:
            self.growth_stage = min(1.0, self.age / 30.0)  # Full growth in 30 days
        
        # Produce fruit if mature
        if self.growth_stage >= 0.5 and self.fruit_count < self.max_fruit:
            # Chance to produce new fruit
            if np.random.random() < 0.01 * delta_time:
                fruit_id = len(self.fruit_maturity)
                self.fruit_maturity[fruit_id] = 0.0
                self.fruit_count += 1
        
        # Mature existing fruit
        for fruit_id in list(self.fruit_maturity.keys()):
            self.fruit_maturity[fruit_id] += delta_time * 0.1
            if self.fruit_maturity[fruit_id] >= 1.0:
                # Fruit is fully mature
                pass
        
        # Check if tree dies
        if self.age >= self.max_age:
            self.is_alive = False
    
    def harvest_fruit(self) -> int:
        """Harvest mature fruit from the tree."""
        mature_fruit = [
            fruit_id for fruit_id, maturity in self.fruit_maturity.items()
            if maturity >= 1.0
        ]
        
        for fruit_id in mature_fruit:
            del self.fruit_maturity[fruit_id]
            self.fruit_count -= 1
        
        return len(mature_fruit)
    
    def get_ripe_fruit_count(self) -> int:
        """Get the number of ripe fruit on the tree."""
        return sum(1 for maturity in self.fruit_maturity.values() if maturity >= 1.0)

