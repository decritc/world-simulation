"""Main world class managing terrain, entities, and simulation state."""

import numpy as np
from typing import List, Tuple, Dict

from .generator import WorldGenerator
from ..entities.npc import NPC
from ..trees.tree import FruitTree
from ..houses.house import House
from .vegetation import Vegetation, VegetationGenerator


class World:
    """Main world class managing terrain, entities, and simulation state."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the world.
        
        Args:
            seed: Random seed for world generation
        """
        self.seed = seed
        self.generator = WorldGenerator(seed)
        
        # Chunk cache for terrain generation
        self.chunks: Dict[Tuple[int, int], np.ndarray] = {}
        
        # Time tracking
        self.time = 0.0
        self.day_length = 120.0  # 120 seconds per day (matches __init__.py)
        self.day_time = 0.0
        self.day_number = 0
        
        # Entities
        self.entities: List[NPC] = []
        self.trees: List[FruitTree] = []
        self.houses: List[House] = []
        self.vegetation: List[Vegetation] = []
        self.vegetation_generator = VegetationGenerator(seed)
        
        # Initialize some entities
        self._initialize_world()
    
    def _initialize_world(self):
        """Initialize world with trees and houses."""
        # Generate trees
        for i in range(20):
            x = np.random.uniform(-50, 50)
            z = np.random.uniform(-50, 50)
            y = self.get_height(x, z)  # Place trees on terrain
            tree = FruitTree(x, y, z)
            self.trees.append(tree)
        
        # Generate houses
        for i in range(5):
            x = np.random.uniform(-40, 40)
            z = np.random.uniform(-40, 40)
            y = self.get_height(x, z)  # Place houses on terrain
            house = House(x, y, z)
            self.houses.append(house)
    
    def is_night(self) -> bool:
        """Check if it's currently night time."""
        hour = (self.day_time / self.day_length) * 24.0
        return hour < 6.0 or hour >= 18.0  # Night from 6pm to 6am
    
    def get_time_of_day(self) -> float:
        """Get current time of day (0.0 to 1.0, where 0.5 is noon)."""
        return (self.day_time / self.day_length) % 1.0
    
    def get_light_intensity(self) -> float:
        """
        Get the current light intensity based on time of day.
        
        Returns:
            Light intensity (0.0 to 1.0)
        """
        hour = (self.day_time / self.day_length) * 24.0
        
        if 6.0 <= hour < 8.0:  # Dawn
            return 0.3 + 0.4 * ((hour - 6.0) / 2.0)
        elif 8.0 <= hour < 18.0:  # Day
            return 1.0
        elif 18.0 <= hour < 20.0:  # Dusk
            return 1.0 - 0.4 * ((hour - 18.0) / 2.0)
        else:  # Night
            return 0.3
        
    def get_chunk(self, chunk_x: int, chunk_y: int) -> np.ndarray:
        """
        Get or generate a chunk at the given coordinates.
        
        Args:
            chunk_x: X coordinate of the chunk
            chunk_y: Y coordinate of the chunk
            
        Returns:
            Heightmap for the chunk
        """
        chunk_key = (chunk_x, chunk_y)
        
        if chunk_key not in self.chunks:
            self.chunks[chunk_key] = self.generator.generate_chunk(chunk_x, chunk_y)
        
        return self.chunks[chunk_key]
    
    def get_height(self, x: float, z: float) -> float:
        """
        Get the height at a specific world coordinate using bilinear interpolation
        for smooth terrain transitions between chunks.
        
        Args:
            x: X coordinate
            z: Z coordinate
            
        Returns:
            Height value at the given coordinates (scaled to world units)
        """
        # Get chunk coordinates
        chunk_x = int(np.floor(x / 64.0))
        chunk_z = int(np.floor(z / 64.0))
        
        # Get local coordinates within chunk (0-64)
        local_x_float = x - (chunk_x * 64.0)
        local_z_float = z - (chunk_z * 64.0)
        
        # Clamp to valid range to prevent index errors
        local_x_float = max(0.0, min(63.999, local_x_float))
        local_z_float = max(0.0, min(63.999, local_z_float))
        
        # Get integer coordinates for grid points
        local_x0 = int(np.floor(local_x_float))
        local_z0 = int(np.floor(local_z_float))
        local_x1 = min(local_x0 + 1, 63)
        local_z1 = min(local_z0 + 1, 63)
        
        # Get fractional parts for interpolation
        fx = local_x_float - local_x0
        fz = local_z_float - local_z0
        
        # Get chunk for seamless boundaries
        chunk = self.get_chunk(chunk_x, chunk_z)
        
        # Get heights at four corners
        h00 = chunk[local_z0][local_x0]
        h10 = chunk[local_z0][local_x1]
        h01 = chunk[local_z1][local_x0]
        h11 = chunk[local_z1][local_x1]
        
        # Bilinear interpolation for smooth transitions
        h0 = h00 * (1.0 - fx) + h10 * fx
        h1 = h01 * (1.0 - fx) + h11 * fx
        interpolated_height = h0 * (1.0 - fz) + h1 * fz
        
        # Scale normalized height (0-1) to world height (0 to max_height)
        return interpolated_height * self.generator.max_height
    
    def update(self, delta_time: float):
        """Update the world simulation."""
        self.time += delta_time
        
        # Update day/night cycle
        old_day = self.day_number
        self.day_time += delta_time
        if self.day_time >= self.day_length:
            self.day_time = 0.0
            self.day_number += 1
        
        # Update entities - ensure they're always on terrain
        for entity in self.entities:
            # Always update Y position to match terrain before updating
            entity.y = self.get_height(entity.x, entity.z)
            entity.update(delta_time, self)
            # Update Y position again after movement to ensure it's correct
            entity.y = self.get_height(entity.x, entity.z)
        
        # Remove dead NPCs
        self.entities = [e for e in self.entities if e.is_alive]
        
        # Update trees - ensure they're on terrain
        for tree in self.trees:
            tree.y = self.get_height(tree.x, tree.z)  # Update tree Y position
            tree.update(delta_time, self)
        
        # Update houses (houses don't have update method currently)
        # Future: could add house decay or other mechanics here