"""Main world class that manages the simulation."""

from typing import Dict, List, Tuple
import numpy as np

from .generator import WorldGenerator


class World:
    """Main world class managing terrain, entities, and simulation state."""
    
    def __init__(self, seed: int = 42):
        """Initialize the world."""
        self.generator = WorldGenerator(seed)
        self.chunks: Dict[Tuple[int, int], np.ndarray] = {}
        self.entities: List = []
        self.trees: List = []
        self.time = 0.0
        
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
        Get the height at a specific world coordinate.
        
        Args:
            x: X coordinate
            z: Z coordinate
            
        Returns:
            Height value at the given coordinates
        """
        chunk_x = int(x // 64)
        chunk_z = int(z // 64)
        
        chunk = self.get_chunk(chunk_x, chunk_z)
        
        local_x = int(x % 64)
        local_z = int(z % 64)
        
        if local_x >= 64:
            local_x = 63
        if local_z >= 64:
            local_z = 63
        
        return chunk[local_z][local_x]
    
    def update(self, delta_time: float):
        """Update the world simulation."""
        self.time += delta_time
        
        # Update entities
        for entity in self.entities:
            entity.update(delta_time, self)
        
        # Update trees
        for tree in self.trees:
            tree.update(delta_time, self)

