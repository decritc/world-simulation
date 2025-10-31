"""Procedural world generation using noise algorithms."""

import numpy as np
from noise import pnoise2


class WorldGenerator:
    """Generates 3D terrain using Perlin noise."""
    
    def __init__(self, seed: int = 42):
        """Initialize the world generator with a seed."""
        self.seed = seed
        np.random.seed(seed)
    
    def generate_heightmap(self, width: int, height: int, scale: float = 100.0) -> np.ndarray:
        """
        Generate a heightmap for the terrain.
        
        Args:
            width: Width of the terrain
            height: Height of the terrain
            scale: Scale factor for noise (higher = smoother terrain)
            
        Returns:
            2D numpy array of height values
        """
        heightmap = np.zeros((height, width))
        
        for y in range(height):
            for x in range(width):
                # Use Perlin noise for natural-looking terrain
                heightmap[y][x] = pnoise2(
                    x / scale,
                    y / scale,
                    octaves=6,
                    persistence=0.5,
                    lacunarity=2.0,
                    repeatx=1024,
                    repeaty=1024,
                    base=self.seed
                )
        
        # Normalize to 0-1 range
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
        
        return heightmap
    
    def generate_chunk(self, chunk_x: int, chunk_y: int, chunk_size: int = 64) -> np.ndarray:
        """
        Generate a chunk of terrain at the given coordinates.
        
        Args:
            chunk_x: X coordinate of the chunk
            chunk_y: Y coordinate of the chunk
            chunk_size: Size of the chunk (chunk_size x chunk_size)
            
        Returns:
            2D numpy array representing the chunk heightmap
        """
        offset_x = chunk_x * chunk_size
        offset_y = chunk_y * chunk_size
        
        return self.generate_heightmap(chunk_size, chunk_size, scale=50.0)

