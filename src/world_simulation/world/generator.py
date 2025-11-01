"""Procedural world generation using noise algorithms."""

import numpy as np
from noise import pnoise2
from typing import List, Dict, Any


class WorldGenerator:
    """Generates 3D terrain using Perlin noise with hills and mountains."""
    
    def __init__(self, seed: int = 42):
        """Initialize the world generator with a seed."""
        self.seed = seed
        np.random.seed(seed)
        
        # Terrain generation parameters - optimized for smooth terrain
        self.max_height = 15.0  # Maximum height for mountains (reduced for smoother terrain)
        self.hill_height = 6.0  # Maximum height for hills
        self.terrain_scale = 150.0  # Larger scale for smoother terrain features
        
    def generate_heightmap(self, width: int, height: int, scale: float = 100.0) -> np.ndarray:
        """
        Generate a heightmap for the terrain with hills and mountains.
        
        Uses multiple octaves of Perlin noise to create:
        - Large-scale mountain ranges (low frequency)
        - Medium-scale hills (medium frequency)
        - Small-scale details (high frequency)
        
        Args:
            width: Width of the terrain
            height: Height of the terrain
            scale: Scale factor for noise (higher = smoother terrain)
            
        Returns:
            2D numpy array of height values (0.0 = sea level, 1.0 = mountain peak)
        """
        heightmap = np.zeros((height, width))
        
        for y in range(height):
            for x in range(width):
                # Large-scale mountain ranges (low frequency, high amplitude)
                mountains = pnoise2(
                    x / (scale * 4.0),
                    y / (scale * 4.0),
                    octaves=4,
                    persistence=0.5,  # Reduced for smoother transitions
                    lacunarity=2.0,
                    repeatx=1024,
                    repeaty=1024,
                    base=self.seed
                )
                
                # Medium-scale hills (medium frequency, medium amplitude)
                hills = pnoise2(
                    x / scale,
                    y / scale,
                    octaves=5,  # Reduced octaves for smoother hills
                    persistence=0.4,  # Reduced for smoother transitions
                    lacunarity=2.0,
                    repeatx=1024,
                    repeaty=1024,
                    base=self.seed + 1000
                )
                
                # Small-scale details (high frequency, low amplitude)
                details = pnoise2(
                    x / (scale * 0.5),
                    y / (scale * 0.5),
                    octaves=4,  # Reduced octaves for smoother details
                    persistence=0.3,  # Reduced for smoother transitions
                    lacunarity=2.0,
                    repeatx=1024,
                    repeaty=1024,
                    base=self.seed + 2000
                )
                
                # Combine noise layers with balanced weights for smoother terrain
                combined = (
                    mountains * 0.4 +  # Mountains (reduced weight)
                    hills * 0.4 +       # Hills (increased weight)
                    details * 0.2       # Details (reduced weight)
                )
                
                heightmap[y][x] = combined
        
        # Normalize to 0-1 range
        min_val = heightmap.min()
        max_val = heightmap.max()
        if max_val > min_val:
            heightmap = (heightmap - min_val) / (max_val - min_val)
        else:
            heightmap = np.zeros_like(heightmap)
        
        # Apply elevation curve with gentler slope for smoother terrain
        heightmap = np.power(heightmap, 1.1)  # Gentler power curve
        
        return heightmap
    
    def generate_chunk(self, chunk_x: int, chunk_y: int, chunk_size: int = 64) -> np.ndarray:
        """
        Generate a chunk of terrain at the given coordinates.
        
        Uses continuous world coordinates to ensure seamless transitions
        between chunks, preventing floating islands.
        
        Args:
            chunk_x: X coordinate of the chunk
            chunk_y: Y coordinate of the chunk
            chunk_size: Size of the chunk (chunk_size x chunk_size)
            
        Returns:
            2D numpy array representing the chunk heightmap
        """
        offset_x = chunk_x * chunk_size
        offset_y = chunk_y * chunk_size
        
        heightmap = np.zeros((chunk_size, chunk_size))
        
        for y in range(chunk_size):
            for x in range(chunk_size):
                # Use continuous world coordinates for seamless chunk boundaries
                world_x = offset_x + x
                world_z = offset_y + y
                
                # Large-scale mountain ranges (smoother with reduced persistence)
                mountains = pnoise2(
                    world_x / (self.terrain_scale * 4.0),
                    world_z / (self.terrain_scale * 4.0),
                    octaves=4,
                    persistence=0.5,  # Reduced for smoother transitions
                    lacunarity=2.0,
                    repeatx=1024,
                    repeaty=1024,
                    base=self.seed
                )
                
                # Medium-scale hills (smoother)
                hills = pnoise2(
                    world_x / self.terrain_scale,
                    world_z / self.terrain_scale,
                    octaves=5,  # Reduced octaves
                    persistence=0.4,  # Reduced for smoother transitions
                    lacunarity=2.0,
                    repeatx=1024,
                    repeaty=1024,
                    base=self.seed + 1000
                )
                
                # Small-scale details (smoother)
                details = pnoise2(
                    world_x / (self.terrain_scale * 0.5),
                    world_z / (self.terrain_scale * 0.5),
                    octaves=4,  # Reduced octaves
                    persistence=0.3,  # Reduced for smoother transitions
                    lacunarity=2.0,
                    repeatx=1024,
                    repeaty=1024,
                    base=self.seed + 2000
                )
                
                # Balanced weights for smoother transitions
                combined = mountains * 0.4 + hills * 0.4 + details * 0.2
                heightmap[y][x] = combined
        
        # Normalize to 0-1 range
        min_val = heightmap.min()
        max_val = heightmap.max()
        if max_val > min_val:
            heightmap = (heightmap - min_val) / (max_val - min_val)
        
        # Apply gentler elevation curve for smoother terrain
        heightmap = np.power(heightmap, 1.1)  # Gentler power curve
        
        return heightmap
    
    def get_terrain_type(self, height: float) -> str:
        """
        Determine terrain type based on height.
        
        Args:
            height: Normalized height value (0.0 to 1.0)
            
        Returns:
            Terrain type: 'water', 'sand', 'dirt', 'grass', 'hill', 'mountain', 'snow'
        """
        if height < 0.15:
            return 'water'  # Very low areas (valleys)
        elif height < 0.25:
            return 'sand'  # Low areas near water
        elif height < 0.35:
            return 'dirt'  # Lowlands
        elif height < 0.5:
            return 'grass'  # Flat grasslands
        elif height < 0.65:
            return 'hill'  # Hills
        elif height < 0.8:
            return 'mountain'  # Mountains
        else:
            return 'snow'  # Mountain peaks
    
    def generate_vegetation_noise(self, x: float, z: float) -> float:
        """
        Generate noise value for vegetation density.
        Higher values indicate more vegetation.
        
        Args:
            x: World X coordinate
            z: World Z coordinate
            
        Returns:
            Noise value between 0.0 and 1.0
        """
        vegetation_noise = pnoise2(
            x / 20.0,
            z / 20.0,
            octaves=4,
            persistence=0.5,
            lacunarity=2.0,
            repeatx=1024,
            repeaty=1024,
            base=self.seed + 5000
        )
        
        # Normalize to 0-1
        return (vegetation_noise + 1.0) / 2.0