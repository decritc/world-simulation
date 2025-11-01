"""Vegetation system for procedural generation."""

from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class Vegetation:
    """Represents a vegetation instance (bush, grass patch, flower, etc.)."""
    
    x: float
    y: float  # Height on terrain
    z: float
    vegetation_type: str  # 'bush', 'grass', 'flower', 'rock'
    size: float
    is_alive: bool = True
    
    def __post_init__(self):
        """Initialize vegetation."""
        if self.size <= 0:
            self.size = 1.0


class VegetationGenerator:
    """Generates vegetation procedurally based on terrain."""
    
    VEGETATION_TYPES = ['bush', 'grass', 'flower', 'rock']
    
    def __init__(self, seed: int = 42):
        """Initialize vegetation generator."""
        self.seed = seed
        np.random.seed(seed)
    
    def generate_vegetation_for_area(
        self,
        x_min: float,
        x_max: float,
        z_min: float,
        z_max: float,
        height_func,
        vegetation_noise_func,
        density: float = 0.1
    ) -> List[Vegetation]:
        """
        Generate vegetation for an area.
        
        Args:
            x_min, x_max: X bounds of the area
            z_min, z_max: Z bounds of the area
            height_func: Function to get terrain height at (x, z)
            vegetation_noise_func: Function to get vegetation noise at (x, z)
            density: Vegetation density (0.0 to 1.0)
            
        Returns:
            List of Vegetation objects
        """
        vegetation = []
        
        # Grid-based generation for consistent placement
        grid_spacing = 5.0  # Check every 5 units
        x_steps = int((x_max - x_min) / grid_spacing)
        z_steps = int((z_max - z_min) / grid_spacing)
        
        for i in range(x_steps):
            for j in range(z_steps):
                x = x_min + i * grid_spacing + np.random.uniform(-1.0, 1.0)
                z = z_min + j * grid_spacing + np.random.uniform(-1.0, 1.0)
                
                # Get terrain height
                terrain_height = height_func(x, z)
                
                # Get vegetation noise
                veg_noise = vegetation_noise_func(x, z)
                
                # Determine if vegetation should spawn here
                # Higher vegetation noise = more likely to spawn
                # Lower terrain height = more likely to spawn (mountains have less vegetation)
                spawn_chance = veg_noise * density * (1.0 - terrain_height * 0.5)
                
                if np.random.random() < spawn_chance:
                    # Choose vegetation type based on terrain height
                    veg_type = self._choose_vegetation_type(terrain_height, veg_noise)
                    
                    # Determine size
                    size = self._get_vegetation_size(veg_type, terrain_height)
                    
                    # Create vegetation
                    veg = Vegetation(
                        x=x,
                        y=terrain_height,
                        z=z,
                        vegetation_type=veg_type,
                        size=size
                    )
                    vegetation.append(veg)
        
        return vegetation
    
    def _choose_vegetation_type(self, terrain_height: float, vegetation_noise: float) -> str:
        """Choose vegetation type based on terrain characteristics."""
        # Valleys and lowlands: more flowers and grass
        if terrain_height < 0.4:
            if vegetation_noise > 0.7:
                return 'flower'
            elif vegetation_noise > 0.4:
                return 'grass'
            else:
                return 'bush'
        # Hills: mix of bushes and grass
        elif terrain_height < 0.6:
            if vegetation_noise > 0.6:
                return 'bush'
            elif vegetation_noise > 0.3:
                return 'grass'
            else:
                return 'rock'
        # Mountains: mostly rocks, some bushes
        elif terrain_height < 0.8:
            if vegetation_noise > 0.5:
                return 'rock'
            else:
                return 'bush'
        # Peaks: only rocks
        else:
            return 'rock'
    
    def _get_vegetation_size(self, veg_type: str, terrain_height: float) -> float:
        """Get size for vegetation based on type and terrain."""
        base_sizes = {
            'bush': (0.3, 0.8),
            'grass': (0.1, 0.3),
            'flower': (0.15, 0.4),
            'rock': (0.2, 0.6)
        }
        
        min_size, max_size = base_sizes.get(veg_type, (0.2, 0.5))
        
        # Size varies slightly with terrain
        size_variation = 1.0 + (terrain_height - 0.5) * 0.2
        
        return np.random.uniform(min_size, max_size) * size_variation

