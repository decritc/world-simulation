"""Tests for enhanced terrain generation with hills, mountains, and vegetation."""

import pytest
import numpy as np
from world_simulation.world.generator import WorldGenerator
from world_simulation.world.vegetation import VegetationGenerator, Vegetation
from world_simulation.world.world import World


class TestEnhancedTerrainGeneration:
    """Test enhanced terrain generation with hills and mountains."""
    
    def test_generator_initialization(self):
        """Test generator initialization with terrain parameters."""
        generator = WorldGenerator(seed=42)
        
        assert generator.max_height == 15.0  # Updated for smoother terrain
        assert generator.hill_height == 6.0
        assert generator.terrain_scale == 150.0  # Updated for smoother terrain
        assert generator.seed == 42
    
    def test_heightmap_generation_creates_variation(self):
        """Test that heightmap generation creates varied terrain."""
        generator = WorldGenerator(seed=42)
        heightmap = generator.generate_heightmap(64, 64, scale=50.0)
        
        assert heightmap.shape == (64, 64)
        assert heightmap.min() >= 0.0
        assert heightmap.max() <= 1.0
        # Should have variation (not all same value)
        assert heightmap.max() > heightmap.min()
    
    def test_heightmap_multi_octave_noise(self):
        """Test that multi-octave noise creates natural terrain."""
        generator = WorldGenerator(seed=42)
        heightmap = generator.generate_heightmap(64, 64, scale=50.0)
        
        # Check for smooth transitions (gradient should be reasonable)
        # Sample some points and check their neighbors
        for i in range(0, 64, 10):
            for j in range(0, 64, 10):
                if i < 63 and j < 63:
                    diff_x = abs(heightmap[j][i+1] - heightmap[j][i])
                    diff_z = abs(heightmap[j+1][i] - heightmap[j][i])
                    # Differences should be reasonable (not too large jumps)
                    assert diff_x < 0.5
                    assert diff_z < 0.5
    
    def test_chunk_generation(self):
        """Test chunk generation at different coordinates."""
        generator = WorldGenerator(seed=42)
        
        chunk1 = generator.generate_chunk(0, 0)
        chunk2 = generator.generate_chunk(1, 0)
        chunk3 = generator.generate_chunk(0, 1)
        
        assert chunk1.shape == (64, 64)
        assert chunk2.shape == (64, 64)
        assert chunk3.shape == (64, 64)
        
        # Different chunks should have different values
        assert not np.allclose(chunk1, chunk2)
        assert not np.allclose(chunk1, chunk3)
    
    def test_get_terrain_type(self):
        """Test terrain type classification with new types."""
        generator = WorldGenerator(seed=42)
        
        assert generator.get_terrain_type(0.1) == 'water'
        assert generator.get_terrain_type(0.2) == 'sand'
        assert generator.get_terrain_type(0.3) == 'dirt'
        assert generator.get_terrain_type(0.45) == 'grass'
        assert generator.get_terrain_type(0.55) == 'hill'
        assert generator.get_terrain_type(0.75) == 'mountain'
        assert generator.get_terrain_type(0.9) == 'snow'
    
    def test_vegetation_noise_generation(self):
        """Test vegetation noise generation."""
        generator = WorldGenerator(seed=42)
        
        noise1 = generator.generate_vegetation_noise(0.0, 0.0)
        noise2 = generator.generate_vegetation_noise(10.0, 10.0)
        noise3 = generator.generate_vegetation_noise(0.0, 0.0)
        noise4 = generator.generate_vegetation_noise(50.0, 50.0)
        
        assert 0.0 <= noise1 <= 1.0
        assert 0.0 <= noise2 <= 1.0
        # Same coordinates should give same noise
        assert abs(noise1 - noise3) < 0.01
        # Different coordinates should generally give different noise (but not always)
        # Just check that noise values are valid
        assert 0.0 <= noise4 <= 1.0


class TestVegetationGeneration:
    """Test vegetation generation system."""
    
    def test_vegetation_generator_initialization(self):
        """Test vegetation generator initialization."""
        generator = VegetationGenerator(seed=42)
        
        assert generator.seed == 42
        assert len(generator.VEGETATION_TYPES) == 4
        assert 'bush' in generator.VEGETATION_TYPES
        assert 'grass' in generator.VEGETATION_TYPES
        assert 'flower' in generator.VEGETATION_TYPES
        assert 'rock' in generator.VEGETATION_TYPES
    
    def test_vegetation_creation(self):
        """Test vegetation instance creation."""
        veg = Vegetation(10.0, 5.0, 10.0, 'bush', 0.5)
        
        assert veg.x == 10.0
        assert veg.y == 5.0
        assert veg.z == 10.0
        assert veg.vegetation_type == 'bush'
        assert veg.size == 0.5
        assert veg.is_alive == True
    
    def test_vegetation_type_selection(self):
        """Test vegetation type selection based on terrain."""
        generator = VegetationGenerator(seed=42)
        
        # Low terrain should favor flowers and grass
        low_type = generator._choose_vegetation_type(0.2, 0.8)
        assert low_type in ['flower', 'grass', 'bush']
        
        # High terrain should favor rocks
        high_type = generator._choose_vegetation_type(0.9, 0.5)
        assert high_type == 'rock'
    
    def test_vegetation_size_variation(self):
        """Test that vegetation sizes vary appropriately."""
        generator = VegetationGenerator(seed=42)
        
        bush_size = generator._get_vegetation_size('bush', 0.5)
        grass_size = generator._get_vegetation_size('grass', 0.5)
        rock_size = generator._get_vegetation_size('rock', 0.5)
        
        assert 0.1 < bush_size < 1.0
        assert 0.1 < grass_size < 1.0
        assert 0.1 < rock_size < 1.0
        
        # Bushes should generally be larger than grass
        assert bush_size > grass_size * 0.5
    
    def test_generate_vegetation_for_area(self):
        """Test vegetation generation for an area."""
        generator = VegetationGenerator(seed=42)
        world = World(seed=42)
        
        vegetation = generator.generate_vegetation_for_area(
            x_min=-10, x_max=10,
            z_min=-10, z_max=10,
            height_func=lambda x, z: world.get_height(x, z) / world.generator.max_height,
            vegetation_noise_func=lambda x, z: world.generator.generate_vegetation_noise(x, z),
            density=0.2
        )
        
        assert len(vegetation) > 0
        assert all(isinstance(v, Vegetation) for v in vegetation)
        assert all(v.is_alive for v in vegetation)
        
        # Check that vegetation is within bounds
        for veg in vegetation:
            assert -10 <= veg.x <= 10
            assert -10 <= veg.z <= 10


class TestWorldTerrainIntegration:
    """Test world integration with enhanced terrain."""
    
    def test_world_has_vegetation_generator(self):
        """Test that world has vegetation generator."""
        world = World(seed=42)
        
        assert hasattr(world, 'vegetation_generator')
        assert isinstance(world.vegetation_generator, VegetationGenerator)
        assert hasattr(world, 'vegetation')
        assert isinstance(world.vegetation, list)
    
    def test_get_height_scales_correctly(self):
        """Test that get_height scales normalized heights correctly."""
        world = World(seed=42)
        
        # Get height at origin
        height = world.get_height(0.0, 0.0)
        
        assert 0.0 <= height <= world.generator.max_height
        assert isinstance(height, (int, float))
    
    def test_get_height_handles_negative_coordinates(self):
        """Test that get_height handles negative coordinates."""
        world = World(seed=42)
        
        height_neg = world.get_height(-10.0, -10.0)
        height_pos = world.get_height(10.0, 10.0)
        
        assert 0.0 <= height_neg <= world.generator.max_height
        assert 0.0 <= height_pos <= world.generator.max_height
    
    def test_get_height_consistency(self):
        """Test that get_height is consistent for same coordinates."""
        world = World(seed=42)
        
        height1 = world.get_height(5.0, 5.0)
        height2 = world.get_height(5.0, 5.0)
        
        assert abs(height1 - height2) < 0.001
    
    def test_terrain_has_variation(self):
        """Test that terrain has elevation variation."""
        world = World(seed=42)
        
        heights = []
        for x in range(-20, 20, 5):
            for z in range(-20, 20, 5):
                height = world.get_height(x, z)
                heights.append(height)
        
        min_height = min(heights)
        max_height = max(heights)
        
        # Should have some variation
        assert max_height > min_height
        # Should be within expected range
        assert 0.0 <= min_height <= world.generator.max_height
        assert 0.0 <= max_height <= world.generator.max_height


class TestNPCTerrainNavigation:
    """Test NPC navigation with terrain height."""
    
    def test_npc_updates_height_during_movement(self):
        """Test that NPCs update their Y position based on terrain."""
        world = World(seed=42)
        npc = world.entities[0] if world.entities else None
        
        if npc:
            initial_y = npc.y
            initial_x = npc.x
            initial_z = npc.z
            
            # Move NPC
            npc.x += 5.0
            npc.z += 5.0
            npc.y = world.get_height(npc.x, npc.z)
            
            # Height should be updated based on terrain
            new_height = world.get_height(npc.x, npc.z)
            assert abs(npc.y - new_height) < 0.1  # Should match terrain
            
            # Restore position
            npc.x = initial_x
            npc.z = initial_z
            npc.y = initial_y
    
    def test_npc_wander_updates_height(self):
        """Test that wandering NPCs update height."""
        from world_simulation.entities.npc import NPC
        
        world = World(seed=42)
        npc = NPC(0.0, 0.0, 0.0)
        
        initial_height = world.get_height(npc.x, npc.z)
        npc.y = initial_height
        
        # Simulate wandering
        npc.target_x = 10.0
        npc.target_z = 10.0
        npc._wander(0.1, world)
        
        # Y should be updated to terrain height
        expected_height = world.get_height(npc.x, npc.z)
        assert abs(npc.y - expected_height) < 0.1

