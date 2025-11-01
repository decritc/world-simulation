"""Tests for world generation."""

import pytest
import numpy as np
from world_simulation.world.generator import WorldGenerator


class TestWorldGenerator:
    """Test world generation functionality."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = WorldGenerator(seed=42)
        assert generator.seed == 42
    
    def test_generate_heightmap(self):
        """Test heightmap generation."""
        generator = WorldGenerator(seed=42)
        heightmap = generator.generate_heightmap(10, 10, scale=50.0)
        
        assert heightmap.shape == (10, 10)
        assert heightmap.min() >= 0.0
        assert heightmap.max() <= 1.0
    
    def test_generate_chunk(self):
        """Test chunk generation."""
        generator = WorldGenerator(seed=42)
        chunk = generator.generate_chunk(0, 0, chunk_size=64)
        
        assert chunk.shape == (64, 64)
        assert chunk.min() >= 0.0
        assert chunk.max() <= 1.0
    
    def test_deterministic_generation(self):
        """Test that same seed produces same results."""
        gen1 = WorldGenerator(seed=42)
        gen2 = WorldGenerator(seed=42)
        
        chunk1 = gen1.generate_chunk(0, 0)
        chunk2 = gen2.generate_chunk(0, 0)
        
        np.testing.assert_array_almost_equal(chunk1, chunk2)
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        gen1 = WorldGenerator(seed=42)
        gen2 = WorldGenerator(seed=100)
        
        chunk1 = gen1.generate_chunk(0, 0)
        chunk2 = gen2.generate_chunk(0, 0)
        
        # Should be different (but might rarely be similar)
        assert not np.allclose(chunk1, chunk2, atol=0.01)

