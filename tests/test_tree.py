"""Tests for fruit trees."""

import pytest
import numpy as np
from world_simulation.trees.tree import FruitTree


class MockWorld:
    """Mock world for testing."""
    pass


class TestFruitTree:
    """Test fruit tree functionality."""
    
    def test_init(self):
        """Test tree initialization."""
        tree = FruitTree(0.0, 0.0, 0.0, species="apple")
        assert tree.x == 0.0
        assert tree.y == 0.0
        assert tree.z == 0.0
        assert tree.species == "apple"
        assert tree.is_alive == True
        assert tree.age == 0.0
        assert tree.growth_stage == 0.0
        assert tree.fruit_count == 0
    
    def test_growth(self):
        """Test tree growth."""
        tree = FruitTree(0.0, 0.0, 0.0)
        world = MockWorld()
        initial_stage = tree.growth_stage
        
        tree.update(15.0, world)
        
        assert tree.growth_stage > initial_stage
    
    def test_fruit_production(self):
        """Test fruit production."""
        tree = FruitTree(0.0, 0.0, 0.0)
        tree.growth_stage = 0.6  # Mature enough
        world = MockWorld()
        
        initial_fruit = tree.fruit_count
        
        # Update many times to allow fruit production
        for _ in range(100):
            tree.update(0.1, world)
        
        # Should have produced some fruit
        assert tree.fruit_count >= initial_fruit
    
    def test_fruit_maturation(self):
        """Test fruit maturation."""
        tree = FruitTree(0.0, 0.0, 0.0)
        tree.growth_stage = 0.6
        tree.fruit_maturity[0] = 0.5
        tree.fruit_count = 1
        world = MockWorld()
        
        tree.update(3.0, world)
        
        # Fruit should be mature (>= 1.0)
        assert tree.fruit_maturity[0] >= 1.0
    
    def test_harvest_fruit(self):
        """Test fruit harvesting."""
        tree = FruitTree(0.0, 0.0, 0.0)
        tree.fruit_maturity[0] = 1.0  # Ripe
        tree.fruit_maturity[1] = 1.0  # Ripe
        tree.fruit_maturity[2] = 0.5  # Not ripe
        tree.fruit_count = 3
        
        harvested = tree.harvest_fruit()
        
        assert harvested == 2  # Only ripe fruit
        assert tree.fruit_count == 1  # One unripe fruit remains
        assert len(tree.fruit_maturity) == 1
    
    def test_get_ripe_fruit_count(self):
        """Test getting ripe fruit count."""
        tree = FruitTree(0.0, 0.0, 0.0)
        tree.fruit_maturity[0] = 1.0
        tree.fruit_maturity[1] = 1.0
        tree.fruit_maturity[2] = 0.5
        
        ripe_count = tree.get_ripe_fruit_count()
        
        assert ripe_count == 2
    
    def test_tree_death(self):
        """Test tree death at max age."""
        tree = FruitTree(0.0, 0.0, 0.0)
        tree.age = tree.max_age - 1.0
        world = MockWorld()
        
        tree.update(2.0, world)
        
        assert tree.is_alive == False

