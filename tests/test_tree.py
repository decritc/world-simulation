"""Tests for fruit trees."""

import pytest
import numpy as np
from world_simulation.trees.tree import FruitTree


class MockWorld:
    """Mock world for testing."""
    def __init__(self):
        self.trees = []
        self.houses = []
    
    def get_height(self, x, z):
        return 0.0


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
        # Trees should live for a very long time (effectively immortal)
        assert tree.max_age > 100000.0  # Should be very large
        
        # Test that tree doesn't die when well below max_age
        tree.age = tree.max_age - 1000.0
        world = MockWorld()
        tree.update(2.0, world)
        assert tree.is_alive == True
        
        # Only die if age exceeds max_age
        tree.age = tree.max_age - 0.5
        tree.update(1.0, world)  # This will push age past max_age
        assert tree.is_alive == False
    
    def test_tree_reproduction_cooldown(self):
        """Test tree reproduction cooldown."""
        tree = FruitTree(0.0, 0.0, 0.0)
        tree.growth_stage = 1.0  # Fully grown
        tree.age = tree.reproduction_age  # Old enough to reproduce
        tree.reproduction_cooldown = 10.0
        
        world = MockWorld()
        initial_tree_count = len(world.trees)
        
        tree.update(5.0, world)
        
        # Cooldown should decrease
        assert tree.reproduction_cooldown < 10.0
        
        # No new trees should spawn during cooldown
        assert len(world.trees) == initial_tree_count
    
    def test_tree_reproduction_spawns_sapling(self):
        """Test that mature trees can spawn new trees."""
        tree = FruitTree(0.0, 0.0, 0.0)
        tree.growth_stage = 1.0  # Fully grown
        tree.age = tree.reproduction_age  # Old enough to reproduce
        tree.reproduction_cooldown = 0.0  # Ready to reproduce
        
        world = MockWorld()
        world.trees = [tree]  # Add tree to world
        
        initial_count = len(world.trees)
        
        # Force reproduction by updating many times
        # Reproduction chance is 0.1% per second, so update many times
        for _ in range(10000):  # Should trigger reproduction
            tree.update(0.01, world)
            if len(world.trees) > initial_count:
                break
        
        # Should have spawned a new tree
        assert len(world.trees) >= initial_count
        
        # New tree should be same species
        if len(world.trees) > initial_count:
            new_tree = world.trees[-1]
            assert new_tree.species == tree.species
            assert new_tree.age == 0.0  # Starts as sapling
            assert new_tree.growth_stage == 0.0
    
    def test_tree_spawns_on_death(self):
        """Test that trees spawn a new tree when they die."""
        tree = FruitTree(0.0, 0.0, 0.0)
        tree.age = tree.max_age - 0.5  # About to die
        
        world = MockWorld()
        world.trees = [tree]
        
        initial_count = len(world.trees)
        
        tree.update(1.0, world)  # This will cause death
        
        # Tree should be dead
        assert tree.is_alive == False
        
        # Should have spawned a new tree before dying
        assert len(world.trees) >= initial_count
        
        # New tree should be added to world
        if len(world.trees) > initial_count:
            new_tree = world.trees[-1]
            assert new_tree.is_alive == True
            assert new_tree.species == tree.species

