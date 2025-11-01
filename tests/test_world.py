"""Tests for world class."""

import pytest
import numpy as np
from world_simulation.world.world import World
from world_simulation.entities.npc import NPC
from world_simulation.trees.tree import FruitTree
from world_simulation.houses.house import House


class TestWorld:
    """Test world functionality."""
    
    def test_init(self):
        """Test world initialization."""
        world = World(seed=42)
        assert world.generator.seed == 42
        assert len(world.entities) == 0
        assert len(world.trees) == 0
        assert len(world.houses) == 0
        assert world.time == 0.0
        assert world.day_time == 0.0
        assert world.day_number == 0
    
    def test_get_height(self):
        """Test height retrieval."""
        world = World(seed=42)
        height = world.get_height(0.0, 0.0)
        assert isinstance(height, (int, float))
        assert height >= 0.0
    
    def test_get_chunk(self):
        """Test chunk retrieval."""
        world = World(seed=42)
        chunk = world.get_chunk(0, 0)
        assert chunk.shape == (64, 64)
    
    def test_is_night(self):
        """Test night detection."""
        world = World(seed=42)
        
        # At start (hour 0), should be night (before 6am)
        assert world.is_night() == True
        
        # Advance to day (10am)
        world.day_time = world.day_length * (10.0 / 24.0)
        assert world.is_night() == False
        
        # Advance to night (8pm)
        world.day_time = world.day_length * (20.0 / 24.0)
        assert world.is_night() == True
    
    def test_get_light_intensity(self):
        """Test light intensity calculation."""
        world = World(seed=42)
        
        # Night should have low intensity
        world.day_time = world.day_length * (2.0 / 24.0)  # 2am
        assert world.get_light_intensity() == 0.3
        
        # Day should have full intensity
        world.day_time = world.day_length * (12.0 / 24.0)  # Noon
        assert world.get_light_intensity() == 1.0
    
    def test_update_advances_time(self):
        """Test that update advances time."""
        world = World(seed=42)
        initial_time = world.time
        initial_day_time = world.day_time
        
        world.update(1.0)
        
        assert world.time > initial_time
        assert world.day_time > initial_day_time
    
    def test_day_advancement(self):
        """Test day advancement."""
        world = World(seed=42)
        initial_day = world.day_number
        
        # Advance past one day
        world.update(world.day_length + 1.0)
        
        assert world.day_number == initial_day + 1
        assert world.day_time < world.day_length
    
    def test_remove_dead_npcs(self):
        """Test that dead NPCs are removed."""
        world = World(seed=42)
        
        # Create some NPCs
        npc1 = NPC(0, 0, 0)
        npc2 = NPC(1, 0, 1)
        npc3 = NPC(2, 0, 2)
        
        world.entities = [npc1, npc2, npc3]
        assert len(world.entities) == 3
        
        # Kill one NPC
        npc2.is_alive = False
        npc2.health = 0
        
        # Update world
        world.update(0.1)
        
        # Dead NPC should be removed
        assert len(world.entities) == 2
        assert npc2 not in world.entities
        assert npc1 in world.entities
        assert npc3 in world.entities

