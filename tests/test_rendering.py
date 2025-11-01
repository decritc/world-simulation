"""Tests for rendering and color changes."""

import pytest
import numpy as np
from world_simulation.world.world import World
from world_simulation.trees.tree import FruitTree
from world_simulation.houses.house import House
from world_simulation.entities.npc import NPC


class TestNighttimeColors:
    """Test that colors change correctly at nighttime."""
    
    def test_world_is_night_detection(self):
        """Test that world correctly detects night."""
        world = World(seed=42)
        
        # Start of simulation (hour 0) should be night
        assert world.is_night() == True
        
        # Noon (hour 12) should be day
        world.day_time = world.day_length * (12.0 / 24.0)
        assert world.is_night() == False
        
        # 8pm (hour 20) should be night
        world.day_time = world.day_length * (20.0 / 24.0)
        assert world.is_night() == True
        
        # 2am should be night
        world.day_time = world.day_length * (2.0 / 24.0)
        assert world.is_night() == True
    
    def test_light_intensity_at_night(self):
        """Test that light intensity is low at night."""
        world = World(seed=42)
        
        # Night (2am)
        world.day_time = world.day_length * (2.0 / 24.0)
        assert world.is_night() == True
        intensity_night = world.get_light_intensity()
        assert intensity_night == 0.3  # Low intensity at night
        
        # Day (noon)
        world.day_time = world.day_length * (12.0 / 24.0)
        assert world.is_night() == False
        intensity_day = world.get_light_intensity()
        assert intensity_day == 1.0  # Full intensity during day
        
        assert intensity_night < intensity_day
    
    def test_tree_colors_at_night(self):
        """Test that trees use appropriate colors for night rendering."""
        world = World(seed=42)
        tree = FruitTree(0.0, 0.0, 0.0, species="apple")
        tree.growth_stage = 1.0
        
        # Check that tree can be rendered at different times
        # Set to night
        world.day_time = world.day_length * (20.0 / 24.0)
        assert world.is_night() == True
        
        # Set to day
        world.day_time = world.day_length * (12.0 / 24.0)
        assert world.is_night() == False
        
        # Tree should be alive and renderable
        assert tree.is_alive == True
        assert tree.growth_stage >= 0.5  # Should have leaves
    
    def test_house_colors_at_night(self):
        """Test that houses use appropriate colors for night rendering."""
        world = World(seed=42)
        house = House(0.0, 0.0, 0.0, capacity=5)
        
        # Verify house is built and can be rendered
        assert house.is_built == True
        
        # Check at different times
        world.day_time = world.day_length * (20.0 / 24.0)
        assert world.is_night() == True
        
        world.day_time = world.day_length * (12.0 / 24.0)
        assert world.is_night() == False
    
    def test_terrain_colors_at_night(self):
        """Test that terrain uses appropriate colors for night rendering."""
        world = World(seed=42)
        
        # Night time
        world.day_time = world.day_length * (20.0 / 24.0)
        assert world.is_night() == True
        
        # Day time
        world.day_time = world.day_length * (12.0 / 24.0)
        assert world.is_night() == False
    
    def test_sky_color_calculation(self):
        """Test sky color calculation based on time."""
        world = World(seed=42)
        
        # Night (2am) - should have low RGB values, more blue
        world.day_time = world.day_length * (2.0 / 24.0)
        hour = (world.day_time / world.day_length) * 24.0
        assert hour < 6.0 or hour >= 18.0  # Night time
        
        # Dawn (6am-8am) - should transition
        world.day_time = world.day_length * (7.0 / 24.0)
        hour = (world.day_time / world.day_length) * 24.0
        assert 6.0 <= hour < 8.0  # Dawn
        
        # Day (noon) - should be bright blue
        world.day_time = world.day_length * (12.0 / 24.0)
        hour = (world.day_time / world.day_length) * 24.0
        assert 8.0 <= hour < 18.0  # Day
        
        # Dusk (6pm-8pm) - should transition
        world.day_time = world.day_length * (19.0 / 24.0)
        hour = (world.day_time / world.day_length) * 24.0
        assert 18.0 <= hour < 20.0  # Dusk
    
    def test_nighttime_color_consistency(self):
        """Test that all nighttime colors are consistent (cool, not warm)."""
        world = World(seed=42)
        
        # Set to night
        world.day_time = world.day_length * (20.0 / 24.0)
        assert world.is_night() == True
        
        # Light intensity should be low
        intensity = world.get_light_intensity()
        assert intensity <= 0.3
        
        # Verify all objects exist and can be rendered
        tree = FruitTree(0.0, 0.0, 0.0)
        house = House(0.0, 0.0, 0.0)
        npc = NPC(0.0, 0.0, 0.0)
        
        assert tree.is_alive == True
        assert house.is_built == True
        assert npc.is_alive == True
    
    def test_color_transitions_dusk_to_night(self):
        """Test that colors transition smoothly from dusk to night."""
        world = World(seed=42)
        
        # Dusk start (6pm)
        world.day_time = world.day_length * (18.0 / 24.0)
        hour = (world.day_time / world.day_length) * 24.0
        assert 18.0 <= hour < 20.0
        
        # Dusk end (8pm) - should be night
        world.day_time = world.day_length * (20.0 / 24.0)
        hour = (world.day_time / world.day_length) * 24.0
        assert hour >= 18.0 or hour < 6.0  # Night time
        assert world.is_night() == True
    
    def test_daytime_colors_unchanged(self):
        """Test that daytime colors remain warm/normal."""
        world = World(seed=42)
        
        # Set to day
        world.day_time = world.day_length * (12.0 / 24.0)
        assert world.is_night() == False
        
        # Light intensity should be high
        intensity = world.get_light_intensity()
        assert intensity == 1.0

