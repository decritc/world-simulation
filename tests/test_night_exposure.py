"""Tests for NPC night exposure health damage."""

import pytest
import numpy as np
from world_simulation.entities.npc import NPC


class MockWorld:
    """Mock world for testing."""
    def __init__(self):
        self.trees = []
        self.houses = []
        self.day_time = 60.0  # Noon
        self.day_length = 120.0
    
    def is_night(self):
        hour = (self.day_time / self.day_length) * 24.0
        return hour < 6.0 or hour >= 18.0
    
    def get_height(self, x, z):
        return 0.0


class TestNightExposureDamage:
    """Test NPC health damage from being outside at night."""
    
    def test_night_exposure_increases_health_loss(self):
        """Test that NPCs outside at night lose health faster."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        # Set to night
        world.day_time = 20.0 * world.day_length / 24.0  # 8 PM
        assert world.is_night() == True
        
        # NPC is outside (no house)
        npc.current_house = None
        npc.state = "wandering"
        npc.hunger = 100.0  # Well fed to isolate night damage
        
        initial_health = npc.health
        
        # Update for some time
        npc.update(10.0, world)
        
        # Health should have decreased
        assert npc.health < initial_health
        
        # Reset to day time
        world.day_time = 12.0 * world.day_length / 24.0  # Noon
        assert world.is_night() == False
        
        # Create another NPC for comparison
        npc2 = NPC(0.0, 0.0, 0.0)
        npc2.hunger = 100.0  # Well fed
        npc2.current_house = None
        npc2.state = "wandering"
        initial_health2 = npc2.health
        
        # Update for same time during day
        npc2.update(10.0, world)
        
        # Night NPC should have lost more health than day NPC
        night_health_lost = initial_health - npc.health
        day_health_lost = initial_health2 - npc2.health
        
        assert night_health_lost > day_health_lost
    
    def test_shelter_protects_from_night_damage(self):
        """Test that NPCs in shelter don't take extra night damage."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        # Set to night
        world.day_time = 20.0 * world.day_length / 24.0  # 8 PM
        assert world.is_night() == True
        
        # NPC is in shelter
        from world_simulation.houses.house import House
        house = House(0.0, 0.0, 0.0)
        npc.current_house = house
        npc.state = "in_shelter"
        npc.hunger = 100.0  # Well fed
        
        initial_health = npc.health
        
        # Update for some time
        npc.update(10.0, world)
        
        # Health should decrease, but not as much as outside
        assert npc.health < initial_health
        
        # Compare with NPC outside at night
        npc_outside = NPC(0.0, 0.0, 0.0)
        npc_outside.current_house = None
        npc_outside.state = "wandering"
        npc_outside.hunger = 100.0
        initial_health_outside = npc_outside.health
        
        npc_outside.update(10.0, world)
        
        # NPC outside should have lost more health
        shelter_health_lost = initial_health - npc.health
        outside_health_lost = initial_health_outside - npc_outside.health
        
        assert outside_health_lost > shelter_health_lost
    
    def test_daytime_no_extra_damage(self):
        """Test that NPCs don't take extra damage during day."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        # Set to day
        world.day_time = 12.0 * world.day_length / 24.0  # Noon
        assert world.is_night() == False
        
        # NPC is outside
        npc.current_house = None
        npc.state = "wandering"
        npc.hunger = 100.0  # Well fed
        
        initial_health = npc.health
        
        # Update for some time
        npc.update(10.0, world)
        
        # Health should decrease only from normal aging
        # Should be less than if it were night
        assert npc.health < initial_health
        
        # Compare with night
        world.day_time = 20.0 * world.day_length / 24.0  # 8 PM
        npc2 = NPC(0.0, 0.0, 0.0)
        npc2.current_house = None
        npc2.state = "wandering"
        npc2.hunger = 100.0
        initial_health2 = npc2.health
        
        npc2.update(10.0, world)
        
        # Night NPC should have lost more health
        day_health_lost = initial_health - npc.health
        night_health_lost = initial_health2 - npc2.health
        
        assert night_health_lost > day_health_lost

