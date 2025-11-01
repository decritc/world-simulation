"""Tests for improved house door and entry functionality."""

import pytest
from world_simulation.houses.house import House


class TestHouseDoor:
    """Test door functionality in houses."""
    
    def test_door_position(self):
        """Test that door position is correctly set."""
        house = House(10.0, 5.0, 15.0)
        
        door_x, door_y, door_z = house.get_door_position()
        
        # Door should be at house center X, but front face Z (+1.5)
        assert door_x == 10.0
        assert door_y == 5.0
        assert door_z == 16.5  # z + 1.5
        
    def test_distance_to_door(self):
        """Test distance calculation to door."""
        house = House(0.0, 0.0, 0.0)
        
        # Distance to door from origin
        distance = house.distance_to_door(0.0, 1.5)
        assert abs(distance - 0.0) < 0.01
        
        # Distance from a point away from door
        distance = house.distance_to_door(3.0, 4.0)
        # Door is at (0, 0+1.5) = (0, 1.5)
        # Distance from (3, 4) to (0, 1.5) = sqrt(9 + 6.25) = sqrt(15.25)
        expected = (3**2 + (4 - 1.5)**2)**0.5
        assert abs(distance - expected) < 0.01
    
    def test_house_colors(self):
        """Test that houses have random colors."""
        house1 = House(0.0, 0.0, 0.0)
        house2 = House(10.0, 0.0, 10.0)
        house3 = House(20.0, 0.0, 20.0)
        
        # All houses should have colors
        assert hasattr(house1, 'wall_color')
        assert hasattr(house1, 'roof_color')
        assert len(house1.wall_color) == 3
        assert len(house1.roof_color) == 3
        
        # Colors should be valid RGB values
        assert 0.0 <= house1.wall_color[0] <= 1.0
        assert 0.0 <= house1.wall_color[1] <= 1.0
        assert 0.0 <= house1.wall_color[2] <= 1.0
        
        assert 0.0 <= house1.roof_color[0] <= 1.0
        assert 0.0 <= house1.roof_color[1] <= 1.0
        assert 0.0 <= house1.roof_color[2] <= 1.0
        
        # Colors may or may not be different (random)
        # But they should exist
    
    def test_door_entry_point(self):
        """Test that door entry point is accessible."""
        house = House(5.0, 2.0, 8.0)
        
        door_pos = house.get_door_position()
        
        # Verify door position is valid
        assert isinstance(door_pos, tuple)
        assert len(door_pos) == 3
        assert door_pos[0] == 5.0  # Same X as house
        assert door_pos[1] == 2.0  # Same Y as house
        assert door_pos[2] == 9.5  # Z + 1.5 (front face)
    
    def test_door_distance_vs_house_distance(self):
        """Test that door distance differs from house center distance."""
        house = House(0.0, 0.0, 0.0)
        
        # Test point
        test_x, test_z = 3.0, 4.0
        
        house_dist = house.distance_to(test_x, test_z)
        door_dist = house.distance_to_door(test_x, test_z)
        
        # Door distance should differ from house center distance
        # (door is at (0, 1.5), house center is at (0, 0))
        assert abs(door_dist - house_dist) > 0.1
        
        # Both distances should be valid
        assert door_dist > 0
        assert house_dist > 0


class TestHouseIntegration:
    """Test house integration with NPC entry."""
    
    def test_npc_can_approach_door(self):
        """Test that NPCs can find and approach the door."""
        from world_simulation.world.world import World
        
        world = World(seed=42)
        
        # Create a house
        house = House(10.0, 0.0, 10.0)
        world.houses.append(house)
        
        # Create an NPC nearby
        from world_simulation.entities.npc import NPC
        npc = NPC(8.0, 0.0, 8.0)
        npc.age_stage = "adult"
        
        # NPC should be able to find door
        door_x, door_y, door_z = house.get_door_position()
        
        # Distance to door
        door_distance = house.distance_to_door(npc.x, npc.z)
        assert door_distance > 0
        
        # Distance to house center
        house_distance = house.distance_to(npc.x, npc.z)
        
        # Door should be further (it's on front face)
        assert door_distance >= house_distance

