"""Tests for houses."""

import pytest
from world_simulation.houses.house import House


class TestHouse:
    """Test house functionality."""
    
    def test_init(self):
        """Test house initialization."""
        house = House(0.0, 0.0, 0.0, capacity=5)
        assert house.x == 0.0
        assert house.y == 0.0
        assert house.z == 0.0
        assert house.capacity == 5
        assert house.is_built == True
        assert len(house.current_occupants) == 0
    
    def test_can_shelter(self):
        """Test shelter availability check."""
        house = House(0.0, 0.0, 0.0, capacity=2)
        assert house.can_shelter() == True
        
        house.add_occupant(1)
        assert house.can_shelter() == True
        
        house.add_occupant(2)
        assert house.can_shelter() == False
    
    def test_add_occupant(self):
        """Test adding occupants."""
        house = House(0.0, 0.0, 0.0, capacity=2)
        
        assert house.add_occupant(1) == True
        assert len(house.current_occupants) == 1
        
        assert house.add_occupant(2) == True
        assert len(house.current_occupants) == 2
        
        # Should fail when full
        assert house.add_occupant(3) == False
        assert len(house.current_occupants) == 2
    
    def test_remove_occupant(self):
        """Test removing occupants."""
        house = House(0.0, 0.0, 0.0, capacity=2)
        house.add_occupant(1)
        house.add_occupant(2)
        
        house.remove_occupant(1)
        assert len(house.current_occupants) == 1
        assert 1 not in house.current_occupants
        
        # Removing non-existent should not error
        house.remove_occupant(99)
        assert len(house.current_occupants) == 1
    
    def test_get_position(self):
        """Test getting house position."""
        house = House(10.0, 5.0, 15.0)
        pos = house.get_position()
        
        assert pos == (10.0, 5.0, 15.0)
    
    def test_distance_to(self):
        """Test distance calculation."""
        house = House(0.0, 0.0, 0.0)
        
        distance = house.distance_to(3.0, 4.0)
        
        # Should be 5.0 (3-4-5 triangle)
        assert abs(distance - 5.0) < 0.01

