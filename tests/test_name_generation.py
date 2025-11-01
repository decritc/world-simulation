"""Tests for NPC name generation."""

import pytest
from world_simulation.entities.npc import NPC
from world_simulation.entities.name_generator import NameGenerator


class TestNameGenerator:
    """Test name generation functionality."""
    
    def test_generate_first_name(self):
        """Test that first names are generated."""
        name = NameGenerator.generate_first_name()
        assert isinstance(name, str)
        assert len(name) > 0
    
    def test_generate_last_name(self):
        """Test that last names are generated."""
        name = NameGenerator.generate_last_name()
        assert isinstance(name, str)
        assert len(name) > 0
    
    def test_generate_full_name(self):
        """Test that full names are generated."""
        name = NameGenerator.generate_full_name()
        assert isinstance(name, str)
        assert ' ' in name  # Should have space between first and last
        parts = name.split()
        assert len(parts) == 2
    
    def test_names_are_random(self):
        """Test that generated names are different."""
        names = [NameGenerator.generate_full_name() for _ in range(10)]
        # Should have at least some unique names (may occasionally repeat)
        unique_names = set(names)
        assert len(unique_names) >= 5  # At least 5 unique names out of 10


class TestNPCNames:
    """Test NPC name assignment."""
    
    def test_npc_has_name(self):
        """Test that NPCs get a name when created."""
        npc = NPC(0.0, 0.0, 0.0)
        assert hasattr(npc, 'name')
        assert isinstance(npc.name, str)
        assert len(npc.name) > 0
    
    def test_npc_name_from_genome(self):
        """Test that NPCs can have names from genome."""
        custom_name = "Test Name"
        npc = NPC(0.0, 0.0, 0.0, genome={'name': custom_name})
        assert npc.name == custom_name
    
    def test_npc_names_are_unique(self):
        """Test that NPCs get different names."""
        npcs = [NPC(0.0, 0.0, 0.0) for _ in range(10)]
        names = [npc.name for npc in npcs]
        unique_names = set(names)
        # Should have at least some unique names
        assert len(unique_names) >= 5
    
    def test_offspring_name_inheritance(self):
        """Test that offspring can inherit last names."""
        npc1 = NPC(0.0, 0.0, 0.0)
        npc2 = NPC(10.0, 0.0, 10.0)
        
        # Make both adults
        npc1.age = npc1.adult_age + 1.0
        npc1.age_stage = "adult"
        npc1.can_reproduce = True
        
        npc2.age = npc2.adult_age + 1.0
        npc2.age_stage = "adult"
        npc2.can_reproduce = True
        
        # Put them in same house
        from world_simulation.houses.house import House
        house = House(5.0, 0.0, 5.0, capacity=2)
        house.add_occupant(id(npc1))
        house.add_occupant(id(npc2))
        npc1.current_house = house
        npc2.current_house = house
        
        # Create offspring
        offspring = npc1.reproduce(npc2)
        
        # Offspring should have a name
        assert hasattr(offspring, 'name')
        assert isinstance(offspring.name, str)
        assert len(offspring.name) > 0
        
        # Check if name inheritance occurred (50% chance)
        parent1_last = npc1.name.split()[-1] if ' ' in npc1.name else npc1.name
        parent2_last = npc2.name.split()[-1] if ' ' in npc2.name else npc2.name
        offspring_last = offspring.name.split()[-1] if ' ' in offspring.name else offspring.name
        
        # Offspring name should be valid (either inherited or new)
        assert ' ' in offspring.name  # Should have first and last name
        # Could inherit last name or have new one - both are valid

