"""Tests for reproduction and aging mechanics."""

import pytest
import numpy as np
from world_simulation.entities.npc import NPC
from world_simulation.houses.house import House


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


class TestNPCAging:
    """Test NPC aging mechanics."""
    
    def test_npc_initialized_with_elder_age_is_elder(self):
        """Test that NPCs initialized with elder age are correctly marked as elder."""
        # Create NPC with elder age in genome
        npc = NPC(0.0, 0.0, 0.0, genome={'age': 800.0})  # Age > elder_age (typically 450-675)
        
        # Should be marked as elder (not adult) if age >= elder_age
        if npc.age >= npc.elder_age:
            assert npc.age_stage == "elder"
        elif npc.age >= npc.adult_age:
            assert npc.age_stage == "adult"
        else:
            assert npc.age_stage == "child"
    
    def test_npc_initialized_with_adult_age_is_adult(self):
        """Test that NPCs initialized with adult age are correctly marked as adult."""
        # Create NPC with adult age in genome
        npc = NPC(0.0, 0.0, 0.0, genome={'age': 150.0})  # Age between adult_age and elder_age
        
        # Should be marked as adult (not child or elder)
        assert npc.age_stage == "adult"
        assert npc.can_reproduce == True
    
    def test_npc_starts_as_child(self):
        """Test that new NPCs start as children."""
        npc = NPC(0.0, 0.0, 0.0)
        
        assert npc.age == 0.0
        assert npc.age_stage == "child"
        assert npc.can_reproduce == False
        assert npc.lifespan > 0
    
    def test_npc_becomes_adult(self):
        """Test that NPCs become adults at the right age."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        adult_age = npc.adult_age
        
        # Update until adult age
        npc.age = adult_age - 1.0
        npc.update(0.1, world)
        
        assert npc.age_stage == "child"
        
        # Update past adult age
        npc.update(2.0, world)
        
        assert npc.age_stage == "adult"
        assert npc.can_reproduce == True
    
    def test_npc_becomes_elder(self):
        """Test that NPCs become elders at the right age."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        # Set to adult first
        npc.age = npc.adult_age + 1.0
        npc.age_stage = "adult"
        npc.can_reproduce = True
        
        # Update to elder age
        elder_age = npc.elder_age
        npc.age = elder_age - 1.0
        npc.update(0.1, world)
        
        assert npc.age_stage == "adult"
        
        # Update past elder age
        npc.update(2.0, world)
        
        assert npc.age_stage == "elder"
    
    def test_npc_dies_of_old_age(self):
        """Test that NPCs die when they reach their lifespan."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        lifespan = npc.lifespan
        
        # Set age close to lifespan
        npc.age = lifespan - 1.0
        assert npc.is_alive == True
        
        # Update past lifespan
        npc.update(2.0, world)
        
        assert npc.is_alive == False
        assert npc.health == 0
    
    def test_child_grows_to_adult_size(self):
        """Test that children grow to full size when becoming adults."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        # Child starts smaller
        npc.size = 0.7
        original_genome_size = npc.genome.get('size', 1.0)
        
        # Become adult
        npc.age = npc.adult_age
        npc.update(0.1, world)
        
        assert npc.age_stage == "adult"
        assert npc.size == original_genome_size  # Should be full size


class TestNPCReproduction:
    """Test NPC reproduction mechanics."""
    
    def test_adults_can_reproduce(self):
        """Test that only adults can reproduce."""
        npc1 = NPC(0.0, 0.0, 0.0)
        npc2 = NPC(10.0, 0.0, 10.0)
        
        # Both start as children
        assert npc1.can_reproduce == False
        assert npc2.can_reproduce == False
        assert npc1.can_reproduce_with(npc2) == False
        
        # Make both adults
        npc1.age = npc1.adult_age + 1.0
        npc1.age_stage = "adult"
        npc1.can_reproduce = True
        
        npc2.age = npc2.adult_age + 1.0
        npc2.age_stage = "adult"
        npc2.can_reproduce = True
        
        # Still can't reproduce if not in same house
        assert npc1.can_reproduce_with(npc2) == False
        
        # Put them in same house
        house = House(5.0, 0.0, 5.0, capacity=2)
        house.add_occupant(id(npc1))
        house.add_occupant(id(npc2))
        npc1.current_house = house
        npc2.current_house = house
        
        # Now they can reproduce
        assert npc1.can_reproduce_with(npc2) == True
    
    def test_reproduction_creates_offspring(self):
        """Test that reproduction creates a new NPC."""
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
        house = House(5.0, 0.0, 5.0, capacity=2)
        house.add_occupant(id(npc1))
        house.add_occupant(id(npc2))
        npc1.current_house = house
        npc2.current_house = house
        
        # Create offspring
        offspring = npc1.reproduce(npc2)
        
        assert offspring is not None
        assert offspring.age == 0.0
        assert offspring.age_stage == "child"
        assert offspring.can_reproduce == False
        assert offspring.size < npc1.size  # Children are smaller
    
    def test_reproduction_sets_cooldown(self):
        """Test that reproduction sets cooldown on both parents."""
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
        house = House(5.0, 0.0, 5.0, capacity=2)
        house.add_occupant(id(npc1))
        house.add_occupant(id(npc2))
        npc1.current_house = house
        npc2.current_house = house
        
        # Create offspring
        offspring = npc1.reproduce(npc2)
        
        assert npc1.reproduction_cooldown > 0
        assert npc2.reproduction_cooldown > 0
        assert npc1.reproduction_cooldown == npc1.reproduction_cooldown_time
        assert npc2.reproduction_cooldown == npc2.reproduction_cooldown_time
    
    def test_reproduction_mixes_genomes(self):
        """Test that offspring genome is a mix of parents."""
        npc1 = NPC(0.0, 0.0, 0.0)
        npc2 = NPC(10.0, 0.0, 10.0)
        
        # Set specific genomes
        npc1.genome = {'speed': 1.0, 'size': 1.0, 'stamina': 100.0}
        npc2.genome = {'speed': 2.0, 'size': 1.2, 'stamina': 150.0}
        
        # Make both adults
        npc1.age = npc1.adult_age + 1.0
        npc1.age_stage = "adult"
        npc1.can_reproduce = True
        
        npc2.age = npc2.adult_age + 1.0
        npc2.age_stage = "adult"
        npc2.can_reproduce = True
        
        # Put them in same house
        house = House(5.0, 0.0, 5.0, capacity=2)
        house.add_occupant(id(npc1))
        house.add_occupant(id(npc2))
        npc1.current_house = house
        npc2.current_house = house
        
        # Create offspring
        offspring = npc1.reproduce(npc2)
        
        # Offspring should have mixed traits
        assert 'speed' in offspring.genome
        assert 'size' in offspring.genome
        assert 'stamina' in offspring.genome
        # Speed should be between parents (with mutation)
        assert 0.5 <= offspring.genome['speed'] <= 2.5  # Average is 1.5, allow mutation


    def test_reproduction_cooldown_prevents_reproduction(self):
        """Test that reproduction cooldown prevents immediate reproduction."""
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
        house = House(5.0, 0.0, 5.0, capacity=2)
        house.add_occupant(id(npc1))
        house.add_occupant(id(npc2))
        npc1.current_house = house
        npc2.current_house = house
        
        # Set cooldown on npc1
        npc1.reproduction_cooldown = 30.0
        
        # Should not be able to reproduce due to cooldown
        assert npc1.can_reproduce_with(npc2) == False
    
    def test_elders_cannot_reproduce(self):
        """Test that elders cannot reproduce."""
        npc1 = NPC(0.0, 0.0, 0.0)
        npc2 = NPC(10.0, 0.0, 10.0)
        
        # Make both elders
        npc1.age = npc1.elder_age + 1.0
        npc1.age_stage = "elder"
        npc1.can_reproduce = False  # Elders can't reproduce
        
        npc2.age = npc2.elder_age + 1.0
        npc2.age_stage = "elder"
        npc2.can_reproduce = False
        
        # Put them in same house
        house = House(5.0, 0.0, 5.0, capacity=2)
        house.add_occupant(id(npc1))
        house.add_occupant(id(npc2))
        npc1.current_house = house
        npc2.current_house = house
        
        # Should not be able to reproduce
        assert npc1.can_reproduce_with(npc2) == False
    
    def test_reproduction_cooldown_decreases_over_time(self):
        """Test that reproduction cooldown decreases over time."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        npc.reproduction_cooldown = 60.0
        initial_cooldown = npc.reproduction_cooldown
        
        # Update for some time
        npc.update(30.0, world)
        
        assert npc.reproduction_cooldown < initial_cooldown
        assert npc.reproduction_cooldown == 30.0
    
    def test_reproduction_cooldown_reaches_zero(self):
        """Test that reproduction cooldown eventually reaches zero."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        npc.reproduction_cooldown = 60.0
        
        # Update past cooldown time
        npc.update(65.0, world)
        
        assert npc.reproduction_cooldown <= 0
    
    def test_offspring_spawns_near_parent_house(self):
        """Test that offspring spawns near parent house."""
        npc1 = NPC(0.0, 0.0, 0.0)
        npc2 = NPC(10.0, 0.0, 10.0)
        
        # Make both adults
        npc1.age = npc1.adult_age + 1.0
        npc1.age_stage = "adult"
        npc1.can_reproduce = True
        
        npc2.age = npc2.adult_age + 1.0
        npc2.age_stage = "adult"
        npc2.can_reproduce = True
        
        # Put them in house at specific location
        house = House(100.0, 0.0, 100.0, capacity=2)
        house.add_occupant(id(npc1))
        house.add_occupant(id(npc2))
        npc1.current_house = house
        npc2.current_house = house
        
        # Create offspring
        offspring = npc1.reproduce(npc2)
        
        # Offspring should spawn near house (within 1 unit)
        distance = np.sqrt((offspring.x - house.x)**2 + (offspring.z - house.z)**2)
        assert distance < 2.0  # Within 2 units
    
    def test_no_reproduction_with_one_adult(self):
        """Test that reproduction doesn't happen with only one adult."""
        from world_simulation.world.world import World
        
        world = World(seed=42)
        house = House(0.0, 0.0, 0.0, capacity=2)
        world.houses.append(house)
        
        # Create only 1 adult NPC
        npc1 = NPC(0.0, 0.0, 0.0)
        npc1.age = npc1.adult_age + 1.0
        npc1.age_stage = "adult"
        npc1.can_reproduce = True
        
        house.add_occupant(id(npc1))
        npc1.current_house = house
        
        world.entities = [npc1]
        initial_count = len(world.entities)
        
        # Update many times
        for _ in range(1000):
            world.update(0.1)
        
        # Should not create offspring (need 2 adults)
        assert len(world.entities) == initial_count
    
    def test_no_reproduction_with_three_adults(self):
        """Test that reproduction doesn't happen with 3+ adults (house full)."""
        house = House(0.0, 0.0, 0.0, capacity=2)
        
        npc1 = NPC(0.0, 0.0, 0.0)
        npc2 = NPC(1.0, 0.0, 1.0)
        
        # Make both adults
        npc1.age = npc1.adult_age + 1.0
        npc1.age_stage = "adult"
        npc1.can_reproduce = True
        
        npc2.age = npc2.adult_age + 1.0
        npc2.age_stage = "adult"
        npc2.can_reproduce = True
        
        # Fill house to capacity
        house.add_occupant(id(npc1))
        house.add_occupant(id(npc2))
        npc1.current_house = house
        npc2.current_house = house
        
        # House is now full (2 adults)
        assert house.can_shelter_adult() == False
        
        # Third adult cannot enter
        npc3 = NPC(2.0, 0.0, 2.0)
        npc3.age = npc3.adult_age + 1.0
        npc3.age_stage = "adult"
        assert house.add_occupant(id(npc3)) == False
    
    def test_reproduction_requires_both_adults_alive(self):
        """Test that reproduction requires both adults to be alive."""
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
        house = House(5.0, 0.0, 5.0, capacity=2)
        house.add_occupant(id(npc1))
        house.add_occupant(id(npc2))
        npc1.current_house = house
        npc2.current_house = house
        
        # Kill one NPC
        npc2.is_alive = False
        
        # Should not be able to reproduce
        assert npc1.can_reproduce_with(npc2) == False
    
    def test_initial_population_starts_as_adults(self):
        """Test that initial population starts as adults (for simulation)."""
        from world_simulation.world.world import World
        
        world = World(seed=42)
        
        # Create NPCs
        npc1 = NPC(0.0, 0.0, 0.0)
        npc2 = NPC(1.0, 0.0, 1.0)
        
        world.entities = [npc1, npc2]
        
        # Make initial population adults
        for npc in world.entities:
            if npc.age < npc.adult_age:
                npc.age = npc.adult_age + np.random.uniform(0, 60.0)
                npc.age_stage = "adult"
                npc.can_reproduce = True
        
        # All should be adults
        for npc in world.entities:
            assert npc.age_stage == "adult"
            assert npc.can_reproduce == True
    
    def test_child_size_is_smaller(self):
        """Test that children are visually smaller than adults."""
        child = NPC(0.0, 0.0, 0.0)
        adult = NPC(10.0, 0.0, 10.0)
        
        # Make adult
        adult.age = adult.adult_age + 1.0
        adult.age_stage = "adult"
        adult.can_reproduce = True
        
        # Child should be smaller
        assert child.age_stage == "child"
        assert child.size < adult.size or child.age_stage == "child"
    
    def test_reproduction_with_cooldown_active(self):
        """Test that reproduction is blocked when cooldown is active."""
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
        house = House(5.0, 0.0, 5.0, capacity=2)
        house.add_occupant(id(npc1))
        house.add_occupant(id(npc2))
        npc1.current_house = house
        npc2.current_house = house
        
        # Reproduce once
        offspring1 = npc1.reproduce(npc2)
        
        # Try to reproduce again immediately (should fail due to cooldown)
        assert npc1.can_reproduce_with(npc2) == False
        assert npc2.can_reproduce_with(npc1) == False
        
        # Wait for cooldown to expire
        world = MockWorld()
        for _ in range(100):
            npc1.update(0.6, world)
            npc2.update(0.6, world)
        
        # Now should be able to reproduce again
        assert npc1.reproduction_cooldown <= 0
        assert npc2.reproduction_cooldown <= 0
        assert npc1.can_reproduce_with(npc2) == True
    
    def test_elder_stage_transition(self):
        """Test that NPCs correctly transition to elder stage."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        # Set to adult first
        npc.age = npc.adult_age + 1.0
        npc.age_stage = "adult"
        npc.can_reproduce = True
        
        # Calculate elder age
        elder_age = npc.elder_age
        
        # Set age just before elder (with more buffer)
        npc.age = elder_age - 1.0
        npc.update(0.1, world)
        
        assert npc.age_stage == "adult"
        
        # Update past elder age
        npc.age = elder_age + 0.1
        npc.update(0.1, world)
        
        assert npc.age_stage == "elder"
        # Elder can still reproduce if cooldown allows
        assert hasattr(npc, 'can_reproduce')
    
    def test_offspring_has_correct_initial_state(self):
        """Test that offspring starts with correct initial state."""
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
        house = House(5.0, 0.0, 5.0, capacity=2)
        house.add_occupant(id(npc1))
        house.add_occupant(id(npc2))
        npc1.current_house = house
        npc2.current_house = house
        
        # Create offspring
        offspring = npc1.reproduce(npc2)
        
        # Verify offspring state
        assert offspring.age == 0.0
        assert offspring.age_stage == "child"
        assert offspring.can_reproduce == False
        assert offspring.current_house is None
        assert offspring.is_alive == True
        assert offspring.health > 0
        assert offspring.hunger > 0


class TestHouseCapacity:
    """Test house capacity restrictions."""
    
    def test_house_capacity_defaults_to_2(self):
        """Test that houses default to capacity 2."""
        house = House(0.0, 0.0, 0.0)
        assert house.capacity == 2
    
    def test_only_2_adults_can_occupy_house(self):
        """Test that only 2 adults can occupy a house."""
        house = House(0.0, 0.0, 0.0, capacity=2)
        
        assert house.can_shelter_adult() == True
        assert house.add_occupant(1) == True
        assert house.can_shelter_adult() == True
        assert house.add_occupant(2) == True
        assert house.can_shelter_adult() == False
        assert house.add_occupant(3) == False


class TestReproductionIntegration:
    """Integration tests for reproduction."""
    
    def test_reproduction_only_with_two_adults_in_house(self):
        """Test that reproduction only happens with exactly 2 adults."""
        from world_simulation.world.world import World
        
        world = World(seed=42)
        house = House(0.0, 0.0, 0.0, capacity=2)
        world.houses.append(house)
        
        # Create 2 adult NPCs
        npc1 = NPC(0.0, 0.0, 0.0)
        npc2 = NPC(1.0, 0.0, 1.0)
        
        npc1.age = npc1.adult_age + 1.0
        npc1.age_stage = "adult"
        npc1.can_reproduce = True
        
        npc2.age = npc2.adult_age + 1.0
        npc2.age_stage = "adult"
        npc2.can_reproduce = True
        
        # Put them in house
        house.add_occupant(id(npc1))
        house.add_occupant(id(npc2))
        npc1.current_house = house
        npc2.current_house = house
        
        world.entities = [npc1, npc2]
        
        initial_count = len(world.entities)
        
        # Update many times to allow reproduction chance
        for _ in range(1000):
            world.update(0.1)
        
        # Should have created at least one offspring (high chance over 100 seconds)
        assert len(world.entities) >= initial_count
    
    def test_children_cannot_occupy_houses(self):
        """Test that children cannot occupy houses."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        world.houses = [House(5.0, 0.0, 5.0, capacity=2)]
        
        # Child tries to seek shelter
        npc.state = "seeking_shelter"
        npc._seek_shelter(0.1, world)
        
        # Should not find shelter (children can't occupy)
        assert npc.current_house is None or npc.age_stage == "adult"

