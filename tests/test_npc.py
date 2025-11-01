"""Tests for NPC entities."""

import pytest
import numpy as np
from world_simulation.entities.npc import NPC
from world_simulation.trees.tree import FruitTree
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


class TestNPC:
    """Test NPC functionality."""
    
    def test_init(self):
        """Test NPC initialization."""
        npc = NPC(0.0, 0.0, 0.0)
        assert npc.x == 0.0
        assert npc.y == 0.0
        assert npc.z == 0.0
        assert npc.is_alive == True
        assert npc.health == 100.0
        assert npc.hunger == 50.0
        assert npc.stamina > 0
        assert 'speed' in npc.genome
        assert 'vision_range' in npc.genome
    
    def test_init_with_genome(self):
        """Test NPC initialization with custom genome."""
        genome = {'speed': 2.0, 'size': 1.5, 'stamina': 150.0}
        npc = NPC(0.0, 0.0, 0.0, genome=genome)
        assert npc.speed == 2.0
        assert npc.size == 1.5
        assert npc.stamina == 150.0
    
    def test_hunger_decreases(self):
        """Test that hunger decreases over time."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        initial_hunger = npc.hunger
        
        npc.update(1.0, world)
        
        assert npc.hunger < initial_hunger
    
    def test_health_decreases_over_time(self):
        """Test that health decreases gradually over time."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        
        initial_health = npc.health
        assert initial_health == 100.0
        
        # Update NPC multiple times
        for _ in range(100):
            npc.update(0.1, world)
        
        # Health should have decreased
        assert npc.health < initial_health
        assert npc.health > 0  # Should still be alive
        
    def test_health_decreases_faster_when_hungry(self):
        """Test that health decreases faster when hunger is low."""
        npc1 = NPC(0.0, 0.0, 0.0)
        npc2 = NPC(10.0, 0.0, 10.0)
        world = MockWorld()
        
        # NPC1: Well fed (high hunger)
        npc1.hunger = 90.0
        
        # NPC2: Hungry (low hunger)
        npc2.hunger = 10.0
        
        initial_health1 = npc1.health
        initial_health2 = npc2.health
        
        # Update both NPCs
        for _ in range(100):
            npc1.update(0.1, world)
            npc2.update(0.1, world)
        
        # Both should lose health, but NPC2 (hungry) should lose more
        health_loss1 = initial_health1 - npc1.health
        health_loss2 = initial_health2 - npc2.health
        
        assert health_loss2 > health_loss1  # Hungry NPC loses health faster
    
    def test_starvation_damage(self):
        """Test that NPCs take damage when starving."""
        npc = NPC(0.0, 0.0, 0.0)
        npc.hunger = 0.0
        world = MockWorld()
        initial_health = npc.health
        
        npc.update(1.0, world)
        
        assert npc.health < initial_health
    
    def test_death_when_health_zero(self):
        """Test that NPCs die when health reaches zero."""
        npc = NPC(0.0, 0.0, 0.0)
        npc.health = 1.0
        npc.hunger = 0.0
        world = MockWorld()
        
        npc.update(1.0, world)
        
        assert npc.is_alive == False
        assert npc.health <= 0
    
    def test_wandering(self):
        """Test wandering behavior."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        initial_x = npc.x
        initial_z = npc.z
        
        # Update multiple times to allow movement
        for _ in range(10):
            npc.update(0.1, world)
        
        # NPC should have moved or changed target
        assert npc.target_x is not None or npc.x != initial_x or npc.z != initial_z
    
    def test_seek_food(self):
        """Test food seeking behavior."""
        npc = NPC(0.0, 0.0, 0.0)
        npc.hunger = 30.0  # Low hunger
        world = MockWorld()
        
        # Add a tree with fruit
        tree = FruitTree(5.0, 0.0, 5.0)
        # Force fruit to be ripe
        tree.fruit_maturity[0] = 1.0
        tree.fruit_count = 1
        world.trees.append(tree)
        
        # Update until NPC seeks food
        for _ in range(20):
            npc.update(0.1, world)
            if npc.state == "seeking_food":
                break
        
        assert npc.state == "seeking_food" or npc.state == "eating"
    
    def test_eat_fruit(self):
        """Test eating fruit from trees."""
        npc = NPC(0.0, 0.0, 0.0)
        npc.hunger = 30.0
        world = MockWorld()
        
        tree = FruitTree(1.0, 0.0, 1.0)
        tree.fruit_maturity[0] = 1.0
        tree.fruit_count = 1
        world.trees.append(tree)
        
        # Move NPC near tree
        npc.x = 1.0
        npc.z = 1.0
        npc.state = "eating"
        npc.target_tree = tree
        
        initial_hunger = npc.hunger
        initial_fruit = tree.get_ripe_fruit_count()
        
        # Update multiple times to allow eating
        for _ in range(10):
            npc.update(0.1, world)
        
        # Hunger should increase or fruit should decrease
        assert npc.hunger > initial_hunger or tree.get_ripe_fruit_count() < initial_fruit
    
    def test_seek_shelter_at_night(self):
        """Test that NPCs seek shelter at night."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        world.day_time = world.day_length * (20.0 / 24.0)  # 8pm - night
        
        house = House(5.0, 0.0, 5.0)
        world.houses.append(house)
        
        # Update until NPC seeks shelter
        for _ in range(20):
            npc.update(0.1, world)
            if npc.state == "seeking_shelter":
                break
        
        assert npc.state == "seeking_shelter" or npc.state == "in_shelter"
    
    def test_leave_shelter_at_dawn(self):
        """Test that NPCs leave shelter at dawn."""
        npc = NPC(0.0, 0.0, 0.0)
        world = MockWorld()
        house = House(0.0, 0.0, 0.0)
        
        # Put NPC in shelter
        house.add_occupant(id(npc))
        npc.current_house = house
        npc.state = "in_shelter"
        
        # Advance to day
        world.day_time = world.day_length * (8.0 / 24.0)  # 8am - day
        
        npc.update(0.1, world)
        
        assert npc.state != "in_shelter"
        assert npc.current_house is None
    
    def test_fitness_calculation(self):
        """Test fitness score calculation."""
        npc = NPC(0.0, 0.0, 0.0)
        npc.age = 10.0
        npc.fruit_collected = 5
        npc.health = 80.0
        
        fitness = npc.get_fitness()
        
        assert fitness > 0
        assert isinstance(fitness, (int, float))

