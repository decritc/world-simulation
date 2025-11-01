"""Tests for animal functionality."""

import pytest
import numpy as np
from world_simulation.entities.animal import Animal
from world_simulation.entities.npc import NPC
from world_simulation.world.world import World


class TestAnimalCreation:
    """Test animal creation and initialization."""
    
    def test_animal_creation(self):
        """Test creating an animal."""
        animal = Animal(10.0, 5.0, 20.0, "deer")
        assert animal.x == 10.0
        assert animal.y == 5.0
        assert animal.z == 20.0
        assert animal.species == "deer"
        assert animal.is_alive
        assert animal.health > 0
        assert animal.max_health > 0
    
    def test_animal_species_stats(self):
        """Test that different species have different stats."""
        deer = Animal(0, 0, 0, "deer")
        rabbit = Animal(0, 0, 0, "rabbit")
        boar = Animal(0, 0, 0, "boar")
        
        assert deer.speed > 0
        assert rabbit.speed > 0
        assert boar.speed > 0
        
        # Deer should have more health than rabbit
        assert deer.max_health > rabbit.max_health
        
        # Boar should have more health than deer
        assert boar.max_health > deer.max_health
        
        # Rabbit should be faster than deer
        assert rabbit.speed > deer.speed
    
    def test_animal_take_damage(self):
        """Test animal taking damage."""
        animal = Animal(0, 0, 0, "deer")
        initial_health = animal.health
        
        animal.take_damage(10.0)
        assert animal.health < initial_health
        
        # Deal enough damage to kill
        animal.take_damage(animal.max_health)
        assert not animal.is_alive
        assert animal.health <= 0


class TestAnimalBehavior:
    """Test animal behavior."""
    
    def test_animal_wandering(self):
        """Test animal wandering behavior."""
        world = World(seed=42)
        animal = Animal(0, 0, 0, "deer")
        world.animals.append(animal)
        
        initial_x = animal.x
        initial_z = animal.z
        
        # Update animal multiple times (should wander)
        for _ in range(10):
            animal.update(0.1, world)
        
        # Animal should have moved or have a target set
        assert animal.target_x is not None or animal.x != initial_x or animal.z != initial_z
    
    def test_animal_fleeing_from_npc(self):
        """Test animal fleeing from nearby NPCs."""
        world = World(seed=42)
        animal = Animal(0, 0, 0, "deer")
        npc = NPC(5.0, 0, 0)  # Close to animal
        world.animals.append(animal)
        world.entities.append(npc)
        
        initial_x = animal.x
        
        # Update animal (should flee)
        animal.update(0.1, world)
        
        # Animal should be fleeing
        assert animal.is_fleeing
        assert animal.flee_target_x is not None
    
    def test_animal_reproduction(self):
        """Test animal reproduction."""
        world = World(seed=42)
        animal = Animal(0, 0, 0, "deer")
        animal.reproduction_cooldown = 0.0  # Ready to reproduce
        world.animals.append(animal)
        
        initial_count = len(world.animals)
        
        # Try to reproduce
        new_animal = animal.reproduce(world)
        
        if new_animal:
            assert new_animal.species == animal.species
            assert new_animal.is_alive
            assert animal.reproduction_cooldown > 0  # Cooldown set
    
    def test_animal_cannot_reproduce_on_cooldown(self):
        """Test animal cannot reproduce while on cooldown."""
        animal = Animal(0, 0, 0, "deer")
        animal.reproduction_cooldown = 10.0  # On cooldown
        
        assert not animal.can_reproduce()


class TestAnimalHunting:
    """Test NPC hunting animals."""
    
    def test_npc_targets_animal_when_hungry(self):
        """Test NPC targets animal when hungry."""
        world = World(seed=42)
        npc = NPC(0, 0, 0)
        npc.hunger = 30.0  # Very hungry
        animal = Animal(5.0, 0, 0, "deer")
        world.entities.append(npc)
        world.animals.append(animal)
        
        # NPC should seek food (which should include animals)
        npc._seek_food(0.1, world)
        
        # NPC should target the animal
        assert npc.target_animal == animal
        assert npc.state == "hunting"
    
    def test_npc_hunts_animal(self):
        """Test NPC hunting and killing an animal."""
        world = World(seed=42)
        npc = NPC(0, 0, 0)
        animal = Animal(1.0, 0, 0, "deer")  # Very close
        npc.target_animal = animal
        npc.state = "hunting"
        world.entities.append(npc)
        world.animals.append(animal)
        
        initial_health = animal.health
        
        # Hunt the animal
        npc._hunt_animal(0.5, world)  # 0.5 seconds should deal significant damage
        
        # Animal should take damage
        assert animal.health < initial_health or not animal.is_alive
    
    def test_npc_eats_animal_after_kill(self):
        """Test NPC eating animal after killing it."""
        world = World(seed=42)
        npc = NPC(0, 0, 0)
        npc.hunger = 50.0
        animal = Animal(0, 0, 0, "deer")
        animal.health = 1.0  # Low health
        npc.target_animal = animal
        npc.state = "hunting"
        world.entities.append(npc)
        world.animals.append(animal)
        
        initial_hunger = npc.hunger
        
        # Kill and eat animal
        npc._hunt_animal(1.0, world)
        
        # NPC hunger should increase
        assert npc.hunger > initial_hunger
        assert npc.animals_hunted > 0


class TestAnimalWorldIntegration:
    """Test animal integration with world."""
    
    def test_world_initializes_animals(self):
        """Test world initializes animals on creation."""
        world = World(seed=42)
        assert len(world.animals) > 0
        
        # All animals should be alive
        for animal in world.animals:
            assert animal.is_alive
    
    def test_world_updates_animals(self):
        """Test world updates animals."""
        world = World(seed=42)
        initial_positions = [(a.x, a.z) for a in world.animals]
        
        # Update world
        world.update(0.1)
        
        # Animals should have moved or updated
        for i, animal in enumerate(world.animals):
            if animal.is_alive:
                assert (animal.x, animal.z) != initial_positions[i] or animal.target_x is not None
    
    def test_dead_animals_removed_from_world(self):
        """Test dead animals are removed from world."""
        world = World(seed=42)
        animal = world.animals[0]
        animal.is_alive = False
        
        world.update(0.1)
        
        # Dead animal should be removed
        assert animal not in world.animals
