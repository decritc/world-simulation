"""Tests for the historian system."""

import pytest
import os
from world_simulation.history.historian import Historian
from world_simulation.entities.npc import NPC
from world_simulation.world.world import World


class TestHistorian:
    """Tests for the Historian class."""

    def test_historian_initialization(self):
        """Test that historian initializes correctly."""
        log_file = "test_colony_history.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        historian = Historian(log_file=log_file)
        assert historian.log_file == log_file
        assert historian.generation_counter == 0
        assert len(historian.npc_generations) == 0
        assert len(historian.npc_parents) == 0
        
        # Cleanup
        if os.path.exists(log_file):
            os.remove(log_file)

    def test_register_npc_birth_initial_population(self):
        """Test registering birth of initial population (generation 0)."""
        log_file = "test_colony_history.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        historian = Historian(log_file=log_file)
        historian.register_npc_birth(1, "Test NPC", None, None, 0.0, 0)
        
        assert historian.get_generation(1) == 0
        assert historian.npc_names[1] == "Test NPC"
        
        # Check log file was created and contains entry
        assert os.path.exists(log_file)
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "BIRTH" in content
            assert "Test NPC" in content
            assert "Generation 0" in content
        
        # Cleanup
        if os.path.exists(log_file):
            os.remove(log_file)

    def test_register_npc_birth_with_parents(self):
        """Test registering birth with parents."""
        log_file = "test_colony_history.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        historian = Historian(log_file=log_file)
        
        # Register parents
        historian.register_npc_birth(1, "Parent 1", None, None, 0.0, 0)
        historian.register_npc_birth(2, "Parent 2", None, None, 0.0, 0)
        
        # Register child
        historian.register_npc_birth(3, "Child", 1, 2, 100.0, 1)
        
        assert historian.get_generation(3) == 1
        assert historian.get_parents(3) == (1, 2)
        assert len(historian.npc_children[1]) == 1
        assert len(historian.npc_children[2]) == 1
        
        # Cleanup
        if os.path.exists(log_file):
            os.remove(log_file)

    def test_register_npc_death(self):
        """Test registering NPC death."""
        log_file = "test_colony_history.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        historian = Historian(log_file=log_file)
        historian.register_npc_birth(1, "Test NPC", None, None, 0.0, 0)
        historian.register_npc_death(1, "Test NPC", 500.0, "old age", 500.0, 5, 10, 2)
        
        # Check log file contains death entry
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "DEATH" in content
            assert "old age" in content
            assert "10" in content  # Fruit collected
            assert "2" in content  # Animals hunted
        
        # Cleanup
        if os.path.exists(log_file):
            os.remove(log_file)

    def test_register_reproduction(self):
        """Test registering reproduction event."""
        log_file = "test_colony_history.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        historian = Historian(log_file=log_file)
        historian.register_reproduction(1, "Parent 1", 2, "Parent 2", 3, "Offspring", 200.0, 2)
        
        # Check log file contains reproduction entry
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "REPRODUCTION" in content
            assert "Parent 1" in content
            assert "Parent 2" in content
            assert "Offspring" in content
        
        # Cleanup
        if os.path.exists(log_file):
            os.remove(log_file)

    def test_register_milestone(self):
        """Test registering milestone event."""
        log_file = "test_colony_history.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        historian = Historian(log_file=log_file)
        historian.register_npc_birth(1, "Test NPC", None, None, 0.0, 0)
        historian.register_milestone(1, "Test NPC", "reached_adult", 100.0, 1, "Age: 100.0 seconds")
        
        # Check log file contains milestone entry
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "MILESTONE" in content
            assert "reached_adult" in content
        
        # Cleanup
        if os.path.exists(log_file):
            os.remove(log_file)

    def test_register_achievement(self):
        """Test registering achievement event."""
        log_file = "test_colony_history.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        historian = Historian(log_file=log_file)
        historian.register_npc_birth(1, "Test NPC", None, None, 0.0, 0)
        historian.register_achievement(1, "Test NPC", "fruit_collection_milestone", 10, 150.0, 2)
        
        # Check log file contains achievement entry
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "ACHIEVEMENT" in content
            assert "fruit_collection_milestone" in content
            assert "10" in content
        
        # Cleanup
        if os.path.exists(log_file):
            os.remove(log_file)

    def test_colony_summary(self):
        """Test generating colony summary."""
        log_file = "test_colony_history.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        historian = Historian(log_file=log_file)
        
        # Create test NPCs
        world = World(seed=42)
        npc1 = NPC(0, 0, 0)
        npc1.name = "NPC 1"
        npc1.age = 100.0
        npc1.fruit_collected = 5
        npc1.animals_hunted = 1
        npc1.is_alive = True
        
        npc2 = NPC(1, 0, 1)
        npc2.name = "NPC 2"
        npc2.age = 200.0
        npc2.fruit_collected = 10
        npc2.animals_hunted = 2
        npc2.is_alive = False
        
        all_npcs = [npc1, npc2]
        alive_npcs = [npc1]
        
        historian.generate_colony_summary(300.0, 3, all_npcs, alive_npcs)
        
        # Check log file contains summary
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "COLONY SUMMARY" in content
            assert "Day 3" in content
        
        # Cleanup
        if os.path.exists(log_file):
            os.remove(log_file)

    def test_npc_name_caching(self):
        """Test that NPC names are cached correctly."""
        log_file = "test_colony_history.txt"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        historian = Historian(log_file=log_file)
        historian.register_npc_birth(1, "Test NPC", None, None, 0.0, 0)
        
        assert historian._get_npc_name_from_id(1) == "Test NPC"
        assert historian._get_npc_name_from_id(999) == "NPC-999"  # Non-existent ID
        
        # Cleanup
        if os.path.exists(log_file):
            os.remove(log_file)

    def test_file_deletion_on_start(self):
        """Test that old log file is deleted on initialization."""
        log_file = "test_colony_history.txt"
        
        # Create old file
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("Old content")
        
        assert os.path.exists(log_file)
        
        # Initialize historian (should delete old file)
        historian = Historian(log_file=log_file)
        
        # File should still exist but content should be new
        assert os.path.exists(log_file)
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Old content" not in content
            assert "COLONY HISTORY LOG" in content
        
        # Cleanup
        if os.path.exists(log_file):
            os.remove(log_file)

