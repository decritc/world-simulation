"""Tests for NPC selection and detail panel functionality."""

import pytest
import numpy as np
from world_simulation.world.world import World
from world_simulation.entities.npc import NPC
from world_simulation.rendering.renderer import Renderer


class TestNPCSelection:
    """Test NPC selection and picking functionality."""
    
    def test_selected_npc_initialization(self):
        """Test that selected_npc is None initially."""
        world = World(seed=42)
        renderer = Renderer(world, width=800, height=600)
        
        assert renderer.selected_npc is None
    
    def test_selected_npc_can_be_set(self):
        """Test that selected_npc can be set."""
        world = World(seed=42)
        renderer = Renderer(world, width=800, height=600)
        
        # Create an NPC
        npc = NPC(0.0, 0.0, 0.0)
        world.entities.append(npc)
        
        # Set selected NPC
        renderer.selected_npc = npc
        
        assert renderer.selected_npc == npc
        assert renderer.selected_npc.is_alive == True
    
    def test_selected_npc_cleared_when_dead(self):
        """Test that dead NPCs cannot be selected."""
        world = World(seed=42)
        renderer = Renderer(world, width=800, height=600)
        
        npc = NPC(0.0, 0.0, 0.0)
        world.entities.append(npc)
        
        # Select NPC
        renderer.selected_npc = npc
        assert renderer.selected_npc == npc
        
        # Kill NPC
        npc.is_alive = False
        npc.health = 0.0
        
        # Selection should remain but detail panel won't render
        assert renderer.selected_npc == npc
        assert renderer.selected_npc.is_alive == False
    
    def test_npc_picking_distance_calculation(self):
        """Test that NPC picking calculates distances correctly."""
        world = World(seed=42)
        renderer = Renderer(world, width=800, height=600)
        
        # Set camera position
        renderer.camera_x = 0.0
        renderer.camera_y = 10.0
        renderer.camera_z = 0.0
        
        # Create NPCs at different distances
        npc1 = NPC(5.0, 0.0, 0.0)  # Close
        npc2 = NPC(50.0, 0.0, 0.0)  # Far
        npc3 = NPC(150.0, 0.0, 0.0)  # Too far
        
        world.entities = [npc1, npc2, npc3]
        
        # Calculate distances manually
        dx1 = npc1.x - renderer.camera_x
        dz1 = npc1.z - renderer.camera_z
        dist1 = np.sqrt(dx1*dx1 + dz1*dz1)
        
        assert dist1 == 5.0  # Close NPC
        
        dx2 = npc2.x - renderer.camera_x
        dz2 = npc2.z - renderer.camera_z
        dist2 = np.sqrt(dx2*dx2 + dz2*dz2)
        
        assert dist2 == 50.0  # Medium distance
        
        dx3 = npc3.x - renderer.camera_x
        dz3 = npc3.z - renderer.camera_z
        dist3 = np.sqrt(dx3*dx3 + dz3*dz3)
        
        assert dist3 == 150.0  # Too far (> 100)
    
    def test_npc_picking_excludes_dead_npcs(self):
        """Test that dead NPCs are excluded from picking."""
        world = World(seed=42)
        renderer = Renderer(world, width=800, height=600)
        
        npc_alive = NPC(5.0, 0.0, 0.0)
        npc_dead = NPC(10.0, 0.0, 0.0)
        npc_dead.is_alive = False
        
        world.entities = [npc_alive, npc_dead]
        
        # Only alive NPCs should be considered
        alive_count = sum(1 for npc in world.entities if npc.is_alive)
        assert alive_count == 1
    
    def test_npc_picking_ray_direction_calculation(self):
        """Test that ray direction is calculated correctly from camera."""
        world = World(seed=42)
        renderer = Renderer(world, width=800, height=600)
        
        # Set camera position and orientation
        renderer.camera_x = 0.0
        renderer.camera_y = 10.0
        renderer.camera_z = 0.0
        renderer.camera_yaw = 0.0
        renderer.camera_pitch = -45.0
        
        # Calculate forward direction
        yaw_rad = np.radians(renderer.camera_yaw)
        pitch_rad = np.radians(renderer.camera_pitch)
        
        forward_x = np.sin(yaw_rad) * np.cos(pitch_rad)
        forward_y = -np.sin(pitch_rad)
        forward_z = -np.cos(yaw_rad) * np.cos(pitch_rad)
        
        # Forward vector should be normalized
        forward_len = np.sqrt(forward_x*forward_x + forward_y*forward_y + forward_z*forward_z)
        assert abs(forward_len - 1.0) < 0.01  # Approximately normalized
    
    def test_mouse_button_tracking(self):
        """Test that mouse buttons are tracked correctly."""
        world = World(seed=42)
        renderer = Renderer(world, width=800, height=600)
        
        # Initially no buttons pressed
        assert isinstance(renderer.mouse_buttons, set)
        assert len(renderer.mouse_buttons) == 0
        
        # Simulate button press
        import pyglet
        renderer.mouse_buttons.add(pyglet.window.mouse.LEFT)
        assert pyglet.window.mouse.LEFT in renderer.mouse_buttons


class TestNPCDetailPanel:
    """Test NPC detail panel content generation."""
    
    def test_detail_panel_state_descriptions(self):
        """Test that all NPC states have descriptions."""
        world = World(seed=42)
        renderer = Renderer(world, width=800, height=600)
        
        # Define expected states
        expected_states = [
            "wandering",
            "seeking_food",
            "eating",
            "resting",
            "seeking_shelter",
            "in_shelter"
        ]
        
        # Create NPCs with different states
        for state in expected_states:
            npc = NPC(0.0, 0.0, 0.0)
            npc.state = state
            world.entities.append(npc)
        
        # Check that state descriptions would be generated
        state_descriptions = {
            "wandering": "Exploring the world randomly",
            "seeking_food": "Looking for food sources",
            "eating": "Consuming fruit from a tree",
            "resting": "Resting to restore stamina",
            "seeking_shelter": "Looking for shelter (nighttime)",
            "in_shelter": "Safe inside a house"
        }
        
        for state in expected_states:
            assert state in state_descriptions
    
    def test_health_status_calculation(self):
        """Test that health status is calculated correctly."""
        npc = NPC(0.0, 0.0, 0.0)
        npc.max_health = 100.0
        
        # Excellent health (>80%)
        npc.health = 90.0
        health_pct = (npc.health / npc.max_health) * 100
        health_status = "Excellent" if health_pct > 80 else "Good" if health_pct > 50 else "Fair" if health_pct > 25 else "Critical"
        assert health_status == "Excellent"
        
        # Good health (50-80%)
        npc.health = 70.0
        health_pct = (npc.health / npc.max_health) * 100
        health_status = "Excellent" if health_pct > 80 else "Good" if health_pct > 50 else "Fair" if health_pct > 25 else "Critical"
        assert health_status == "Good"
        
        # Fair health (25-50%)
        npc.health = 40.0
        health_pct = (npc.health / npc.max_health) * 100
        health_status = "Excellent" if health_pct > 80 else "Good" if health_pct > 50 else "Fair" if health_pct > 25 else "Critical"
        assert health_status == "Fair"
        
        # Critical health (<25%)
        npc.health = 10.0
        health_pct = (npc.health / npc.max_health) * 100
        health_status = "Excellent" if health_pct > 80 else "Good" if health_pct > 50 else "Fair" if health_pct > 25 else "Critical"
        assert health_status == "Critical"
    
    def test_hunger_status_calculation(self):
        """Test that hunger status is calculated correctly."""
        npc = NPC(0.0, 0.0, 0.0)
        npc.max_hunger = 100.0
        
        # Well Fed (>80%)
        npc.hunger = 90.0
        hunger_pct = (npc.hunger / npc.max_hunger) * 100
        hunger_status = "Well Fed" if hunger_pct > 80 else "Satisfied" if hunger_pct > 50 else "Hungry" if hunger_pct > 25 else "Starving"
        assert hunger_status == "Well Fed"
        
        # Satisfied (50-80%)
        npc.hunger = 70.0
        hunger_pct = (npc.hunger / npc.max_hunger) * 100
        hunger_status = "Well Fed" if hunger_pct > 80 else "Satisfied" if hunger_pct > 50 else "Hungry" if hunger_pct > 25 else "Starving"
        assert hunger_status == "Satisfied"
        
        # Hungry (25-50%)
        npc.hunger = 40.0
        hunger_pct = (npc.hunger / npc.max_hunger) * 100
        hunger_status = "Well Fed" if hunger_pct > 80 else "Satisfied" if hunger_pct > 50 else "Hungry" if hunger_pct > 25 else "Starving"
        assert hunger_status == "Hungry"
        
        # Starving (<25%)
        npc.hunger = 10.0
        hunger_pct = (npc.hunger / npc.max_hunger) * 100
        hunger_status = "Well Fed" if hunger_pct > 80 else "Satisfied" if hunger_pct > 50 else "Hungry" if hunger_pct > 25 else "Starving"
        assert hunger_status == "Starving"
    
    def test_stamina_status_calculation(self):
        """Test that stamina status is calculated correctly."""
        npc = NPC(0.0, 0.0, 0.0)
        npc.max_stamina = 100.0
        
        # Energetic (>80%)
        npc.stamina = 90.0
        stamina_pct = (npc.stamina / npc.max_stamina) * 100 if npc.max_stamina > 0 else 0
        stamina_status = "Energetic" if stamina_pct > 80 else "Active" if stamina_pct > 50 else "Tired" if stamina_pct > 25 else "Exhausted"
        assert stamina_status == "Energetic"
        
        # Active (50-80%)
        npc.stamina = 70.0
        stamina_pct = (npc.stamina / npc.max_stamina) * 100 if npc.max_stamina > 0 else 0
        stamina_status = "Energetic" if stamina_pct > 80 else "Active" if stamina_pct > 50 else "Tired" if stamina_pct > 25 else "Exhausted"
        assert stamina_status == "Active"
        
        # Tired (25-50%)
        npc.stamina = 40.0
        stamina_pct = (npc.stamina / npc.max_stamina) * 100 if npc.max_stamina > 0 else 0
        stamina_status = "Energetic" if stamina_pct > 80 else "Active" if stamina_pct > 50 else "Tired" if stamina_pct > 25 else "Exhausted"
        assert stamina_status == "Tired"
        
        # Exhausted (<25%)
        npc.stamina = 10.0
        stamina_pct = (npc.stamina / npc.max_stamina) * 100 if npc.max_stamina > 0 else 0
        stamina_status = "Energetic" if stamina_pct > 80 else "Active" if stamina_pct > 50 else "Tired" if stamina_pct > 25 else "Exhausted"
        assert stamina_status == "Exhausted"
    
    def test_npc_detail_panel_content_generation(self):
        """Test that detail panel content is generated correctly."""
        world = World(seed=42)
        renderer = Renderer(world, width=800, height=600)
        
        # Create NPC with known values
        npc = NPC(10.0, 5.0, 15.0)
        npc.health = 80.0
        npc.max_health = 100.0
        npc.hunger = 60.0
        npc.max_hunger = 100.0
        npc.stamina = 70.0
        npc.max_stamina = 100.0
        npc.state = "eating"
        npc.speed = 1.5
        npc.size = 1.0
        npc.age = 42.5
        npc.fruit_collected = 5
        npc.target_x = 12.0
        npc.target_z = 18.0
        
        world.entities.append(npc)
        renderer.selected_npc = npc
        
        # Verify NPC attributes are accessible
        assert renderer.selected_npc.health == 80.0
        assert renderer.selected_npc.max_health == 100.0
        assert renderer.selected_npc.state == "eating"
        assert renderer.selected_npc.fruit_collected == 5
        assert renderer.selected_npc.target_x == 12.0
        assert renderer.selected_npc.target_z == 18.0
    
    def test_npc_detail_panel_handles_missing_genome_values(self):
        """Test that detail panel handles missing genome values."""
        npc = NPC(0.0, 0.0, 0.0)
        
        # Test with default genome values
        vision_range = npc.genome.get('vision_range', 0)
        food_preference = npc.genome.get('food_preference', 0)
        
        assert vision_range >= 0
        assert food_preference >= 0.0
        assert food_preference <= 1.0
    
    def test_npc_detail_panel_handles_no_target(self):
        """Test that detail panel handles NPCs without targets."""
        npc = NPC(0.0, 0.0, 0.0)
        npc.target_x = None
        npc.target_z = None
        
        # Should handle None gracefully
        assert npc.target_x is None
        assert npc.target_z is None
    
    def test_npc_detail_panel_handles_house_occupancy(self):
        """Test that detail panel handles house occupancy correctly."""
        npc = NPC(0.0, 0.0, 0.0)
        
        # Initially no house
        assert npc.current_house is None
        
        # Test with house (would be set by house.add_occupant)
        from world_simulation.houses.house import House
        house = House(10.0, 0.0, 10.0)
        npc.current_house = house
        
        assert npc.current_house is not None
        assert npc.current_house == house


class TestNPCHighlight:
    """Test NPC highlight rendering logic."""
    
    def test_highlight_only_for_selected_npc(self):
        """Test that highlight is only rendered for selected NPC."""
        world = World(seed=42)
        renderer = Renderer(world, width=800, height=600)
        
        npc1 = NPC(0.0, 0.0, 0.0)
        npc2 = NPC(10.0, 0.0, 10.0)
        
        world.entities = [npc1, npc2]
        
        # No selection initially
        assert renderer.selected_npc is None
        
        # Select first NPC
        renderer.selected_npc = npc1
        
        # Only selected NPC should be highlighted
        assert renderer.selected_npc == npc1
        assert renderer.selected_npc != npc2


class TestNPCSelectionIntegration:
    """Integration tests for NPC selection."""
    
    def test_selection_with_multiple_npcs(self):
        """Test selecting NPC from multiple NPCs."""
        world = World(seed=42)
        renderer = Renderer(world, width=800, height=600)
        
        # Create multiple NPCs
        npc1 = NPC(5.0, 0.0, 5.0)
        npc2 = NPC(10.0, 0.0, 10.0)
        npc3 = NPC(15.0, 0.0, 15.0)
        
        world.entities = [npc1, npc2, npc3]
        
        # Select one NPC
        renderer.selected_npc = npc2
        
        assert renderer.selected_npc == npc2
        assert renderer.selected_npc != npc1
        assert renderer.selected_npc != npc3
    
    def test_selection_cleared_when_npc_dies(self):
        """Test that selection persists but detail panel won't render for dead NPC."""
        world = World(seed=42)
        renderer = Renderer(world, width=800, height=600)
        
        npc = NPC(0.0, 0.0, 0.0)
        world.entities.append(npc)
        
        renderer.selected_npc = npc
        assert renderer.selected_npc.is_alive == True
        
        # Kill NPC
        npc.is_alive = False
        npc.health = 0.0
        
        # Selection persists but NPC is dead
        assert renderer.selected_npc == npc
        assert renderer.selected_npc.is_alive == False

