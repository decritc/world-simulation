"""Tests for camera terrain collision prevention."""

import pytest
from world_simulation.world.world import World
from world_simulation.rendering.renderer import Renderer


class TestCameraTerrainCollision:
    """Test that camera stays above terrain."""
    
    def test_camera_stays_above_terrain(self):
        """Test that camera is kept above terrain height."""
        world = World(seed=42)
        
        try:
            renderer = Renderer(world, width=800, height=600)
            
            # Move camera to different positions
            test_positions = [
                (0.0, 0.0),
                (10.0, 10.0),
                (-10.0, -10.0),
                (50.0, 50.0),
            ]
            
            for x, z in test_positions:
                renderer.camera.x = x
                renderer.camera.z = z
                renderer.camera.y = 1.0  # Try to set camera low
                
                # Get terrain height at this position
                terrain_height = world.get_height(x, z)
                
                # Update renderer (this should enforce minimum height)
                renderer.update(0.016)  # ~60fps delta time
                
                # Camera should be at least 5 units above terrain
                min_expected_height = terrain_height + 5.0
                assert renderer.camera.y >= min_expected_height, \
                    f"Camera at ({x}, {z}) is below terrain! Y={renderer.camera.y}, terrain={terrain_height}, min={min_expected_height}"
            
        except Exception as e:
            if "OpenGL" in str(e) or "GL" in str(e) or "window" in str(e).lower():
                pytest.skip(f"OpenGL context not available: {e}")
            else:
                raise
    
    def test_camera_minimum_height_on_hills(self):
        """Test that camera adjusts height when moving over hills."""
        world = World(seed=42)
        
        try:
            renderer = Renderer(world, width=800, height=600)
            
            # Set camera to a low position
            renderer.camera.x = 0.0
            renderer.camera.z = 0.0
            renderer.camera.y = 1.0
            
            # Get initial terrain height
            initial_terrain = world.get_height(0.0, 0.0)
            
            # Simulate moving camera horizontally
            renderer.camera.x = 20.0
            renderer.camera.z = 20.0
            
            # Update (should enforce height based on new terrain)
            renderer.update(0.016)
            
            new_terrain = world.get_height(20.0, 20.0)
            min_expected_height = new_terrain + 5.0
            
            assert renderer.camera.y >= min_expected_height, \
                f"Camera didn't adjust for terrain! Y={renderer.camera.y}, terrain={new_terrain}, min={min_expected_height}"
            
        except Exception as e:
            if "OpenGL" in str(e) or "GL" in str(e) or "window" in str(e).lower():
                pytest.skip(f"OpenGL context not available: {e}")
            else:
                raise
    
    def test_camera_can_go_above_terrain(self):
        """Test that camera can go above minimum height if user wants."""
        world = World(seed=42)
        
        try:
            renderer = Renderer(world, width=800, height=600)
            
            terrain_height = world.get_height(0.0, 0.0)
            
            # Set camera high above terrain
            renderer.camera.x = 0.0
            renderer.camera.z = 0.0
            renderer.camera.y = 100.0  # High above terrain
            
            # Update should not lower it
            renderer.update(0.016)
            
            min_expected_height = terrain_height + 5.0
            assert renderer.camera.y >= min_expected_height
            # Camera should stay at 100.0 since it's above minimum
            assert renderer.camera.y == 100.0
            
        except Exception as e:
            if "OpenGL" in str(e) or "GL" in str(e) or "window" in str(e).lower():
                pytest.skip(f"OpenGL context not available: {e}")
            else:
                raise
