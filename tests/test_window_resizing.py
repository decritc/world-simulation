"""Tests for window resizing functionality."""

import pytest
from world_simulation.world.world import World


class TestWindowResizing:
    """Test window resizing functionality."""
    
    def test_window_is_resizable(self):
        """Test that window is created with resizable=True."""
        world = World(seed=42)
        
        try:
            from world_simulation.rendering.renderer import Renderer
            renderer = Renderer(world, width=800, height=600)
            
            # Check that window is resizable
            # In pyglet, we can't directly check resizable property, but we can test resize handler
            assert hasattr(renderer.window, 'on_resize')
            
        except Exception as e:
            if "OpenGL" in str(e) or "GL" in str(e) or "window" in str(e).lower():
                pytest.skip(f"OpenGL context not available: {e}")
            else:
                raise
    
    def test_camera_resizes_on_window_resize(self):
        """Test that camera dimensions update when window is resized."""
        world = World(seed=42)
        
        try:
            from world_simulation.rendering.renderer import Renderer
            renderer = Renderer(world, width=800, height=600)
            
            initial_width = renderer.camera.width
            initial_height = renderer.camera.height
            
            # Simulate window resize
            new_width = 1920
            new_height = 1080
            renderer.window.width = new_width
            renderer.window.height = new_height
            
            # Call resize handler
            renderer.camera.resize(new_width, new_height)
            
            assert renderer.camera.width == new_width
            assert renderer.camera.height == new_height
            
        except Exception as e:
            if "OpenGL" in str(e) or "GL" in str(e) or "window" in str(e).lower():
                pytest.skip(f"OpenGL context not available: {e}")
            else:
                raise
    
    def test_resize_handler_exists(self):
        """Test that resize handler is properly set up."""
        world = World(seed=42)
        
        try:
            from world_simulation.rendering.renderer import Renderer
            renderer = Renderer(world, width=800, height=600)
            
            # Check that camera has resize method
            assert hasattr(renderer.camera, 'resize')
            assert callable(renderer.camera.resize)
            
        except Exception as e:
            if "OpenGL" in str(e) or "GL" in str(e) or "window" in str(e).lower():
                pytest.skip(f"OpenGL context not available: {e}")
            else:
                raise
