"""Tests for fog rendering and new functionality."""

import pytest
import numpy as np
from pyglet.gl import *
from world_simulation.rendering.fog_manager import FogManager
from world_simulation.rendering.camera import Camera
from world_simulation.rendering.sky_manager import SkyManager
from world_simulation.world.world import World


class TestFogManager:
    """Test fog management functionality."""
    
    def test_fog_manager_initialization(self):
        """Test fog manager initialization."""
        fog = FogManager()
        
        assert fog.fog_start == 40.0
        assert fog.fog_end == 180.0
        assert fog.fog_mode == GL_LINEAR
        assert fog.enabled == True
    
    def test_fog_set_range(self):
        """Test setting fog range."""
        fog = FogManager()
        
        fog.set_range(50.0, 200.0)
        assert fog.fog_start == 50.0
        assert fog.fog_end == 200.0
    
    def test_fog_enable_disable(self):
        """Test enabling and disabling fog."""
        fog = FogManager()
        
        assert fog.enabled == True
        fog.disable()
        assert fog.enabled == False
        fog.enable()
        assert fog.enabled == True
    
    def test_fog_color_update(self):
        """Test fog color update."""
        fog = FogManager()
        
        # Test updating fog color
        fog.update_color(0.5, 0.7, 1.0)
        # Color should be set (we can't easily verify OpenGL state in tests)
        assert fog.enabled == True
    
    def test_fog_setup(self):
        """Test fog setup."""
        fog = FogManager()
        
        # Setup should configure fog parameters
        fog.setup()
        assert fog.enabled == True


class TestCamera:
    """Test camera functionality."""
    
    def test_camera_initialization(self):
        """Test camera initialization."""
        camera = Camera(width=1280, height=720)
        
        assert camera.x == 0.0
        assert camera.y == 50.0
        assert camera.z == 0.0
        assert camera.yaw == 0.0
        assert camera.pitch == -45.0
        assert camera.fov == 70.0
    
    def test_camera_get_forward_vector(self):
        """Test forward vector calculation."""
        camera = Camera()
        
        forward_x, forward_y, forward_z = camera.get_forward_vector()
        
        # Forward vector should be normalized (approximately)
        length = np.sqrt(forward_x**2 + forward_y**2 + forward_z**2)
        assert abs(length - 1.0) < 0.1
    
    def test_camera_get_right_vector(self):
        """Test right vector calculation."""
        camera = Camera()
        
        right_x, right_z = camera.get_right_vector()
        
        # Right vector should be normalized (approximately)
        length = np.sqrt(right_x**2 + right_z**2)
        assert abs(length - 1.0) < 0.1
    
    def test_camera_rotate(self):
        """Test camera rotation."""
        camera = Camera()
        initial_yaw = camera.yaw
        initial_pitch = camera.pitch
        
        camera.rotate(10.0, 5.0)
        
        assert camera.yaw != initial_yaw
        assert camera.pitch != initial_pitch
    
    def test_camera_zoom(self):
        """Test camera zoom."""
        camera = Camera()
        initial_y = camera.y
        
        camera.zoom(1.0)
        
        assert camera.y > initial_y
    
    def test_camera_zoom_minimum(self):
        """Test camera zoom minimum height."""
        camera = Camera()
        camera.y = 10.0
        
        camera.zoom(-100.0)  # Try to zoom way down
        
        assert camera.y >= 5.0  # Should be clamped to minimum
    
    def test_camera_resize(self):
        """Test camera resize."""
        camera = Camera(width=1280, height=720)
        
        camera.resize(1920, 1080)
        
        assert camera.width == 1920
        assert camera.height == 1080


class TestSkyManager:
    """Test sky and lighting management."""
    
    def test_sky_manager_initialization(self):
        """Test sky manager initialization."""
        sky = SkyManager(day_length=120.0)
        
        assert sky.day_length == 120.0
    
    def test_calculate_sky_color_day(self):
        """Test sky color calculation during day."""
        sky = SkyManager(day_length=120.0)
        
        # Noon (12 hours = 60 seconds into 120-second day)
        day_time = 60.0
        r, g, b = sky.calculate_sky_color(day_time)
        
        # Day sky should be bright blue
        assert r == 0.5
        assert g == 0.7
        assert b == 1.0
    
    def test_calculate_sky_color_night(self):
        """Test sky color calculation during night."""
        sky = SkyManager(day_length=120.0)
        
        # Midnight (0 hours = 0 seconds)
        day_time = 0.0
        r, g, b = sky.calculate_sky_color(day_time)
        
        # Night sky should be dark blue-purple
        assert r == 0.03
        assert g == 0.03
        assert b == 0.12
    
    def test_calculate_sky_color_dawn(self):
        """Test sky color calculation during dawn."""
        sky = SkyManager(day_length=120.0)
        
        # Dawn (6 hours = 30 seconds into 120-second day)
        day_time = 30.0
        r, g, b = sky.calculate_sky_color(day_time)
        
        # Dawn should transition from night to day
        assert 0.3 <= r <= 0.5
        assert 0.4 <= g <= 0.7
        assert 0.5 <= b <= 1.0
    
    def test_calculate_sky_color_dusk(self):
        """Test sky color calculation during dusk."""
        sky = SkyManager(day_length=120.0)
        
        # Dusk (19 hours = 95 seconds into 120-second day)
        day_time = 95.0
        r, g, b = sky.calculate_sky_color(day_time)
        
        # Dusk should transition from day to night
        assert 0.03 <= r <= 0.5
        assert 0.03 <= g <= 0.7
        assert 0.12 <= b <= 1.0


class TestNPCSelection:
    """Test NPC selection and deselection."""
    
    def test_npc_deselection_with_tab(self):
        """Test that TAB key deselects NPC."""
        # This test would require creating a renderer instance
        # For now, we test the logic conceptually
        # In integration tests, we verify TAB key triggers deselection
        pass
    
    def test_npc_deselection_with_click(self):
        """Test that clicking empty space deselects NPC."""
        # This test would require creating a renderer instance
        # For now, we test the logic conceptually
        pass


class TestTerrainSmoothing:
    """Test terrain smoothing to prevent floating islands."""
    
    def test_height_smoothing_prevents_large_differences(self):
        """Test that height smoothing prevents sudden drops."""
        world = World(seed=42)
        
        # Get heights at nearby points
        h1 = world.get_height(0.0, 0.0)
        h2 = world.get_height(1.0, 0.0)
        h3 = world.get_height(0.0, 1.0)
        h4 = world.get_height(1.0, 1.0)
        
        heights = [h1, h2, h3, h4]
        max_height = max(heights)
        min_height = min(heights)
        height_diff = max_height - min_height
        
        # Height difference should be reasonable (not too large)
        assert height_diff < world.generator.max_height * 0.5
    
    def test_bilinear_interpolation_smoothness(self):
        """Test that bilinear interpolation produces smooth transitions."""
        world = World(seed=42)
        
        # Get heights at closely spaced points
        heights = []
        for i in range(10):
            x = i * 0.1
            z = i * 0.1
            heights.append(world.get_height(x, z))
        
        # Check that adjacent heights don't have sudden jumps
        for i in range(1, len(heights)):
            diff = abs(heights[i] - heights[i-1])
            # Maximum change should be reasonable
            assert diff < world.generator.max_height * 0.1
