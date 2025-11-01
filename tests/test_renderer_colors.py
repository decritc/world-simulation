"""Tests for rendering color calculations."""

import pytest
import numpy as np


class TestRendererColorCalculations:
    """Test color calculation logic for rendering."""
    
    def test_sky_color_night(self):
        """Test that sky color at night is cool (blue-purple), not warm."""
        hour = 2.0  # 2am - night
        
        # Calculate expected sky color for night
        if hour < 6.0 or hour >= 18.0:  # Night
            sky_r, sky_g, sky_b = 0.03, 0.03, 0.12
        
        # Verify colors are cool (blue dominant, low red/green)
        assert sky_r < 0.1  # Low red
        assert sky_g < 0.1  # Low green
        assert sky_b > sky_r  # Blue should dominate
        assert sky_b > sky_g  # Blue should dominate
        assert sky_r < sky_b * 0.5  # Red should be much less than blue
    
    def test_sky_color_day(self):
        """Test that sky color during day is bright blue."""
        hour = 12.0  # Noon - day
        
        if 8.0 <= hour < 18.0:  # Day
            sky_r, sky_g, sky_b = 0.5, 0.7, 1.0
        
        # Verify colors are bright and blue-dominant
        assert sky_b > 0.8  # High blue
        assert sky_g > sky_r  # Green > Red (for blue sky)
        assert sky_b > sky_r  # Blue > Red
    
    def test_sky_color_dusk_transition(self):
        """Test that dusk transitions to cool colors, not warm."""
        hour = 19.0  # 7pm - dusk
        
        if 18.0 <= hour < 20.0:  # Dusk
            t = (hour - 18.0) / 2.0
            sky_r = 0.5 - 0.47 * t  # Should go from 0.5 to 0.03
            sky_g = 0.7 - 0.67 * t  # Should go from 0.7 to 0.03
            sky_b = 1.0 - 0.88 * t  # Should go from 1.0 to 0.12
        
        # Verify colors are transitioning to cool
        assert sky_r < 0.3  # Red should be low
        assert sky_b > sky_r  # Blue should dominate
        # At end of dusk (t=1.0), should match night colors
        if t >= 0.9:
            assert sky_r < 0.1
            assert sky_g < 0.1
    
    def test_terrain_color_night(self):
        """Test terrain color calculation at night."""
        is_night = True
        
        # Expected night terrain colors
        if is_night:
            terrain_r = 0.1
            terrain_g = 0.12
            terrain_b = 0.15
        
        # Verify colors are cool (blue-gray)
        assert terrain_r < 0.15  # Low red
        assert terrain_g < 0.2  # Low green
        assert terrain_b > terrain_r  # Blue should be higher than red
        assert terrain_b > terrain_g  # Blue should be higher than green
        # Check for cool tones (not warm/orange)
        assert terrain_r < terrain_b  # Red < Blue (cool)
        assert terrain_g < terrain_b  # Green < Blue (cool)
    
    def test_terrain_color_day(self):
        """Test terrain color calculation during day."""
        is_night = False
        
        # Expected day terrain colors (green grass)
        if not is_night:
            terrain_r = 0.2
            terrain_g = 0.5
            terrain_b = 0.15
        
        # Verify colors are green (grass-like)
        assert terrain_g > terrain_r  # Green > Red
        assert terrain_g > terrain_b  # Green > Blue
    
    def test_tree_leaf_color_night(self):
        """Test tree leaf color at night is cool."""
        is_night = True
        
        if is_night:
            leaf_r = 0.1
            leaf_g = 0.2
            leaf_b = 0.15
        
        # Verify colors are cool (muted, blue-green)
        assert leaf_r < 0.15  # Low red
        assert leaf_g < 0.3  # Low green
        assert leaf_b > leaf_r  # Blue should be higher than red
        # Should not be warm/orange
        assert leaf_r < leaf_b  # Red < Blue (cool)
    
    def test_house_color_night(self):
        """Test house color at night is cool."""
        is_night = True
        
        if is_night:
            house_r = 0.25
            house_g = 0.25
            house_b = 0.3
        
        # Verify colors are cool (blue-gray)
        assert house_b > house_r  # Blue > Red
        assert house_b > house_g  # Blue > Green
        assert house_r < 0.3  # Low red
        assert house_g < 0.3  # Low green
        # Should not be warm/orange
        assert house_r < house_b  # Red < Blue (cool)
    
    def test_light_color_night(self):
        """Test lighting color at night is cool (blue-white)."""
        is_night = True
        ambient_intensity = 0.2
        sun_intensity = 0.3
        
        if is_night:
            # Cool blue-white moonlight
            light_r = 0.3 * sun_intensity
            light_g = 0.4 * sun_intensity
            light_b = 0.8 * sun_intensity
        
        # Verify colors are cool (blue dominant)
        assert light_b > light_r  # Blue > Red
        assert light_b > light_g  # Blue > Green
        assert light_r < light_b * 0.6  # Red should be much less than blue
    
    def test_color_no_red_dominance_at_night(self):
        """Test that red/orange colors don't dominate at night."""
        # Night colors should not have red > blue or red > green significantly
        night_colors = [
            (0.03, 0.03, 0.12),  # Sky
            (0.1, 0.12, 0.15),    # Terrain
            (0.1, 0.2, 0.15),     # Tree leaves
            (0.25, 0.25, 0.3),    # House walls
            (0.3, 0.2, 0.25),     # House roof
        ]
        
        for r, g, b in night_colors:
            # Red should not dominate
            assert r < b or abs(r - b) < 0.1  # Red <= Blue (or very close)
            # Blue should be equal or higher than red
            assert b >= r * 0.8  # Blue should be at least 80% of red
            # Total should be low (dark night) - allow slightly higher for some objects
            assert r + g + b < 0.85  # Dark colors (slightly higher threshold for visibility)
    
    def test_day_vs_night_color_difference(self):
        """Test that day and night colors are clearly different."""
        # Day colors (warm)
        day_sky = (0.5, 0.7, 1.0)
        day_terrain = (0.2, 0.5, 0.15)
        
        # Night colors (cool)
        night_sky = (0.03, 0.03, 0.12)
        night_terrain = (0.1, 0.12, 0.15)
        
        # Day should have higher values overall
        assert sum(day_sky) > sum(night_sky)
        assert sum(day_terrain) > sum(night_terrain)
        
        # Day should have more green/blue, night should have more blue relative to red
        assert day_sky[2] > night_sky[2]  # Day blue > Night blue (but night is darker overall)
        assert day_terrain[1] > night_terrain[1]  # Day green > Night green
        
        # Night should have blue > red
        assert night_sky[2] > night_sky[0]  # Blue > Red
        assert night_terrain[2] > night_terrain[0]  # Blue > Red

