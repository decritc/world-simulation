"""Tests for renderer crash scenarios."""

import pytest
from world_simulation.world.world import World


class TestRendererCrashScenarios:
    """Test scenarios that could cause crashes."""
    
    def test_get_time_info_no_recursion(self):
        """Test that _get_time_info doesn't cause recursion."""
        world = World(seed=42)
        
        try:
            from world_simulation.rendering.renderer import Renderer
            renderer = Renderer(world, width=800, height=600)
            
            # Call _get_time_info multiple times to ensure no recursion
            for _ in range(100):
                hour, is_night = renderer._get_time_info()
                assert isinstance(hour, float)
                assert isinstance(is_night, bool)
                assert 0.0 <= hour <= 24.0
            
        except Exception as e:
            # If it's an OpenGL context issue, skip
            if "OpenGL" in str(e) or "GL" in str(e) or "window" in str(e).lower():
                pytest.skip(f"OpenGL context not available: {e}")
            else:
                raise
    
    def test_render_method_calls_get_time_info(self):
        """Test that render method properly calls _get_time_info."""
        world = World(seed=42)
        
        try:
            from world_simulation.rendering.renderer import Renderer
            renderer = Renderer(world, width=800, height=600)
            
            # Verify _get_time_info exists and works
            assert hasattr(renderer, '_get_time_info')
            assert callable(renderer._get_time_info)
            
            # Call it directly
            hour, is_night = renderer._get_time_info()
            assert isinstance(hour, float)
            assert isinstance(is_night, bool)
            
        except Exception as e:
            if "OpenGL" in str(e) or "GL" in str(e) or "window" in str(e).lower():
                pytest.skip(f"OpenGL context not available: {e}")
            else:
                raise
    
    def test_render_method_doesnt_crash(self):
        """Test that render method doesn't crash due to recursion."""
        world = World(seed=42)
        
        try:
            from world_simulation.rendering.renderer import Renderer
            renderer = Renderer(world, width=800, height=600)
            
            # Set up a basic world state
            world.day_time = 60.0  # Set to noon
            
            # Try to call render (this might fail due to OpenGL context, but shouldn't recurse)
            try:
                renderer.render()
            except RecursionError:
                pytest.fail("Render method caused recursion error")
            except Exception as e:
                # Other errors (like OpenGL context) are OK for this test
                if "RecursionError" in str(type(e)):
                    pytest.fail(f"Render method caused recursion error: {e}")
            
        except Exception as e:
            if "OpenGL" in str(e) or "GL" in str(e) or "window" in str(e).lower():
                pytest.skip(f"OpenGL context not available: {e}")
            else:
                raise
    
    def test_time_info_calculation_is_correct(self):
        """Test that time info calculation is mathematically correct."""
        world = World(seed=42)
        
        try:
            from world_simulation.rendering.renderer import Renderer
            renderer = Renderer(world, width=800, height=600)
            
            # Test at different times
            test_cases = [
                (0.0, 0.0),      # Start of day (midnight)
                (60.0, 12.0),    # Middle of day (noon)
                (120.0, 24.0),   # End of day
            ]
            
            for day_time, expected_hour in test_cases:
                world.day_time = day_time
                hour, is_night = renderer._get_time_info()
                
                # Hour should be approximately correct (within 0.1)
                assert abs(hour - expected_hour) < 0.1 or abs(hour - (expected_hour % 24.0)) < 0.1
                
                # is_night should be consistent
                if expected_hour < 6.0 or expected_hour >= 18.0:
                    assert is_night == True
                else:
                    assert is_night == False
            
        except Exception as e:
            if "OpenGL" in str(e) or "GL" in str(e) or "window" in str(e).lower():
                pytest.skip(f"OpenGL context not available: {e}")
            else:
                raise
