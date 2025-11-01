"""Tests for GPU instancing vegetation system."""

import pytest
import numpy as np
from world_simulation.world.world import World
from world_simulation.world.vegetation import Vegetation
from world_simulation.rendering.vegetation_instancer import VegetationInstancer


class TestVegetationInstancer:
    """Test the GPU instancing system for vegetation."""
    
    def test_instancer_initialization(self):
        """Test that instancer initializes correctly."""
        instancer = VegetationInstancer()
        
        assert instancer.instances_by_type is not None
        assert 'bush' in instancer.instances_by_type
        assert 'grass' in instancer.instances_by_type
        assert 'flower' in instancer.instances_by_type
        assert 'rock' in instancer.instances_by_type
        
        # All should be empty initially
        assert len(instancer.instances_by_type['bush']) == 0
        assert len(instancer.instances_by_type['grass']) == 0
        assert len(instancer.instances_by_type['flower']) == 0
        assert len(instancer.instances_by_type['rock']) == 0
    
    def test_prepare_instances_grouping(self):
        """Test that instances are grouped correctly by type."""
        instancer = VegetationInstancer()
        
        # Create test vegetation
        vegetation = [
            Vegetation(0.0, 0.5, 0.0, 'bush', 0.5),
            Vegetation(1.0, 0.5, 1.0, 'bush', 0.6),
            Vegetation(2.0, 0.5, 2.0, 'grass', 0.3),
            Vegetation(3.0, 0.5, 3.0, 'grass', 0.4),
            Vegetation(4.0, 0.5, 4.0, 'flower', 0.2),
            Vegetation(5.0, 0.5, 5.0, 'rock', 0.5),
        ]
        
        instancer.prepare_instances(vegetation)
        
        # Check grouping
        assert len(instancer.instances_by_type['bush']) == 2
        assert len(instancer.instances_by_type['grass']) == 2
        assert len(instancer.instances_by_type['flower']) == 1
        assert len(instancer.instances_by_type['rock']) == 1
    
    def test_prepare_instances_filters_dead(self):
        """Test that dead vegetation is filtered out."""
        instancer = VegetationInstancer()
        
        vegetation = [
            Vegetation(0.0, 0.5, 0.0, 'bush', 0.5, is_alive=True),
            Vegetation(1.0, 0.5, 1.0, 'bush', 0.6, is_alive=False),
            Vegetation(2.0, 0.5, 2.0, 'grass', 0.3, is_alive=True),
        ]
        
        instancer.prepare_instances(vegetation)
        
        # Only alive instances should be included
        assert len(instancer.instances_by_type['bush']) == 1
        assert len(instancer.instances_by_type['grass']) == 1
    
    def test_prepare_instances_clears_previous(self):
        """Test that previous instances are cleared when preparing new ones."""
        instancer = VegetationInstancer()
        
        # First batch
        vegetation1 = [
            Vegetation(0.0, 0.5, 0.0, 'bush', 0.5),
            Vegetation(1.0, 0.5, 1.0, 'grass', 0.3),
        ]
        instancer.prepare_instances(vegetation1)
        
        assert len(instancer.instances_by_type['bush']) == 1
        assert len(instancer.instances_by_type['grass']) == 1
        
        # Second batch (different instances)
        vegetation2 = [
            Vegetation(2.0, 0.5, 2.0, 'flower', 0.2),
        ]
        instancer.prepare_instances(vegetation2)
        
        # Previous instances should be cleared
        assert len(instancer.instances_by_type['bush']) == 0
        assert len(instancer.instances_by_type['grass']) == 0
        assert len(instancer.instances_by_type['flower']) == 1
    
    def test_render_all_empty(self):
        """Test that rendering empty instances doesn't crash."""
        instancer = VegetationInstancer()
        
        # Should not crash with empty instances
        try:
            instancer.render_all(is_night=False, max_height=10.0)
        except Exception as e:
            pytest.fail(f"render_all should handle empty instances: {e}")
    
    def test_render_batch_methods_exist(self):
        """Test that all batch render methods exist."""
        instancer = VegetationInstancer()
        
        assert hasattr(instancer, 'render_bush_batch')
        assert hasattr(instancer, 'render_grass_batch')
        assert hasattr(instancer, 'render_flower_batch')
        assert hasattr(instancer, 'render_rock_batch')
        assert hasattr(instancer, 'render_all')
    
    def test_integration_with_world(self):
        """Test integration with World vegetation."""
        world = World(seed=42)
        
        # World may or may not have vegetation initially (it's generated procedurally)
        # Create instancer
        instancer = VegetationInstancer()
        
        # If world has vegetation, test grouping
        if len(world.vegetation) > 0:
            instancer.prepare_instances(world.vegetation)
            
            # Check that instances were grouped
            total_instances = (
                len(instancer.instances_by_type['bush']) +
                len(instancer.instances_by_type['grass']) +
                len(instancer.instances_by_type['flower']) +
                len(instancer.instances_by_type['rock'])
            )
            
            # Should have at least some instances (filtered by alive status)
            alive_count = sum(1 for v in world.vegetation if v.is_alive)
            assert total_instances <= alive_count
            
            # All instances should be valid vegetation types
            for veg_type in instancer.instances_by_type:
                for veg in instancer.instances_by_type[veg_type]:
                    assert veg.vegetation_type == veg_type
                    assert veg.is_alive
                    assert veg.size > 0
        else:
            # If no vegetation, test that instancer handles empty list
            instancer.prepare_instances([])
            total_instances = (
                len(instancer.instances_by_type['bush']) +
                len(instancer.instances_by_type['grass']) +
                len(instancer.instances_by_type['flower']) +
                len(instancer.instances_by_type['rock'])
            )
            assert total_instances == 0

