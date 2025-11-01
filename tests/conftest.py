"""Test configuration and fixtures."""

import pytest
import numpy as np


@pytest.fixture
def world():
    """Create a test world instance."""
    from world_simulation.world.world import World
    return World(seed=42)


@pytest.fixture
def npc():
    """Create a test NPC instance."""
    from world_simulation.entities.npc import NPC
    return NPC(0.0, 0.0, 0.0)


@pytest.fixture
def tree():
    """Create a test tree instance."""
    from world_simulation.trees.tree import FruitTree
    return FruitTree(0.0, 0.0, 0.0, species="apple")


@pytest.fixture
def house():
    """Create a test house instance."""
    from world_simulation.houses.house import House
    return House(0.0, 0.0, 0.0, capacity=5)

