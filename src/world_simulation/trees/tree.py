"""Fruit tree implementation."""

import numpy as np


class FruitTree:
    """A fruit tree that grows and produces fruit."""
    
    def __init__(self, x: float, y: float, z: float, species: str = "apple"):
        """
        Initialize a fruit tree.
        
        Args:
            x: X position in world
            y: Y position in world
            z: Z position in world
            species: Type of fruit tree
        """
        self.x = x
        self.y = y
        self.z = z
        self.species = species
        
        self.age = 0.0
        # max_age in seconds: 100 days * 120 seconds per day = 12,000 seconds (very long lifespan)
        # But for gameplay, let's make trees live much longer - effectively immortal
        self.max_age = 1000000.0  # Effectively immortal (prevents trees from dying too quickly)
        self.growth_stage = 0.0  # 0.0 to 1.0
        
        self.fruit_count = 0
        self.max_fruit = 30  # Increased from 10
        self.fruit_maturity = {}  # fruit_id -> maturity (0.0 to 1.0)
        
        # Tree reproduction
        self.reproduction_cooldown = 0.0  # Time until can reproduce again
        self.reproduction_cooldown_time = 60.0  # 1 minute cooldown between reproductions
        self.reproduction_age = 30.0  # Must be mature (fully grown) to reproduce
        
        self.is_alive = True
        
    def update(self, delta_time: float, world):
        """Update the tree's state."""
        if not self.is_alive:
            return
        
        # Age the tree
        self.age += delta_time
        
        # Grow the tree
        if self.growth_stage < 1.0:
            self.growth_stage = min(1.0, self.age / 30.0)  # Full growth in 30 seconds
        
        # Produce fruit if mature - increase production rate significantly
        if self.growth_stage >= 0.5 and self.fruit_count < self.max_fruit:
            # Chance to produce new fruit (much higher rate)
            if np.random.random() < 0.1 * delta_time:  # Increased from 0.02
                fruit_id = len(self.fruit_maturity)
                self.fruit_maturity[fruit_id] = 0.0
                self.fruit_count += 1
        
        # Mature existing fruit - faster maturation
        for fruit_id in list(self.fruit_maturity.keys()):
            self.fruit_maturity[fruit_id] += delta_time * 0.2  # Faster maturation
            if self.fruit_maturity[fruit_id] >= 1.0:
                # Fruit is fully mature and ready to eat
                pass
        
        # Update reproduction cooldown
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= delta_time
        
        # Tree reproduction: mature trees can spawn new trees nearby
        if (self.growth_stage >= 1.0 and 
            self.age >= self.reproduction_age and 
            self.reproduction_cooldown <= 0):
            # Chance to reproduce (0.1% chance per second)
            reproduction_chance = 0.001 * delta_time
            if np.random.random() < reproduction_chance:
                self._spawn_sapling(world)
                self.reproduction_cooldown = self.reproduction_cooldown_time
        
        # Check if tree dies - spawn new tree when dying
        if self.age >= self.max_age:
            # Spawn a new tree before dying
            self._spawn_sapling(world)
            self.is_alive = False
    
    def harvest_fruit(self) -> int:
        """Harvest mature fruit from the tree."""
        mature_fruit = [
            fruit_id for fruit_id, maturity in self.fruit_maturity.items()
            if maturity >= 1.0
        ]
        
        for fruit_id in mature_fruit:
            del self.fruit_maturity[fruit_id]
            self.fruit_count -= 1
        
        return len(mature_fruit)
    
    def get_ripe_fruit_count(self) -> int:
        """Get the number of ripe fruit on the tree."""
        return sum(1 for maturity in self.fruit_maturity.values() if maturity >= 1.0)
    
    def _spawn_sapling(self, world):
        """
        Spawn a new tree (sapling) near this tree.
        
        Args:
            world: World instance to add the new tree to
        """
        # Try to find a valid spawn location nearby
        spawn_radius = 3.0  # Spawn within 3 units
        max_attempts = 10
        
        for attempt in range(max_attempts):
            # Random angle and distance
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(1.0, spawn_radius)
            
            spawn_x = self.x + np.cos(angle) * distance
            spawn_z = self.z + np.sin(angle) * distance
            
            # Check if location is valid (not too close to other trees or houses)
            too_close = False
            
            # Check distance to other trees
            for tree in world.trees:
                if tree != self:
                    dx = tree.x - spawn_x
                    dz = tree.z - spawn_z
                    dist = np.sqrt(dx*dx + dz*dz)
                    if dist < 2.0:  # Minimum spacing between trees
                        too_close = True
                        break
            
            # Check distance to houses
            if not too_close:
                for house in world.houses:
                    dx = house.x - spawn_x
                    dz = house.z - spawn_z
                    dist = np.sqrt(dx*dx + dz*dz)
                    if dist < 2.0:  # Minimum spacing from houses
                        too_close = True
                        break
            
            # If location is valid, spawn the tree
            if not too_close:
                spawn_y = world.get_height(spawn_x, spawn_z)
                # Create new tree with same species
                new_tree = FruitTree(spawn_x, spawn_y, spawn_z, species=self.species)
                # New tree starts as sapling (age 0, growth_stage 0)
                new_tree.age = 0.0
                new_tree.growth_stage = 0.0
                
                world.trees.append(new_tree)
                
                # Log if renderer is available
                if hasattr(world, 'renderer') and world.renderer:
                    world.renderer.log(f"New {self.species} tree spawned!")
                break

