"""Animal entity implementation."""

import numpy as np
from typing import Optional


class Animal:
    """A simple animal that can be hunted by NPCs."""
    
    def __init__(self, x: float, y: float, z: float, species: str = "deer"):
        """
        Initialize an animal.
        
        Args:
            x: Initial X position
            y: Initial Y position
            z: Initial Z position
            species: Type of animal (deer, rabbit, etc.)
        """
        self.x = x
        self.y = y
        self.z = z
        
        self.species = species
        
        # Physical attributes (varies by species)
        species_stats = {
            "deer": {"speed": 1.5, "size": 1.0, "health": 50.0, "meat_value": 40.0},
            "rabbit": {"speed": 2.0, "size": 0.5, "health": 20.0, "meat_value": 15.0},
            "boar": {"speed": 1.0, "size": 1.2, "health": 80.0, "meat_value": 50.0},
        }
        
        stats = species_stats.get(species, species_stats["deer"])
        self.speed = stats["speed"]
        self.size = stats["size"]
        self.health = stats["health"]
        self.max_health = stats["health"]
        self.meat_value = stats["meat_value"]  # How much hunger this animal satisfies
        
        # Behavior
        self.target_x: Optional[float] = None
        self.target_z: Optional[float] = None
        self.flee_target_x: Optional[float] = None
        self.flee_target_z: Optional[float] = None
        self.flee_timer = 0.0
        self.is_fleeing = False
        
        # State
        self.is_alive = True
        self.age = 0.0
        self.max_age = np.random.uniform(300.0, 600.0)  # 5-10 minutes lifespan
        
        # Reproduction
        self.reproduction_cooldown = 0.0
        self.reproduction_cooldown_time = 120.0  # 2 minutes cooldown
    
    def update(self, delta_time: float, world):
        """
        Update animal state.
        
        Args:
            delta_time: Time since last update
            world: World instance
        """
        if not self.is_alive:
            return
        
        # Update age
        self.age += delta_time
        
        # Die of old age
        if self.age >= self.max_age:
            self.is_alive = False
            return
        
        # Update reproduction cooldown
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= delta_time
        
        # Update fleeing timer
        if self.is_fleeing:
            self.flee_timer -= delta_time
            if self.flee_timer <= 0:
                self.is_fleeing = False
                self.flee_target_x = None
                self.flee_target_z = None
        
        # Check for nearby NPCs (threat detection)
        threat_range = 8.0
        nearest_threat = None
        nearest_threat_distance = float('inf')
        
        for npc in world.entities:
            if not npc.is_alive:
                continue
            
            dx = npc.x - self.x
            dz = npc.z - self.z
            distance = np.sqrt(dx*dx + dz*dz)
            
            if distance < threat_range and distance < nearest_threat_distance:
                nearest_threat = npc
                nearest_threat_distance = distance
        
        # Flee from threat
        if nearest_threat is not None:
            self.is_fleeing = True
            self.flee_timer = 3.0  # Flee for 3 seconds
            
            # Run away from threat
            flee_dx = self.x - nearest_threat.x
            flee_dz = self.z - nearest_threat.z
            flee_distance = np.sqrt(flee_dx*flee_dx + flee_dz*flee_dz)
            
            if flee_distance > 0.01:
                # Normalize and set flee target
                self.flee_target_x = self.x + (flee_dx / flee_distance) * 15.0
                self.flee_target_z = self.z + (flee_dz / flee_distance) * 15.0
                
                # Move towards flee target
                move_speed = self.speed * 1.5 * delta_time  # Run faster when fleeing
                move_dx = self.flee_target_x - self.x
                move_dz = self.flee_target_z - self.z
                move_distance = np.sqrt(move_dx*move_dx + move_dz*move_dz)
                
                if move_distance > 0.5:
                    self.x += (move_dx / move_distance) * move_speed
                    self.z += (move_dz / move_distance) * move_speed
                    self.y = world.get_height(self.x, self.z)
        else:
            # No threat, wander around
            if self.target_x is None or self.target_z is None or not self.is_fleeing:
                # Pick new wander target
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(3.0, 8.0)
                self.target_x = self.x + np.cos(angle) * distance
                self.target_z = self.z + np.sin(angle) * distance
                # Ensure target is at valid terrain height
                target_y = world.get_height(self.target_x, self.target_z)
                if abs(target_y - self.y) > 5.0:  # Too steep
                    self.target_x = None
                    self.target_z = None
                    return
            
            # Move towards wander target
            if self.target_x is not None and self.target_z is not None:
                dx = self.target_x - self.x
                dz = self.target_z - self.z
                distance = np.sqrt(dx*dx + dz*dz)
                
                if distance < 0.5:
                    self.target_x = None
                    self.target_z = None
                else:
                    move_speed = self.speed * delta_time
                    self.x += (dx / distance) * move_speed
                    self.z += (dz / distance) * move_speed
                    self.y = world.get_height(self.x, self.z)
    
    def take_damage(self, damage: float):
        """Take damage from being hunted."""
        self.health -= damage
        if self.health <= 0:
            self.is_alive = False
    
    def can_reproduce(self) -> bool:
        """Check if animal can reproduce."""
        return self.is_alive and self.reproduction_cooldown <= 0
    
    def reproduce(self, world) -> Optional['Animal']:
        """
        Spawn a new animal nearby.
        
        Args:
            world: World instance
            
        Returns:
            New Animal instance or None if reproduction failed
        """
        if not self.can_reproduce():
            return None
        
        # Spawn nearby
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(2.0, 5.0)
        spawn_x = self.x + np.cos(angle) * distance
        spawn_z = self.z + np.sin(angle) * distance
        spawn_y = world.get_height(spawn_x, spawn_z)
        
        # Create new animal
        new_animal = Animal(spawn_x, spawn_y, spawn_z, self.species)
        
        # Set cooldown
        self.reproduction_cooldown = self.reproduction_cooldown_time
        
        return new_animal
