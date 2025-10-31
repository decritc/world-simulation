"""NPC entity implementation."""

from typing import List, Dict, Any
import numpy as np


class NPC:
    """An AI-controlled NPC that can evolve using genetic algorithms."""
    
    def __init__(self, x: float, y: float, z: float, genome: Dict[str, Any] = None):
        """
        Initialize an NPC.
        
        Args:
            x: Initial X position
            y: Initial Y position
            z: Initial Z position
            genome: Genetic traits for the NPC
        """
        self.x = x
        self.y = y
        self.z = z
        
        # Initialize genome if not provided
        if genome is None:
            self.genome = self._create_random_genome()
        else:
            self.genome = genome
        
        # Physical attributes based on genome
        self.speed = self.genome.get('speed', 1.0)
        self.size = self.genome.get('size', 1.0)
        self.stamina = self.genome.get('stamina', 100.0)
        self.max_stamina = self.stamina
        
        # Behavioral attributes
        self.hunger = 50.0
        self.max_hunger = 100.0
        self.health = 100.0
        self.max_health = 100.0
        
        # State
        self.target_x = None
        self.target_z = None
        self.state = "wandering"  # wandering, seeking_food, eating, resting
        
        # Statistics
        self.age = 0.0
        self.fruit_collected = 0
        self.is_alive = True
        
    def _create_random_genome(self) -> Dict[str, Any]:
        """Create a random genome for the NPC."""
        return {
            'speed': np.random.uniform(0.5, 2.0),
            'size': np.random.uniform(0.8, 1.2),
            'stamina': np.random.uniform(50.0, 150.0),
            'vision_range': np.random.uniform(5.0, 20.0),
            'food_preference': np.random.uniform(0.0, 1.0),
        }
    
    def update(self, delta_time: float, world):
        """Update the NPC's state."""
        if not self.is_alive:
            return
        
        self.age += delta_time
        
        # Update hunger
        self.hunger -= delta_time * 0.5
        if self.hunger < 0:
            self.hunger = 0
            self.health -= delta_time * 5.0  # Starvation damage
        
        # Update stamina
        if self.state == "resting":
            self.stamina = min(self.max_stamina, self.stamina + delta_time * 10.0)
        else:
            self.stamina -= delta_time * 0.5
        
        # Check if dead
        if self.health <= 0:
            self.is_alive = False
            return
        
        # State machine
        if self.state == "wandering":
            self._wander(delta_time, world)
        elif self.state == "seeking_food":
            self._seek_food(delta_time, world)
        elif self.state == "eating":
            self._eat(delta_time, world)
        elif self.state == "resting":
            if self.stamina >= self.max_stamina * 0.8:
                self.state = "wandering"
    
    def _wander(self, delta_time: float, world):
        """Wander randomly around the world."""
        if self.target_x is None or self.target_z is None:
            # Pick a new random target
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(5.0, 15.0)
            self.target_x = self.x + np.cos(angle) * distance
            self.target_z = self.z + np.sin(angle) * distance
        
        # Move towards target
        dx = self.target_x - self.x
        dz = self.target_z - self.z
        distance = np.sqrt(dx**2 + dz**2)
        
        if distance < 0.5:
            self.target_x = None
            self.target_z = None
        else:
            # Move
            move_speed = self.speed * delta_time
            self.x += (dx / distance) * move_speed
            self.z += (dz / distance) * move_speed
            self.y = world.get_height(self.x, self.z)
        
        # Check for food
        if self.hunger < 50.0:
            self.state = "seeking_food"
        
        # Check for rest
        if self.stamina < 20.0:
            self.state = "resting"
    
    def _seek_food(self, delta_time: float, world):
        """Seek out fruit trees."""
        vision_range = self.genome.get('vision_range', 10.0)
        
        # Find nearest tree with fruit
        nearest_tree = None
        nearest_distance = float('inf')
        
        for tree in world.trees:
            if tree.is_alive and tree.get_ripe_fruit_count() > 0:
                dx = tree.x - self.x
                dz = tree.z - self.z
                distance = np.sqrt(dx**2 + dz**2)
                
                if distance < vision_range and distance < nearest_distance:
                    nearest_distance = distance
                    nearest_tree = tree
        
        if nearest_tree:
            # Move towards tree
            dx = nearest_tree.x - self.x
            dz = nearest_tree.z - self.z
            distance = np.sqrt(dx**2 + dz**2)
            
            if distance < 1.5:
                self.state = "eating"
                self.target_tree = nearest_tree
            else:
                move_speed = self.speed * delta_time
                self.x += (dx / distance) * move_speed
                self.z += (dz / distance) * move_speed
                self.y = world.get_height(self.x, self.z)
        else:
            # No food found, go back to wandering
            self.state = "wandering"
    
    def _eat(self, delta_time: float, world):
        """Eat fruit from a tree."""
        if hasattr(self, 'target_tree') and self.target_tree:
            tree = self.target_tree
            if tree.get_ripe_fruit_count() > 0:
                fruit_eaten = tree.harvest_fruit()
                self.fruit_collected += fruit_eaten
                self.hunger = min(self.max_hunger, self.hunger + fruit_eaten * 20.0)
                
                if self.hunger >= self.max_hunger * 0.9:
                    self.state = "wandering"
                    self.target_tree = None
            else:
                self.state = "seeking_food"
                self.target_tree = None
        else:
            self.state = "wandering"
    
    def get_fitness(self) -> float:
        """Calculate fitness score for genetic algorithm."""
        # Fitness based on survival time, fruit collected, and health
        survival_score = self.age * 0.1
        collection_score = self.fruit_collected * 10.0
        health_score = self.health * 0.5
        
        return survival_score + collection_score + health_score

