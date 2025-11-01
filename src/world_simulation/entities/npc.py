"""NPC entity implementation."""

from typing import List, Dict, Any
import numpy as np
from world_simulation.entities.name_generator import NameGenerator


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
        self.state = "wandering"  # wandering, seeking_food, eating, resting, seeking_shelter, in_shelter
        self.current_house = None  # House NPC is currently in
        
        # Age and lifecycle
        self.age = 0.0  # Age in seconds
        self.lifespan = np.random.uniform(600.0, 900.0)  # Lifespan: 10-15 minutes
        self.age_stage = "child"  # child, adult, elder
        self.adult_age = 120.0  # Become adult at 2 minutes
        self.elder_age = self.lifespan * 0.75  # Elder at 75% of lifespan
        
        # Reproduction
        self.can_reproduce = False  # Only adults can reproduce
        self.reproduction_cooldown = 0.0  # Time until can reproduce again
        self.reproduction_cooldown_time = 60.0  # 1 minute cooldown
        
        # Statistics
        self.fruit_collected = 0
        self.is_alive = True
        
        # Name generation
        self.name = self.genome.get('name') if genome is not None and 'name' in genome else NameGenerator.generate_full_name()
        
        # Initialize as adult if starting older (for initial population)
        if genome is not None and 'age' in genome:
            self.age = genome.pop('age', 0.0)
            # Check elder first since elder_age > adult_age
            if self.age >= self.elder_age:
                self.age_stage = "elder"
            elif self.age >= self.adult_age:
                self.age_stage = "adult"
                self.can_reproduce = True
        
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
        
        # Update age stage
        if self.age >= self.adult_age and self.age_stage == "child":
            self.age_stage = "adult"
            self.can_reproduce = True
            # Grow to full size when becoming adult
            if hasattr(self, 'genome'):
                self.size = self.genome.get('size', 1.0)
        elif self.age >= self.elder_age and self.age_stage == "adult":
            self.age_stage = "elder"
        
        # Death from old age
        if self.age >= self.lifespan:
            self.is_alive = False
            self.health = 0
            return
        
        # Update reproduction cooldown
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= delta_time
        
        # Update hunger
        self.hunger -= delta_time * 0.5
        if self.hunger < 0:
            self.hunger = 0
            self.health -= delta_time * 5.0  # Starvation damage
        
        # Gradual health decrease over time (aging/wear)
        # Health decreases slowly as NPC ages, but faster when hunger is low
        hunger_factor = max(0.0, self.hunger / self.max_hunger)  # 0.0 when starving, 1.0 when full
        aging_rate = 0.1 * (1.0 - hunger_factor * 0.5)  # Faster aging when hungry
        
        # Night penalty: Being outside at night causes faster health loss
        is_night = world.is_night()
        is_outside = self.current_house is None or self.state != "in_shelter"
        if is_night and is_outside:
            # Night exposure doubles the health loss rate
            aging_rate *= 2.0
        
        self.health -= delta_time * aging_rate
        
        # Ensure health doesn't go below 0
        if self.health < 0:
            self.health = 0
        
        # Update stamina
        if self.state == "resting":
            self.stamina = min(self.max_stamina, self.stamina + delta_time * 10.0)
        else:
            self.stamina -= delta_time * 0.5
        
        # Check if dead
        if self.health <= 0:
            self.is_alive = False
            return
        
        # Check if night and need shelter
        if world.is_night() and self.state not in ["in_shelter", "seeking_shelter"]:
            self.state = "seeking_shelter"
        elif not world.is_night() and self.state == "in_shelter":
            # Leave shelter when day comes
            if self.current_house:
                self.current_house.remove_occupant(id(self))
                self.current_house = None
            self.state = "wandering"
        
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
        elif self.state == "seeking_shelter":
            self._seek_shelter(delta_time, world)
        elif self.state == "in_shelter":
            # NPC is safe in shelter, restore stamina slower
            self.stamina = min(self.max_stamina, self.stamina + delta_time * 5.0)
            self.hunger -= delta_time * 0.2  # Less hunger loss in shelter
            # Reproduction can happen in shelter (handled by world.update)
    
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
    
    def _seek_shelter(self, delta_time: float, world):
        """Seek out a house for shelter."""
        vision_range = self.genome.get('vision_range', 10.0) * 1.5  # Better vision for shelter
        
        # Find nearest house with space
        nearest_house = None
        nearest_distance = float('inf')
        
        for house in world.houses:
            # Only adults can occupy houses (capacity is 2 adults)
            if house.is_built and house.can_shelter_adult() and self.age_stage == "adult":
                dx = house.x - self.x
                dz = house.z - self.z
                distance = np.sqrt(dx**2 + dz**2)
                
                if distance < vision_range and distance < nearest_distance:
                    nearest_distance = distance
                    nearest_house = house
        
        if nearest_house:
            # Move towards house
            dx = nearest_house.x - self.x
            dz = nearest_house.z - self.z
            distance = np.sqrt(dx**2 + dz**2)
            
            if distance < 1.5:
                # Enter house
                if nearest_house.add_occupant(id(self)):
                    self.current_house = nearest_house
                    self.state = "in_shelter"
            else:
                move_speed = self.speed * delta_time
                self.x += (dx / distance) * move_speed
                self.z += (dz / distance) * move_speed
                self.y = world.get_height(self.x, self.z)
        else:
            # No shelter found, keep wandering (will try again next frame)
            pass
    
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
            
            # Check if still near tree
            dx = tree.x - self.x
            dz = tree.z - self.z
            distance = np.sqrt(dx**2 + dz**2)
            
            if distance > 2.0:  # Too far, move closer
                move_speed = self.speed * delta_time
                self.x += (dx / distance) * move_speed
                self.z += (dz / distance) * move_speed
                self.y = world.get_height(self.x, self.z)
                return
            
            # Eat fruit continuously while near tree
            if tree.get_ripe_fruit_count() > 0:
                # Eat at a rate (1 fruit per second)
                if np.random.random() < delta_time * 2.0:  # Eat faster
                    fruit_eaten = tree.harvest_fruit()
                    if fruit_eaten > 0:
                        self.fruit_collected += fruit_eaten
                        self.hunger = min(self.max_hunger, self.hunger + fruit_eaten * 20.0)
                
                # Continue eating until full or no more fruit
                if self.hunger >= self.max_hunger * 0.9 or tree.get_ripe_fruit_count() == 0:
                    self.state = "wandering"
                    self.target_tree = None
            else:
                # No more fruit, seek new food
                self.state = "seeking_food"
                self.target_tree = None
        else:
            self.state = "wandering"
    
    def can_reproduce_with(self, other: 'NPC') -> bool:
        """Check if this NPC can reproduce with another NPC."""
        return (self.is_alive and other.is_alive and
                self.age_stage == "adult" and other.age_stage == "adult" and
                self.can_reproduce and other.can_reproduce and
                self.reproduction_cooldown <= 0 and other.reproduction_cooldown <= 0 and
                self.current_house == other.current_house and
                self.current_house is not None)
    
    def reproduce(self, partner: 'NPC') -> 'NPC':
        """
        Create an offspring NPC from two parents.
        
        Args:
            partner: The other parent NPC
            
        Returns:
            New offspring NPC
        """
        # Set cooldown for both parents
        self.reproduction_cooldown = self.reproduction_cooldown_time
        partner.reproduction_cooldown = partner.reproduction_cooldown_time
        
        # Create offspring genome (mix of parents with some mutation)
        offspring_genome = {}
        for key in self.genome.keys():
            if key == 'name':
                # Skip name - will be generated separately
                continue
            if key in partner.genome:
                # Average of parents with some mutation
                parent_avg = (self.genome[key] + partner.genome[key]) / 2.0
                mutation = np.random.uniform(-0.1, 0.1) * parent_avg
                offspring_genome[key] = max(0.1, parent_avg + mutation)
            else:
                offspring_genome[key] = self.genome[key]
        
        # Generate offspring name (mix of parents' last names or generate new)
        # 50% chance to inherit last name from one parent, 50% chance for new name
        if np.random.random() < 0.5:
            # Inherit last name from one parent
            parent1_last = self.name.split()[-1] if ' ' in self.name else self.name
            parent2_last = partner.name.split()[-1] if ' ' in partner.name else partner.name
            # Choose one parent's last name randomly
            inherited_last = np.random.choice([parent1_last, parent2_last])
            # Generate new first name
            offspring_first = NameGenerator.generate_first_name()
            offspring_name = f"{offspring_first} {inherited_last}"
        else:
            # Generate completely new name
            offspring_name = NameGenerator.generate_full_name()
        
        offspring_genome['name'] = offspring_name
        
        # Spawn offspring near parents' house
        house = self.current_house
        spawn_x = house.x + np.random.uniform(-1.0, 1.0)
        spawn_z = house.z + np.random.uniform(-1.0, 1.0)
        spawn_y = 0.0  # Will be set by world.get_height
        
        # Create offspring
        offspring = NPC(spawn_x, spawn_y, spawn_z, genome=offspring_genome)
        
        # Offspring starts as child
        offspring.age = 0.0
        offspring.age_stage = "child"
        offspring.can_reproduce = False
        offspring.size *= 0.7  # Smaller as child
        
        return offspring
    
    def get_fitness(self) -> float:
        """Calculate fitness score for genetic algorithm."""
        # Fitness based on survival time, fruit collected, and health
        survival_score = self.age * 0.1
        collection_score = self.fruit_collected * 10.0
        health_score = self.health * 0.5
        
        return survival_score + collection_score + health_score

