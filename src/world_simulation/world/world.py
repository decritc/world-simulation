"""Main world class that manages the simulation."""

from typing import Dict, List, Tuple
import numpy as np

from .generator import WorldGenerator


class World:
    """Main world class managing terrain, entities, and simulation state."""
    
    def __init__(self, seed: int = 42):
        """Initialize the world."""
        self.generator = WorldGenerator(seed)
        self.chunks: Dict[Tuple[int, int], np.ndarray] = {}
        self.entities: List = []
        self.trees: List = []
        self.houses: List = []
        
        # Time tracking
        self.time = 0.0  # Total simulation time
        self.day_time = 0.0  # Time within current day (0-24 hours)
        self.day_length = 120.0  # Seconds per day
        self.day_number = 0
        
    def is_night(self) -> bool:
        """Check if it's currently night time."""
        hour = (self.day_time / self.day_length) * 24.0
        return hour < 6.0 or hour >= 18.0  # Night from 6pm to 6am
    
    def get_time_of_day(self) -> float:
        """Get current time of day (0.0 to 1.0, where 0.5 is noon)."""
        return (self.day_time / self.day_length) % 1.0
    
    def get_light_intensity(self) -> float:
        """Get light intensity based on time of day (0.0 to 1.0)."""
        hour = (self.day_time / self.day_length) * 24.0
        
        if 6.0 <= hour < 8.0:  # Dawn
            return 0.3 + 0.4 * ((hour - 6.0) / 2.0)
        elif 8.0 <= hour < 18.0:  # Day
            return 1.0
        elif 18.0 <= hour < 20.0:  # Dusk
            return 1.0 - 0.4 * ((hour - 18.0) / 2.0)
        else:  # Night
            return 0.3
        
    def get_chunk(self, chunk_x: int, chunk_y: int) -> np.ndarray:
        """
        Get or generate a chunk at the given coordinates.
        
        Args:
            chunk_x: X coordinate of the chunk
            chunk_y: Y coordinate of the chunk
            
        Returns:
            Heightmap for the chunk
        """
        chunk_key = (chunk_x, chunk_y)
        
        if chunk_key not in self.chunks:
            self.chunks[chunk_key] = self.generator.generate_chunk(chunk_x, chunk_y)
        
        return self.chunks[chunk_key]
    
    def get_height(self, x: float, z: float) -> float:
        """
        Get the height at a specific world coordinate.
        
        Args:
            x: X coordinate
            z: Z coordinate
            
        Returns:
            Height value at the given coordinates
        """
        chunk_x = int(x // 64)
        chunk_z = int(z // 64)
        
        chunk = self.get_chunk(chunk_x, chunk_z)
        
        local_x = int(x % 64)
        local_z = int(z % 64)
        
        if local_x >= 64:
            local_x = 63
        if local_z >= 64:
            local_z = 63
        
        return chunk[local_z][local_x]
    
    def update(self, delta_time: float):
        """Update the world simulation."""
        self.time += delta_time
        
        # Update day/night cycle
        old_day = self.day_number
        self.day_time += delta_time
        if self.day_time >= self.day_length:
            self.day_time = 0.0
            self.day_number += 1
        
        # Update entities
        for entity in self.entities:
            entity.update(delta_time, self)
        
        # Remove dead NPCs
        self.entities = [entity for entity in self.entities if entity.is_alive]
        
        # Clean up houses - remove dead NPCs from occupancy
        for house in self.houses:
            # Remove dead NPC IDs from house occupants
            dead_ids = set()
            for npc_id in house.current_occupants:
                # Check if any NPC still has this ID (simple check)
                found = False
                for entity in self.entities:
                    if id(entity) == npc_id:
                        found = True
                        break
                if not found:
                    dead_ids.add(npc_id)
            for npc_id in dead_ids:
                house.remove_occupant(npc_id)
        
        # Handle reproduction for adults in houses
        # Only allow reproduction if exactly 2 adults in a house
        reproduction_chance = 0.01 * delta_time  # 1% chance per second
        
        for house in self.houses:
            # Find all adult NPCs in this house
            adult_npcs = []
            for npc in self.entities:
                if (npc.is_alive and npc.current_house == house and 
                    npc.age_stage == "adult"):
                    adult_npcs.append(npc)
            
            # Only reproduce if exactly 2 adults in house
            if len(adult_npcs) == 2:
                npc1, npc2 = adult_npcs[0], adult_npcs[1]
                
                # Check if they can reproduce
                if npc1.can_reproduce_with(npc2):
                    # Random chance to reproduce
                    if np.random.random() < reproduction_chance:
                        # Create offspring
                        offspring = npc1.reproduce(npc2)
                        offspring.y = self.get_height(offspring.x, offspring.z)
                        self.entities.append(offspring)
                        # Log reproduction (will be displayed in renderer if available)
                        if hasattr(self, 'renderer') and self.renderer:
                            self.renderer.log(f"New offspring born! Age stage: {offspring.age_stage}")
        
        # Update trees
        trees_to_remove = []
        for tree in self.trees:
            was_alive = tree.is_alive
            tree.update(delta_time, self)
            # Track trees that died (but don't remove yet - they might spawn new trees)
            if not tree.is_alive and was_alive:
                trees_to_remove.append(tree)
        
        # Remove dead trees after they've had a chance to spawn new ones
        for tree in trees_to_remove:
            if tree in self.trees:
                self.trees.remove(tree)

