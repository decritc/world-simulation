"""Genetic algorithm implementation for evolving NPCs."""

import random
from typing import List, Dict, Any, Tuple
import numpy as np

from ..entities.npc import NPC


class EvolutionEngine:
    """Engine for evolving NPCs using genetic algorithms."""
    
    def __init__(self, population_size: int = 50):
        """
        Initialize the evolution engine.
        
        Args:
            population_size: Number of NPCs in each generation
        """
        self.population_size = population_size
        self.generation = 0
        
    def create_initial_population(self, spawn_points: List[Tuple[float, float, float]]) -> List[NPC]:
        """
        Create the initial population of NPCs.
        
        Args:
            spawn_points: List of (x, y, z) spawn points
            
        Returns:
            List of NPCs
        """
        population = []
        
        for i in range(self.population_size):
            if i < len(spawn_points):
                x, y, z = spawn_points[i]
            else:
                # Random spawn if not enough spawn points
                idx = i % len(spawn_points)
                x, y, z = spawn_points[idx]
                x += np.random.uniform(-5, 5)
                z += np.random.uniform(-5, 5)
            
            npc = NPC(x, y, z)
            population.append(npc)
        
        return population
    
    def evaluate_fitness(self, population: List[NPC]) -> List[float]:
        """
        Evaluate fitness of all NPCs in the population.
        
        Args:
            population: List of NPCs
            
        Returns:
            List of fitness scores
        """
        return [npc.get_fitness() for npc in population]
    
    def select_parents(self, population: List[NPC], fitness_scores: List[float], 
                       num_parents: int = 10) -> List[NPC]:
        """
        Select parents for reproduction using tournament selection.
        
        Args:
            population: List of NPCs
            fitness_scores: Corresponding fitness scores
            num_parents: Number of parents to select
            
        Returns:
            List of selected parent NPCs
        """
        parents = []
        
        for _ in range(num_parents):
            # Tournament selection
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        
        return parents
    
    def crossover(self, parent1: NPC, parent2: NPC) -> Dict[str, Any]:
        """
        Create a child genome by crossing over two parent genomes.
        
        Args:
            parent1: First parent NPC
            parent2: Second parent NPC
            
        Returns:
            Child genome dictionary
        """
        child_genome = {}
        
        for key in parent1.genome.keys():
            # Uniform crossover: randomly choose from each parent
            if random.random() < 0.5:
                child_genome[key] = parent1.genome[key]
            else:
                child_genome[key] = parent2.genome[key]
        
        return child_genome
    
    def mutate(self, genome: Dict[str, Any], mutation_rate: float = 0.1) -> Dict[str, Any]:
        """
        Mutate a genome with random changes.
        
        Args:
            genome: Genome to mutate
            mutation_rate: Probability of mutation for each gene
            
        Returns:
            Mutated genome
        """
        mutated_genome = genome.copy()
        
        for key in mutated_genome.keys():
            if random.random() < mutation_rate:
                # Apply mutation based on gene type
                if isinstance(mutated_genome[key], float):
                    # Add small random change
                    mutation_strength = mutated_genome[key] * 0.1
                    mutated_genome[key] += np.random.uniform(-mutation_strength, mutation_strength)
                    # Clamp to reasonable bounds
                    if key == 'speed':
                        mutated_genome[key] = np.clip(mutated_genome[key], 0.1, 3.0)
                    elif key == 'size':
                        mutated_genome[key] = np.clip(mutated_genome[key], 0.5, 2.0)
                    elif key == 'stamina':
                        mutated_genome[key] = np.clip(mutated_genome[key], 10.0, 200.0)
                    elif key == 'vision_range':
                        mutated_genome[key] = np.clip(mutated_genome[key], 3.0, 30.0)
                    elif key == 'food_preference':
                        mutated_genome[key] = np.clip(mutated_genome[key], 0.0, 1.0)
        
        return mutated_genome
    
    def evolve(self, population: List[NPC], spawn_points: List[Tuple[float, float, float]]) -> List[NPC]:
        """
        Evolve the population to create the next generation.
        
        Args:
            population: Current population of NPCs
            spawn_points: Spawn points for new generation
            
        Returns:
            New generation of NPCs
        """
        # Evaluate fitness
        fitness_scores = self.evaluate_fitness(population)
        
        # Select parents
        parents = self.select_parents(population, fitness_scores)
        
        # Create new generation
        new_population = []
        
        # Keep top 10% elite
        elite_count = max(1, int(self.population_size * 0.1))
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            elite_npc = population[idx]
            # Create new NPC with same genome
            if len(new_population) < len(spawn_points):
                x, y, z = spawn_points[len(new_population)]
            else:
                x, y, z = spawn_points[0]
            new_npc = NPC(x, y, z, elite_npc.genome.copy())
            new_population.append(new_npc)
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            child_genome = self.crossover(parent1, parent2)
            child_genome = self.mutate(child_genome)
            
            if len(new_population) < len(spawn_points):
                x, y, z = spawn_points[len(new_population)]
            else:
                x, y, z = spawn_points[0]
                x += np.random.uniform(-5, 5)
                z += np.random.uniform(-5, 5)
            
            new_npc = NPC(x, y, z, child_genome)
            new_population.append(new_npc)
        
        self.generation += 1
        return new_population

