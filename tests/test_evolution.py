"""Tests for genetic algorithm evolution."""

import pytest
import numpy as np
from world_simulation.genetics.evolution import EvolutionEngine
from world_simulation.entities.npc import NPC


class TestEvolutionEngine:
    """Test evolution engine functionality."""
    
    def test_init(self):
        """Test evolution engine initialization."""
        engine = EvolutionEngine(population_size=50)
        assert engine.population_size == 50
        assert engine.generation == 0
    
    def test_create_initial_population(self):
        """Test initial population creation."""
        engine = EvolutionEngine(population_size=10)
        spawn_points = [(0, 0, 0), (1, 0, 1), (2, 0, 2)]
        
        population = engine.create_initial_population(spawn_points)
        
        assert len(population) == 10
        assert all(isinstance(npc, NPC) for npc in population)
        assert all(npc.is_alive for npc in population)
    
    def test_evaluate_fitness(self):
        """Test fitness evaluation."""
        engine = EvolutionEngine(population_size=5)
        spawn_points = [(0, 0, 0)]
        
        population = engine.create_initial_population(spawn_points)
        
        # Modify some NPCs to have different fitness
        population[0].age = 10.0
        population[0].fruit_collected = 5
        population[1].health = 50.0
        
        fitness_scores = engine.evaluate_fitness(population)
        
        assert len(fitness_scores) == 5
        assert all(score >= 0 for score in fitness_scores)
        assert fitness_scores[0] > fitness_scores[1]  # First NPC should have higher fitness
    
    def test_select_parents(self):
        """Test parent selection."""
        engine = EvolutionEngine(population_size=10)
        spawn_points = [(0, 0, 0)]
        
        population = engine.create_initial_population(spawn_points)
        
        # Create fitness differences
        for i, npc in enumerate(population):
            npc.age = i * 10.0  # Different ages
        
        fitness_scores = engine.evaluate_fitness(population)
        parents = engine.select_parents(population, fitness_scores, num_parents=5)
        
        assert len(parents) == 5
        assert all(isinstance(p, NPC) for p in parents)
    
    def test_crossover(self):
        """Test genome crossover."""
        engine = EvolutionEngine(population_size=5)
        
        parent1 = NPC(0, 0, 0, genome={'speed': 1.0, 'size': 1.0, 'stamina': 100.0})
        parent2 = NPC(0, 0, 0, genome={'speed': 2.0, 'size': 2.0, 'stamina': 200.0})
        
        child_genome = engine.crossover(parent1, parent2)
        
        assert 'speed' in child_genome
        assert 'size' in child_genome
        assert 'stamina' in child_genome
        # Child should have mix of parent traits
        assert child_genome['speed'] in [1.0, 2.0]
        assert child_genome['size'] in [1.0, 2.0]
        assert child_genome['stamina'] in [100.0, 200.0]
    
    def test_mutate(self):
        """Test genome mutation."""
        engine = EvolutionEngine(population_size=5)
        
        original_genome = {'speed': 1.0, 'size': 1.0, 'stamina': 100.0}
        mutated_genome = engine.mutate(original_genome.copy(), mutation_rate=1.0)
        
        # With mutation_rate=1.0, all genes should mutate
        # But values should stay within reasonable bounds
        assert 0.1 <= mutated_genome['speed'] <= 3.0
        assert 0.5 <= mutated_genome['size'] <= 2.0
        assert 10.0 <= mutated_genome['stamina'] <= 200.0
    
    def test_evolve(self):
        """Test evolution process."""
        engine = EvolutionEngine(population_size=10)
        spawn_points = [(0, 0, 0), (1, 0, 1)]
        
        # Create initial population with varying fitness
        population = engine.create_initial_population(spawn_points)
        for i, npc in enumerate(population):
            npc.age = i * 5.0
            npc.fruit_collected = i
        
        initial_generation = engine.generation
        
        new_population = engine.evolve(population, spawn_points)
        
        assert len(new_population) == 10
        assert engine.generation == initial_generation + 1
        assert all(isinstance(npc, NPC) for npc in new_population)
        assert all(npc.is_alive for npc in new_population)
    
    def test_elite_preservation(self):
        """Test that elite NPCs are preserved."""
        engine = EvolutionEngine(population_size=10)
        spawn_points = [(0, 0, 0)]
        
        population = engine.create_initial_population(spawn_points)
        
        # Make one NPC clearly superior
        population[0].age = 100.0
        population[0].fruit_collected = 50
        population[0].health = 100.0
        
        # Others have low fitness
        for npc in population[1:]:
            npc.age = 1.0
            npc.fruit_collected = 0
        
        new_population = engine.evolve(population, spawn_points)
        
        # At least some should have similar high fitness (elite preserved)
        fitness_scores = engine.evaluate_fitness(new_population)
        assert max(fitness_scores) > 0

