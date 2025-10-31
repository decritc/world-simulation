"""Main entry point for the world simulation."""

import pyglet
from world_simulation.world.world import World
from world_simulation.rendering.renderer import Renderer
from world_simulation.genetics.evolution import EvolutionEngine
from world_simulation.trees.tree import FruitTree
import numpy as np


def main() -> None:
    """Run the world simulation."""
    print("Starting World Simulation...")
    
    # Create world
    world = World(seed=42)
    
    # Create renderer
    renderer = Renderer(world)
    
    # Create evolution engine
    evolution_engine = EvolutionEngine(population_size=30)
    
    # Initialize trees
    for i in range(20):
        x = np.random.uniform(-20, 20)
        z = np.random.uniform(-20, 20)
        y = world.get_height(x, z)
        tree = FruitTree(x, y, z)
        world.trees.append(tree)
    
    # Create initial population
    spawn_points = [(0, world.get_height(0, 0), 0)]
    for i in range(29):
        angle = i * 2 * np.pi / 29
        radius = 5.0
        x = np.cos(angle) * radius
        z = np.sin(angle) * radius
        y = world.get_height(x, z)
        spawn_points.append((x, y, z))
    
    world.entities = evolution_engine.create_initial_population(spawn_points)
    
    # Simulation variables
    generation_time = 0.0
    generation_length = 300.0  # seconds per generation
    
    # Main game loop
    def update(dt):
        nonlocal generation_time
        
        # Update world
        world.update(dt)
        
        # Update renderer
        renderer.update(dt)
        
        generation_time += dt
        
        # Check if generation should evolve
        alive_count = sum(1 for npc in world.entities if npc.is_alive)
        
        if generation_time >= generation_length or alive_count < 5:
            print(f"Generation {evolution_engine.generation} complete!")
            print(f"Survivors: {alive_count}/{len(world.entities)}")
            
            # Get best fitness
            fitness_scores = evolution_engine.evaluate_fitness(world.entities)
            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            print(f"Best fitness: {best_fitness:.2f}, Average fitness: {avg_fitness:.2f}")
            
            # Evolve
            world.entities = evolution_engine.evolve(world.entities, spawn_points)
            generation_time = 0.0
            
            # Reset camera to center
            renderer.camera_x = 0.0
            renderer.camera_z = 0.0
    
    # Schedule updates
    pyglet.clock.schedule_interval(update, 1/60.0)
    
    # Run the simulation
    pyglet.app.run()


if __name__ == "__main__":
    main()
