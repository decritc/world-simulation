"""Main entry point for the world simulation."""

import pyglet
from world_simulation.world.world import World
from world_simulation.rendering.renderer import Renderer
from world_simulation.genetics.evolution import EvolutionEngine
from world_simulation.trees.tree import FruitTree
from world_simulation.houses.house import House
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
    
    # Initialize trees with different species
    tree_species = ["apple", "orange", "cherry"]
    for i in range(20):
        x = np.random.uniform(-20, 20)
        z = np.random.uniform(-20, 20)
        y = world.get_height(x, z)
        species = np.random.choice(tree_species)
        tree = FruitTree(x, y, z, species=species)
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
    
    # Make initial population adults for faster start
    for npc in world.entities:
        if npc.age < npc.adult_age:
            npc.age = npc.adult_age + np.random.uniform(0, 60.0)  # Start as adults
            npc.age_stage = "adult"
            npc.can_reproduce = True
            npc.size = npc.genome.get('size', 1.0)  # Full size as adult
    
    # Initialize houses (capacity 2 for adults only)
    for i in range(5):
        angle = i * 2 * np.pi / 5
        radius = 15.0
        x = np.cos(angle) * radius
        z = np.sin(angle) * radius
        y = world.get_height(x, z)
        house = House(x, y, z, capacity=2)  # Only 2 adults per house
        world.houses.append(house)
    
    renderer.log("World initialized with houses and trees")
    
    # Simulation variables
    generation_time = 0.0
    generation_length = 300.0  # seconds per generation
    last_day = 0
    
    # Main game loop
    def update(dt):
        nonlocal generation_time, last_day
        
        # Update world
        world.update(dt)
        
        # Update renderer
        renderer.update(dt)
        
        generation_time += dt
        
        # Log day changes
        if world.day_number != last_day:
            renderer.log(f"Day {world.day_number + 1} has begun!")
            last_day = world.day_number
        
        # Log day/night transitions
        hour = (world.day_time / world.day_length) * 24.0
        if abs(hour - 6.0) < 0.1 or abs(hour - 18.0) < 0.1:
            if world.is_night():
                renderer.log("Night falls - NPCs seek shelter")
            else:
                renderer.log("Dawn breaks - NPCs leave shelter")
        
        # Check if generation should evolve
        alive_count = sum(1 for npc in world.entities if npc.is_alive)
        
        if generation_time >= generation_length or alive_count < 5:
            print(f"Generation {evolution_engine.generation} complete!")
            print(f"Survivors: {alive_count}/{len(world.entities)}")
            
            renderer.log(f"Generation {evolution_engine.generation} complete!")
            renderer.log(f"Survivors: {alive_count}/{len(world.entities)}")
            
            # Get best fitness
            fitness_scores = evolution_engine.evaluate_fitness(world.entities)
            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            print(f"Best fitness: {best_fitness:.2f}, Average fitness: {avg_fitness:.2f}")
            
            renderer.log(f"Best fitness: {best_fitness:.2f}, Avg: {avg_fitness:.2f}")
            
            # Evolve
            world.entities = evolution_engine.evolve(world.entities, spawn_points)
            generation_time = 0.0
            
            renderer.log("New generation spawned!")
            
            # Reset camera to center
            renderer.camera_x = 0.0
            renderer.camera_z = 0.0
    
    # Schedule updates
    pyglet.clock.schedule_interval(update, 1/60.0)
    
    # Run the simulation
    pyglet.app.run()


if __name__ == "__main__":
    main()
