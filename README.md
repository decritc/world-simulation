# World Simulation

A 3D procedurally generated world simulation featuring AI NPCs evolved using genetic algorithms, neural networks, and generative AI.

## Features

### 🌍 World Generation
- **3D Procedurally Generated Terrain**: Infinite terrain generation using Perlin noise with smooth chunk transitions
- **Bilinear Interpolation**: Smooth terrain height lookups preventing floating islands and jagged hills
- **Multiple Terrain Types**: 
  - Water (dark blue valleys)
  - Sand (light sandy beaches)
  - Dirt (brown lowlands)
  - Grass (green grasslands)
  - Hills (darker green rolling hills)
  - Mountains (gray-brown peaks)
  - Snow (light gray/white mountain peaks)
- **Procedural Vegetation**: Bushes, grass patches, flowers, and rocks distributed based on terrain height and noise
- **Day/Night Cycle**: Dynamic lighting with smooth transitions between dawn, day, dusk, and night
- **Wildlife**: Animals (deer, rabbits, boars) that roam the world and can be hunted

### 🤖 AI NPCs
- **Neural Network-Based Decision Making**: PyTorch neural networks replace traditional state machines
  - 18 input features (health, hunger, stamina, age, environmental factors)
  - 64→64→32 hidden layer architecture
  - 6 action outputs (wander, seek_food, eat, rest, seek_shelter, stay_in_shelter)
- **Generative AI Integration**: Optional OpenAI GPT integration for complex reasoning
- **Genetic Algorithm Evolution**: NPCs evolve over generations with:
  - Neural network weight crossover and mutation
  - Fitness-based selection
  - Trait inheritance (speed, size, stamina, vision range)
- **Lifecycle System**: 
  - Age stages: Child → Adult → Elder
  - Aging mechanics with lifespan limits
  - Reproduction system (2 adults per house)
  - Name generation with inheritance
- **Survival Mechanics**:
  - Health, hunger, and stamina systems
  - Night exposure damage (health decreases faster outside at night)
  - Food gathering from fruit trees
  - Animal hunting (deer, rabbits, boars) for meat
  - Shelter seeking and house occupancy (2 adults per house)
- **Terrain-Aware Navigation**: NPCs automatically adjust height to match terrain, preventing floating

### 🌳 Fruit Trees
- **Growth System**: Trees grow from saplings to full maturity
- **Fruit Production**: Continuous fruit spawning with maturity stages
- **Reproduction**: Mature trees spawn new saplings nearby
- **Multiple Species**: Apple, orange, and cherry trees
- **Terrain Positioning**: Trees automatically placed on terrain height

### 🦌 Animals
- **Multiple Species**: Deer, rabbits, and boars with unique stats and behaviors
- **Animal Behavior**: 
  - Wandering when no threats nearby
  - Fleeing from NPCs when detected
  - Reproduction system to maintain population
- **Hunting System**: NPCs can hunt animals when hungry
  - Animals provide meat value based on species
  - NPCs track hunting statistics (`animals_hunted`)
  - Faster hunting when very hungry (hunger < 40)

### 🏠 Houses
- **Shelter System**: NPCs seek shelter during night
- **Capacity Management**: Maximum 2 adult NPCs per house
- **Reproduction**: Offspring spawn when 2 adults occupy a house
- **Building Time**: Houses require time to be built before use
- **Door Entry System**: NPCs approach and enter through visible doors
- **Varied Colors**: Each house has unique wall and roof colors for visual variety
- **Improved Model**: Houses feature detailed doors, frames, and colored roofs

### 📜 Colony History & Logging
- **Centralized Historian**: Single file (`colony_history.txt`) logs all NPC stories and colony events
- **Real-Time Logging**: Events logged as they happen during simulation
- **In-App Log Viewer**: View the colony history log directly in the application (press 'H' or use Menu → View History Log)
- **Scrollable Log Viewer**: Scroll through the entire history with mouse wheel, color-coded events (green=birth, red=death, orange=reproduction, blue=milestones, purple=achievements, yellow=summaries)
- **Menu System**: Press 'M' to open menu, navigate with UP/DOWN arrows, select with ENTER
- **Event Tracking**:
  - Birth events with parent information and generation numbers
  - Death events with cause, age, and lifetime achievements
  - Reproduction events tracking parent-offspring relationships
  - Milestones (reaching adulthood, first hunt, first fruit collected)
  - Achievements (fruit collection milestones, hunting milestones)
- **Generation Tracking**: 
  - Parent-child relationships with family trees
  - Generation numbers (Gen 0 = initial population, Gen 1+ = offspring)
  - Generation statistics (population size, average lifespan, reproduction rates)
- **Colony Summaries**: Periodic summaries (every 5 days) including:
  - Population growth rate
  - Average lifespan
  - Reproduction rate
  - Survival rate
  - Success/failure assessment based on multiple metrics
  - Overall colony status (SUCCESS/STABLE/FAILURE)
- **File Management**: Old log file automatically deleted when simulation starts, fresh file created each run

### 🎨 Rendering & Visualization
- **3D OpenGL Rendering**: High-quality 3D graphics with proper lighting
- **Level of Detail (LOD)**: Optimized rendering with distance-based detail levels
- **Fog Rendering**: Atmospheric fog to hide LOD transitions and distant artifacts
- **Frustum Culling**: Only render objects within camera view
- **Spatial Partitioning**: Efficient entity management with grid-based spatial organization
- **NPC Detail Panel**: Click any NPC to see:
  - Health, hunger, and stamina bars with labels
  - Neural network architecture visualization
  - Detailed status information (scrollable)
  - Current action and state
- **Debug Overlay**: Real-time FPS, statistics, and log viewer
- **Screenshot Capture**: Press 'F12' to capture screenshots or automatic capture when detail panel opens

### 🎮 Controls
- **WASD Movement**: Camera movement relative to current orientation
- **Mouse**: Click NPCs to view details
- **Mouse Wheel**: Scroll detail panel content or log viewer
- **F12 Key**: Capture screenshot
- **TAB Key**: Deselect NPC (close detail panel)
- **H Key**: Toggle history log viewer
- **M Key**: Toggle menu
- **UP/DOWN Keys**: Navigate menu (when menu is open)
- **ENTER Key**: Select menu item (when menu is open)
- **T Key**: Toggle debug color mode
- **P Key**: Toggle performance profiling

### 🧪 Testing
- Comprehensive test suite covering all features
- Unit tests for individual components
- Integration tests for system interactions
- Performance profiling capabilities

## Requirements

- Python 3.10+
- uv (Python package manager)
- PyTorch (for neural networks)
- OpenAI API key (optional, for generative AI features)

## Installation

```bash
# Install dependencies
uv sync

# Run the simulation
uv run world-simulation
```

## Project Structure

```
world_simulation/
├── world/          # World generation, terrain, and vegetation
│   ├── generator.py      # Terrain generation with noise algorithms
│   ├── world.py          # Main world management
│   └── vegetation.py     # Vegetation generation system
├── entities/       # NPCs and AI systems
│   ├── npc.py            # NPC implementation with neural networks
│   ├── animal.py          # Animal entities (deer, rabbit, boar)
│   ├── neural_network.py # PyTorch neural network for NPC decisions
│   ├── generative_ai.py  # OpenAI GPT integration
│   └── name_generator.py # Procedural name generation
├── genetics/       # Genetic algorithm implementation
│   └── evolution.py      # Evolution engine for NPC populations
├── trees/          # Fruit tree implementation
│   └── tree.py           # Tree growth, fruit production, reproduction
├── houses/         # House and shelter system
│   └── house.py          # House capacity and shelter mechanics
└── rendering/      # 3D rendering and visualization
    ├── renderer.py       # Main rendering engine
    ├── detail_panel.py    # NPC detail panel UI
    └── neural_network_viz.py # Neural network visualization
```

## Development

This project uses `uv` for dependency management. To add new dependencies:

```bash
uv add package-name
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_terrain_generation.py

# Run with verbose output
uv run pytest -v
```

## Performance Optimizations

- **Terrain Caching**: Chunk-based terrain caching with display lists (10-unit cache radius for seamless transitions)
- **Unified Terrain Grid**: Single continuous terrain grid eliminates gaps and fragmentation
- **LOD System**: Distance-based detail levels for trees and terrain
  - Trees: Full detail (< 20 units), Medium (20-50 units), Minimal (> 50 units)
  - Fruit rendering only for trees within 50 units
- **Distance Culling**: Only render objects within camera range
- **Batch Rendering**: Efficient OpenGL batch rendering with display lists
- **Spatial Partitioning**: Grid-based entity organization with cached grids
- **Frustum Culling**: Camera view-based culling
- **GPU Instancing**: Batched rendering for vegetation (bushes, grass, flowers, rocks) grouped by type

## Recent Improvements

- ✅ Smooth terrain generation with bilinear interpolation
- ✅ Fixed terrain gaps and fragmentation with unified grid system
- ✅ Fixed floating islands and NPCs
- ✅ Multiple terrain types with visual variety
- ✅ Neural network-based NPC decision making
- ✅ Generative AI integration for complex reasoning
- ✅ Refactored detail panel into separate modules
- ✅ Improved FPS with performance optimizations
- ✅ Enhanced terrain rendering with proper normals
- ✅ Animal hunting system with multiple species
- ✅ Distance-based LOD for tree rendering (trunk, foliage, fruit)
- ✅ Optimized terrain cache regeneration (10-unit radius)
- ✅ Seamless terrain rendering without gaps or holes
- ✅ GPU instancing for vegetation rendering (improved performance)
- ✅ Improved house models with doors, entry points, and multiple colors
- ✅ NPCs approach doors specifically when seeking shelter

## License

MIT