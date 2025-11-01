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
  - Shelter seeking and house occupancy (2 adults per house)
- **Terrain-Aware Navigation**: NPCs automatically adjust height to match terrain, preventing floating

### 🌳 Fruit Trees
- **Growth System**: Trees grow from saplings to full maturity
- **Fruit Production**: Continuous fruit spawning with maturity stages
- **Reproduction**: Mature trees spawn new saplings nearby
- **Multiple Species**: Apple, orange, and cherry trees
- **Terrain Positioning**: Trees automatically placed on terrain height

### 🏠 Houses
- **Shelter System**: NPCs seek shelter during night
- **Capacity Management**: Maximum 2 adult NPCs per house
- **Reproduction**: Offspring spawn when 2 adults occupy a house
- **Building Time**: Houses require time to be built before use

### 🎨 Rendering & Visualization
- **3D OpenGL Rendering**: High-quality 3D graphics with proper lighting
- **Level of Detail (LOD)**: Optimized rendering with distance-based detail levels
- **Frustum Culling**: Only render objects within camera view
- **Spatial Partitioning**: Efficient entity management with grid-based spatial organization
- **NPC Detail Panel**: Click any NPC to see:
  - Health, hunger, and stamina bars with labels
  - Neural network architecture visualization
  - Detailed status information (scrollable)
  - Current action and state
- **Debug Overlay**: Real-time FPS, statistics, and log viewer
- **Screenshot Capture**: Press 'S' to capture screenshots or automatic capture when detail panel opens

### 🎮 Controls
- **WASD Movement**: Camera movement relative to current orientation
- **Mouse**: Click NPCs to view details
- **Mouse Wheel**: Scroll detail panel content
- **S Key**: Capture screenshot

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

- **Terrain Caching**: Chunk-based terrain caching with display lists
- **LOD System**: Multiple detail levels based on distance
- **Distance Culling**: Only render objects within camera range
- **Batch Rendering**: Efficient OpenGL batch rendering
- **Spatial Partitioning**: Grid-based entity organization
- **Frustum Culling**: Camera view-based culling

## Recent Improvements

- ✅ Smooth terrain generation with bilinear interpolation
- ✅ Fixed floating islands and NPCs
- ✅ Multiple terrain types with visual variety
- ✅ Neural network-based NPC decision making
- ✅ Generative AI integration for complex reasoning
- ✅ Refactored detail panel into separate modules
- ✅ Improved FPS with performance optimizations
- ✅ Enhanced terrain rendering with proper normals

## License

MIT