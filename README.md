# World Simulation

A 3D procedurally generated world simulation featuring AI NPCs evolved using genetic algorithms.

## Features

- **3D Procedurally Generated World**: Infinite terrain generation using noise algorithms
- **Fruit Trees**: Trees that grow and produce fruit throughout the simulation
- **AI NPCs**: Intelligent agents that evolve using genetic algorithms
- **Genetic Algorithm Evolution**: NPCs adapt and improve over generations

## Requirements

- Python 3.10+
- uv (Python package manager)

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
├── world/          # World generation and management
├── entities/       # NPCs and other entities
├── genetics/       # Genetic algorithm implementation
├── trees/          # Fruit tree implementation
└── rendering/      # 3D rendering and visualization
```

## Development

This project uses `uv` for dependency management. To add new dependencies:

```bash
uv add package-name
```

## License

MIT

