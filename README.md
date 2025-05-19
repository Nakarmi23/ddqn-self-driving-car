# Self-Driving Car Simulation with Deep Q-Learning

This project implements a self-driving car simulation using Deep Q-Learning (DQN) reinforcement learning algorithm. The car learns to navigate a custom race track while avoiding collisions and maximizing its score through checkpoints.

## Features

- Realistic car physics simulation with drift mechanics
- Custom race track with checkpoints and scoring system
- Deep Q-Learning (DQN) implementation for autonomous driving
- Manual control mode with keyboard inputs
- Dynamic scoring system with combo multipliers
- Visual feedback for collisions and checkpoint completion

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`:
  - numpy >= 2.0.2
  - tensorflow >= 2.19.0
  - pygame >= 2.6.1

## Installation

1. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

3. Run the simulation:

```bash
python main.py
```

### Game Features

- **Scoring System**: Earn points by passing checkpoints
- **Combo System**: Maintain speed and consecutive checkpoints to increase score multiplier
- **Collision Detection**: Realistic collision detection with walls
- **Visual Feedback**: Speed indicators and combo displays
- **AI Training**: The car learns from experience using DQN algorithm

## Project Structure

```
.
├── ai/
│   └── ddqn_agent.py      # Deep Q-Network implementation
├── game/
│   ├── assets/           # Game assets (car image, etc.)
│   ├── game_car.py       # Car physics and controls
│   ├── game_checkpoint.py # Checkpoint system
│   ├── game_grid.py      # Spatial grid for collision detection
│   ├── game_intersection.py # Intersection detection
│   ├── game_point.py     # Point class for coordinates
│   ├── game_track.py     # Track management
│   ├── game_wall.py      # Wall implementation
│   └── track_templates/  # Track layouts
├── main.py              # Main game loop
└── requirements.txt     # Project dependencies
```

## AI Implementation

The project uses a Double Deep Q-Network (DDQN) for the AI agent, which includes:

- Experience replay buffer
- Target network for stable learning
- Epsilon-greedy exploration strategy
- State representation including car position, velocity, and checkpoint information
