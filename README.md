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

- Python 3.9+ (any CPython 3.9-3.13 works with current PyTorch wheels)
- Dependencies listed in `requirements.txt`:
  - numpy >= 2.0.2
  - torch >= 2.8.0
  - pygame >= 2.6.1

## Installation

1. Create and activate a virtual environment:

```bash
# On Windows:
py -3.12 -m venv .venv
.venv\Scripts\activate
# On macOS/Linux:
python3.12 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

This installs the CPU build of PyTorch everywhere, which already includes
GPU support on Apple Silicon Macs (via MPS) with no extra steps. On Windows
or Linux with an NVIDIA GPU, reinstall `torch` from the CUDA wheel index to
enable GPU acceleration instead of CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

(Pick the CUDA tag matching your driver from https://pytorch.org/get-started/locally/
if `cu128` doesn't apply.) The agent auto-detects the best available device
at startup (CUDA > MPS > CPU) — no code changes needed either way.

## Usage

3. Run the simulation:

```bash
python main.py
```

By default the window draws every step at 60 FPS so you can watch. To train faster, set one of these environment variables:

| Variable | Effect | Throughput (CUDA, template5) |
|---|---|---|
| _(none)_ | Draw every step, capped at 60 FPS | ~60 steps/s |
| `RENDER_EVERY=10` | Draw every 10th step, no frame cap | ~390 steps/s |
| `HEADLESS=1` | No window, no drawing, no frame cap | ~435 steps/s |

```bash
HEADLESS=1 python main.py           # bash
$env:HEADLESS=1; python main.py     # PowerShell
```

In headless mode there is no window to close, so stop with Ctrl+C. Training resumes from the last checkpoint (saved every 10 episodes), so anything since then is lost.

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
├── scripts/
│   └── check_spawns.py   # Checks every track's spawn is valid
└── requirements.txt     # Project dependencies
```

## AI Implementation

The project uses a Double Deep Q-Network (DDQN) for the AI agent, which includes:

- Experience replay buffer
- Target network for stable learning
- Epsilon-greedy exploration strategy
- State representation including car position, velocity, and checkpoint information
