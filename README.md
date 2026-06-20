# Door Robot

A TD3 (Twin Delayed DDPG) reinforcement learning agent that trains a Panda robot arm to open a door, using the [robosuite](https://robosuite.ai/) `Door` environment.

## Overview

- `td3_torch.py` — TD3 agent: twin critics, delayed/soft target updates, action smoothing noise, and checkpoint save/load.
- `networks.py` — Actor and critic MLPs (PyTorch).
- `buffer.py` — Fixed-size replay buffer for experience storage and sampling.
- `main.py` — Training loop: builds the `Door` env with a `Panda` robot under joint-velocity control, trains for up to 10,000 episodes, logs scores to TensorBoard, and periodically saves model checkpoints to `tmp/td3`.
- `test.py` — Loads saved checkpoints and runs a few episodes with rendering for visual evaluation (no learning, no exploration noise).

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `robosuite`, `gym`, `pybullet`/`mujoco`, `tensorboard`.

## Usage

### Train

```bash
python main.py
```

Checkpoints are saved to `tmp/td3/`, and training scores are logged for TensorBoard:

```bash
tensorboard --logdir tmp/td3
```

### Evaluate

Run a trained policy with rendering enabled:

```bash
python test.py
```

This loads checkpoints from `tmp/td3/` and runs 3 episodes with no exploration noise.

## Notes

- The environment uses joint-velocity control with reward shaping enabled and a 300-step horizon at 20 Hz control frequency.
- `main.py` calls `agent.load_models()` at startup, so training resumes from `tmp/td3/` if checkpoints already exist there.
