# RL Franka Panda (MuJoCo)

Reinforcement learning environments and training scripts for robotic manipulation with the **Franka Panda** in **MuJoCo**, built on **Gymnasium** and **Stable-Baselines3**.

This project includes three custom tasks:
- Pick-and-Place
- Push
- Slide

Each task supports:
- Sparse reward (`Sparse` envs)
- Dense reward (`Dense` envs)

## Highlights

- Custom Gymnasium robotics environments under `panda_mujoco_gym`
- MuJoCo-based simulation for Panda manipulation
- Training scripts for PPO, SAC, and TQC
- Dense reward shaping with tunable components
- Episode tracking and trajectory statistics in environment `info`
- Evaluation scripts with MP4 video recording
- Docker + Docker Compose for reproducible runs

## Important Algorithm Findings

For **Pick-and-Place + Dense reward** (`FrankaPickAndPlaceDense-v0`):

- ❌ **PPO**: training unsuccessful
- ❌ **SAC**: training unstable with catastrophic forgetting
- ✅ **TQC**: converges reliably

**Recommendation:** use **TQC** for dense Pick-and-Place in this repository.

## Table of Contents

- [RL Franka Panda (MuJoCo)](#rl-franka-panda-mujoco)
  - [Highlights](#highlights)
  - [Important Algorithm Findings](#important-algorithm-findings)
  - [Table of Contents](#table-of-contents)
  - [Environment IDs](#environment-ids)
  - [Project Structure](#project-structure)
  - [Prerequisites](#prerequisites)
  - [Installation (Local)](#installation-local)
  - [Installation (Docker)](#installation-docker)
    - [Build image and create container](#build-image-and-create-container)
    - [Enter container](#enter-container)
  - [Quick Start](#quick-start)
    - [1) Random interaction test](#1-random-interaction-test)
    - [2) Train recommended model for dense Pick-and-Place](#2-train-recommended-model-for-dense-pick-and-place)
    - [3) Evaluate and record video](#3-evaluate-and-record-video)
  - [Tasks](#tasks)
    - [Pick-and-Place](#pick-and-place)
    - [Push](#push)
    - [Slide](#slide)
  - [Reward Types](#reward-types)
    - [Sparse reward](#sparse-reward)
    - [Dense reward](#dense-reward)
  - [Dense Reward Hyperparameters](#dense-reward-hyperparameters)
  - [Training](#training)
    - [PPO](#ppo)
    - [SAC](#sac)
    - [TQC](#tqc)
    - [TensorBoard](#tensorboard)
  - [Algorithm Performance Summary](#algorithm-performance-summary)
  - [Evaluation and Video Recording](#evaluation-and-video-recording)
    - [Evaluate SAC model with video export](#evaluate-sac-model-with-video-export)
    - [Evaluate TQC model with video export](#evaluate-tqc-model-with-video-export)
  - [Model Saving and Loading](#model-saving-and-loading)
    - [Saving (already in scripts)](#saving-already-in-scripts)
    - [Loading example](#loading-example)
  - [Known Issues](#known-issues)
  - [Troubleshooting](#troubleshooting)
    - [`ModuleNotFoundError: panda_mujoco_gym`](#modulenotfounderror-panda_mujoco_gym)
    - [MuJoCo/OpenGL rendering errors](#mujocoopengl-rendering-errors)
    - [No video output](#no-video-output)
    - [Training appears unstable](#training-appears-unstable)
  - [Future Work](#future-work)
  - [Citation](#citation)
  - [Acknowledgments](#acknowledgments)

## Environment IDs

All environments are registered automatically in `panda_mujoco_gym/__init__.py`:

- `FrankaPickAndPlaceSparse-v0`
- `FrankaPickAndPlaceDense-v0`
- `FrankaPushSparse-v0`
- `FrankaPushDense-v0`
- `FrankaSlideSparse-v0`
- `FrankaSlideDense-v0`

## Project Structure

```text
RL-Franka-Panda/
├── panda_mujoco_gym/
│   ├── __init__.py
│   ├── assets/
│   │   ├── pick_and_place.xml
│   │   ├── push.xml
│   │   ├── slide.xml
│   │   └── panda_mocap.xml
│   └── envs/
│       ├── panda_env.py
│       ├── pick_and_place.py
│       ├── push.py
│       └── slide.py
├── PPO_pick_and_place_train.py
├── PPO_push_train.py
├── PPO_slide_train.py
├── SAC_pick_and_place_train.py
├── TQC_pick_and_place_train.py
├── SAC_pick_and_place_play.py
├── TQC_pick_and_place_play.py
├── model/
├── logs/
├── videos/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Prerequisites

- Python 3.10+
- MuJoCo runtime dependencies (OpenGL/GLFW on Linux)
- Optional: NVIDIA GPU + Docker + NVIDIA Container Toolkit

## Installation (Local)

1. Clone and enter the repo.

```bash
git clone https://github.com/yichuan-huang/RL-Franka-Panda.git
cd RL-Franka-Panda
```

2. Create and activate a virtual environment.

```bash
conda create -n rl-franka-panda python=3.10
conda activate rl-franka-panda
```

3. Install dependencies.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Installation (Docker)

### Build image and create container

```bash
docker-compose up -d
```

### Enter container

```bash
docker exec -it rl-franka-panda /bin/bash
```

Inside the container, run training/evaluation commands from `/workspace`.

Notes:
- `docker-compose.yml` requests `gpus: all`
- If GPU is unavailable, remove GPU-specific settings and run CPU-only

## Quick Start

### 1) Random interaction test

```python
import time
import gymnasium as gym
import panda_mujoco_gym

env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")
obs, info = env.reset()

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
    time.sleep(0.02)

env.close()
```

### 2) Train recommended model for dense Pick-and-Place

```bash
python TQC_pick_and_place_train.py
```

### 3) Evaluate and record video

```bash
python TQC_pick_and_place_play.py
```

Recorded video is saved under `videos/`.

## Tasks

### Pick-and-Place

- Environment classes: `FrankaPickAndPlaceEnv`
- IDs: `FrankaPickAndPlaceSparse-v0`, `FrankaPickAndPlaceDense-v0`
- Gripper: enabled (`block_gripper=False`)
- Goal: reach 3D target with object, includes lifting behavior

### Push

- Environment class: `FrankaPushEnv`
- IDs: `FrankaPushSparse-v0`, `FrankaPushDense-v0`
- Gripper: blocked (`block_gripper=True`)
- Goal: push object on table plane to target

### Slide

- Environment class: `FrankaSlideEnv`
- IDs: `FrankaSlideSparse-v0`, `FrankaSlideDense-v0`
- Gripper: blocked (`block_gripper=True`)
- Goal: slide object to target (different x-offset task geometry)

## Reward Types

### Sparse reward

- Success: `+1.0`
- Otherwise: `-1.0`
- Success condition: object-goal distance below threshold (`distance_threshold=0.05` by default)

### Dense reward

Dense reward is a weighted sum of shaping terms from `panda_mujoco_gym/envs/panda_env.py`:

- Object-to-goal distance shaping
- End-effector-to-object distance shaping
- Lift progress shaping
- Optional gripper-width shaping near object
- Optional smoothness/action penalties
- Success bonus

Dense reward is configurable through hyperparameters described below.

## Dense Reward Hyperparameters

Key dense reward parameters (defined in `FrankaEnv.__init__`):

- `w_obj_goal`: object-to-goal shaping weight
- `w_ee_obj`: end-effector-to-object shaping weight
- `w_lift`: lift progress shaping weight
- `w_gripper`: gripper shaping weight
- `w_action`: action magnitude penalty
- `w_action_change`: action delta penalty
- `w_smooth`: velocity smoothness penalty
- `success_reward`: success bonus for dense reward
- `terminal_bonus`: optional extra bonus on terminal success
- `gripper_activation_distance`: activate gripper shaping near object
- `place_activation_height`: gate object-goal shaping until object is lifted
- `success_activation_height`: require minimum lift for success
- `gripper_target_width`: target width for gripper shaping
- `pos_ctrl_scale`: end-effector position control scale

Pick-and-Place currently overrides several defaults in `panda_mujoco_gym/envs/pick_and_place.py`, e.g.:
- `w_obj_goal=2.0`
- `w_ee_obj=1.0`
- `w_lift=5.0`
- `w_gripper=0.2`

## Training

All provided training scripts run for `1000000` timesteps and save models under `model/`.

### PPO

```bash
python PPO_pick_and_place_train.py
python PPO_push_train.py
python PPO_slide_train.py
```

### SAC

```bash
python SAC_pick_and_place_train.py
```

### TQC

```bash
python TQC_pick_and_place_train.py
```

### TensorBoard

```bash
tensorboard --logdir <logs_dir>
```

## Algorithm Performance Summary

| Task | Reward | PPO | SAC | TQC | Recommended |
|---|---|---|---|---|---|
| Pick-and-Place | Dense | ❌ Unsuccessful | ❌ Catastrophic forgetting | ✅ Converges | ✅ TQC |
| Pick-and-Place | Sparse | ⚠️ Not fully benchmarked in this repo | ⚠️ Planned with HER | ⚠️ Planned with HER | ⚠️ TQC/SAC + HER (future work) |
| Push | Sparse/Dense | ⚠️ Script available, benchmark incomplete | ⚠️ Not benchmarked | ⚠️ Not benchmarked | ⚠️ Evaluate per setup |
| Slide | Sparse/Dense | ⚠️ Script available, benchmark incomplete | ⚠️ Not benchmarked | ⚠️ Not benchmarked | ⚠️ Evaluate per setup |

For users focused on robust Pick-and-Place dense training, start with **TQC**.

## Evaluation and Video Recording

### Evaluate SAC model with video export

```bash
python SAC_pick_and_place_play.py
```

### Evaluate TQC model with video export

```bash
python TQC_pick_and_place_play.py
```

Both scripts:
- Load model from `model/*.zip`
- Run multiple episodes (`episodes = 50`)
- Capture RGB frames from `render_mode="rgb_array"`
- Save combined MP4 to `videos/`
- Support playback slowdown via frame duplication (`slowdown_factor`)

If file paths differ from your setup, edit:
- `env_id`
- `model_path`
- `video_filename`

## Model Saving and Loading

### Saving (already in scripts)

- PPO: `model/PPO_*.zip`
- SAC: `model/SAC_pick_and_place.zip`
- TQC: `model/TQC_pick_and_place.zip`

### Loading example

```python
import gymnasium as gym
import panda_mujoco_gym
from sb3_contrib import TQC

env = gym.make("FrankaPickAndPlaceDense-v0", render_mode="rgb_array")
model = TQC.load("model/TQC_pick_and_place.zip")

obs, info = env.reset()
for _ in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Known Issues

- SAC on dense Pick-and-Place can exhibit catastrophic forgetting during long training.
- PPO is not reliable for dense Pick-and-Place in current setup.
- Some scripts assume directories already exist (`model/`, `logs/`, `videos/`). Create them if missing.
- Rendering may fail on headless systems without proper OpenGL/EGL/OSMesa setup.

## Troubleshooting

### `ModuleNotFoundError: panda_mujoco_gym`

Run commands from repo root, or add repo to `PYTHONPATH`:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### MuJoCo/OpenGL rendering errors

- Install required system GL libraries
- Use Docker setup provided in this repo
- For headless training, avoid `render_mode="human"`

### No video output

- Ensure `ffmpeg` is installed
- Confirm write permission to `videos/`
- Check that evaluation script uses `render_mode="rgb_array"`

### Training appears unstable

- Prefer `TQC_pick_and_place_train.py` for dense Pick-and-Place
- Revisit dense reward weights (`w_obj_goal`, `w_ee_obj`, `w_lift`, etc.)
- Increase training timesteps and monitor TensorBoard curves

## Future Work

- Implement and benchmark **SAC + HER** for sparse / near-sparse Pick-and-Place
- Implement and benchmark **TQC + HER** for sparse / near-sparse Pick-and-Place
- Compare sample efficiency of HER vs dense shaping
- Expand benchmark matrix across Push and Slide with standardized seeds and metrics

## Citation

If you use this repository in academic work, cite:

```bibtex
@misc{xu2023opensource,
  title={Open-Source Reinforcement Learning Environments Implemented in MuJoCo with Franka Manipulator},
  author={Zichun Xu and Yuntao Li and Xiaohang Yang and Zhiyuan Zhao and Lei Zhuang and Jingdong Zhao},
  year={2023},
  eprint={2312.13788},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/)
- [Gymnasium Robotics](https://robotics.farama.org/)
- [MuJoCo](https://mujoco.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [SB3-Contrib](https://sb3-contrib.readthedocs.io/)
