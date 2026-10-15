# Exo Elbow Sim

MuJoCo simulation of our single-DOF elbow exoskeleton: a motor at the elbow, triggered by an EMG sensor on the biceps. The sim is a Gymnasium environment, so we can train RL controllers on it with Stable-Baselines3.

## Setup

You need Python 3.10 or newer. Using a separate environment keeps these packages away from your other projects.

With conda:

```
conda create -n exo python=3.12 -c conda-forge
conda activate exo
pip install -r requirements.txt
```

Without conda:

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

`pip` also installs PyTorch (CPU build) as part of Stable-Baselines3. It's a large download, so the first install takes a few minutes.

In VS Code, run **Python: Select Interpreter** and pick the environment you just created.

## Running

From this folder:

```
python run_sim.py              # compare the baseline controllers (no window)
python run_sim.py --viewer     # watch the arm in the 3D viewer
python run_sim.py --viewer --policy zero    # no assistance: user lifts alone
```

Policies: `zero`, `random`, `emg_prop`, `emg_gravcomp`. On macOS, use `mjpython run_sim.py --viewer` for the viewer.

Expected output of `python run_sim.py`:

```
policy            return  RMS err (deg)  human drive  |motor| N*m
zero             -1074.7           46.6         0.84         0.00
random           -1132.7           46.3         0.80         3.02
emg_prop          -145.8           10.9         0.38         2.29
emg_gravcomp      -155.1           19.4         0.13         2.86
```

The `random` row may differ slightly; the others should match.

## Files

| File | What it is |
|---|---|
| `exo_arm.xml` | MuJoCo model: fixed upper arm, forearm on an elbow hinge, the human's biceps, and the exo motor |
| `exo_env.py` | `ExoElbowEnv`, the Gymnasium environment: simulated user intent, biceps control, EMG sensor model, reward |
| `run_sim.py` | Runs baseline controllers and prints a comparison, or shows one in the viewer |

## How the sim works

Each episode is 10 s. The control loop runs at 100 Hz; physics runs at 500 Hz.

- **The user** wants to move their elbow to a target angle that changes every few seconds (curls, holds, lowering). The target is hidden from the controller. The user drives their biceps to follow it, but they're deliberately weak: their biceps maxes out at 2 N·m and holding the forearm level takes about 3.1 N·m, so they can't lift their arm unassisted.
- **The EMG sensor** sees the user's biceps drive with noise, rectification, a 5 Hz smoothing filter and a 20 ms delay, roughly like a MyoWare-style board.
- **The controller** (the RL agent) commands the exo motor: an action in [-1, 1] maps to ±6 N·m at the elbow.
  - It observes the elbow angle, the elbow velocity, the last 5 EMG samples and its own previous action.
  - Its reward penalizes missing the user's target angle, the user's effort, motor torque, and jerky commands.
- **Randomized each episode:** a 0–0.8 kg weight in the hand, the user's strength (60–120%) and the EMG gain (60–140%), so a policy can't overfit to one exact setup.

Conventions: angles are in radians. An elbow angle of 0 means the arm is fully extended (hanging straight down), π/2 means the forearm is level, and positive is flexion.

## Using the environment

```python
from exo_env import ExoElbowEnv

env = ExoElbowEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step([0.5])   # 3 N*m of assist
```

`info` includes the hidden target angle, the user's biceps drive and the motor torque, which is useful for plotting. Constructor arguments set the control rate, episode length, reward weights and the user model; `randomize=False` turns off the per-episode randomization.

## Placeholders to replace with our hardware

The masses, motor torque limit, joint friction, EMG filter and delay, and control rate are reasonable guesses, not measurements. They live in `exo_arm.xml` (physical parameters) and in `ExoElbowEnv.__init__` / `EMGSensor` (control rate, sensor model). Update them as we measure the real device.
