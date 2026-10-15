"""Single-DOF elbow exoskeleton environment (MuJoCo).

The simulation has three pieces:

  1. Human "intent": a hidden target elbow angle that the user wants to reach
     (random curls / holds / lowering, min-jerk transitions).
  2. Human motor control: a weak PD + partial gravity-feedforward controller
     that turns intent into biceps neural drive u_h in [0, 1]. The biceps alone
     cannot hold the arm horizontal (max 2 N*m vs ~3.1 N*m gravity load), so
     the user genuinely needs assistance.
  3. EMG sensor: observes u_h through a noisy carrier, rectification,
     low-pass envelope filter, gain variation and sensor delay -- roughly what
     a MyoWare-style board gives you on an ADC pin.

The RL agent commands the elbow motor (normalized torque in [-1, 1]) and sees
only what the real controller will see: elbow angle, velocity, recent EMG
envelope samples, and its previous action. The hidden intent is used only for
the reward (privileged information available in simulation).

API follows Gymnasium (reset/step). If gymnasium is installed the class is a
real gymnasium.Env with observation/action spaces, so it plugs directly into
Stable-Baselines3.
"""

import os
from collections import deque

import numpy as np
import mujoco
import mujoco.viewer

try:
    import gymnasium as gym
    from gymnasium import spaces

    _Base = gym.Env
except ImportError:  # still usable without gymnasium
    gym = None
    spaces = None
    _Base = object


XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exo_arm.xml")


class IntentGenerator:
    """Random sequence of target elbow angles with min-jerk transitions."""

    def __init__(self, rng, q_min=0.15, q_max=2.2, hold=(0.8, 2.5), move=(0.6, 1.5)):
        self.rng = rng
        self.q_min, self.q_max = q_min, q_max
        self.hold, self.move = hold, move

    def reset(self, q0):
        self.q_start = q0
        self.q_goal = q0
        self.t0 = 0.0
        self.T_move = 0.0
        self.t_next = self.rng.uniform(*self.hold)

    def __call__(self, t):
        if t >= self.t_next:
            self.q_start = self.target(t)
            self.q_goal = self.rng.uniform(self.q_min, self.q_max)
            self.t0 = t
            self.T_move = self.rng.uniform(*self.move)
            self.t_next = t + self.T_move + self.rng.uniform(*self.hold)
        return self.target(t)

    def target(self, t):
        if self.T_move <= 0.0:
            return self.q_goal
        s = np.clip((t - self.t0) / self.T_move, 0.0, 1.0)
        s = 10 * s**3 - 15 * s**4 + 6 * s**5  # min-jerk profile
        return self.q_start + (self.q_goal - self.q_start) * s


class EMGSensor:
    """Neural drive -> raw surface EMG -> rectified, low-pass envelope."""

    def __init__(self, rng, dt, cutoff_hz=5.0, noise_std=0.02, delay_s=0.02):
        self.rng = rng
        self.dt = dt
        self.alpha = 1.0 - np.exp(-2 * np.pi * cutoff_hz * dt)  # one-pole LPF
        self.noise_std = noise_std
        self.delay_steps = max(1, int(round(delay_s / dt)))
        self.gain = 1.0

    def reset(self, gain=1.0):
        self.gain = gain
        self.env = 0.0
        self.buf = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)

    def step(self, drive):
        # raw EMG is roughly zero-mean noise whose amplitude scales with drive
        raw = self.gain * drive * self.rng.standard_normal() + self.noise_std * self.rng.standard_normal()
        # full-wave rectify; sqrt(pi/2) makes the envelope mean ~= gain * drive
        rect = abs(raw) * np.sqrt(np.pi / 2)
        self.env += self.alpha * (rect - self.env)
        self.buf.append(self.env)
        return self.buf[0]  # delayed output


class ExoElbowEnv(_Base):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        xml_path=XML_PATH,
        frame_skip=5,            # 0.002 s physics * 5 = 100 Hz control (match your MCU loop)
        episode_seconds=10.0,
        emg_history=5,           # control steps of EMG envelope in the observation
        randomize=True,          # domain randomization (payload, EMG gain, human strength)
        # human motor-control model
        human_kp=4.0,
        human_kd=0.4,
        human_ff=0.6,            # fraction of own gravity load the user feed-forwards
        # reward weights
        w_track=1.0,
        w_effort=0.5,
        w_motor=0.05,
        w_smooth=0.1,
        render_mode=None,
        seed=None,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip
        self.max_steps = int(round(episode_seconds / self.dt))
        self.emg_history = emg_history
        self.randomize = randomize
        self.human_kp, self.human_kd, self.human_ff = human_kp, human_kd, human_ff
        self.w = dict(track=w_track, effort=w_effort, motor=w_motor, smooth=w_smooth)
        self.render_mode = render_mode
        self._viewer = None
        self._renderer = None

        m = self.model
        self.jnt = m.joint("elbow").id
        self.qadr = m.jnt_qposadr[self.jnt]
        self.dadr = m.jnt_dofadr[self.jnt]
        self.act_biceps = m.actuator("biceps").id
        self.act_motor = m.actuator("exo_motor").id
        self.forearm_body = m.body("forearm").id
        self.q_lo, self.q_hi = m.jnt_range[self.jnt]

        # nominal values for domain randomization
        self._nom_mass = m.body_mass[self.forearm_body]
        self._nom_ipos = m.body_ipos[self.forearm_body].copy()
        self._nom_biceps_gain = m.actuator_gainprm[self.act_biceps, 0]
        self.motor_max = m.actuator_gear[self.act_motor, 0]

        self.rng = np.random.default_rng(seed)
        self.intent = IntentGenerator(self.rng)
        self.emg = EMGSensor(self.rng, m.opt.timestep)

        obs_dim = 2 + emg_history + 1
        if spaces is not None:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    @property
    def q(self):
        return self.data.qpos[self.qadr]

    @property
    def qd(self):
        return self.data.qvel[self.dadr]

    def gravity_torque(self, q):
        """Gravity torque about the elbow (positive = needed to hold flexion)."""
        mass = self.model.body_mass[self.forearm_body]
        r_com = -self.model.body_ipos[self.forearm_body][2]  # COM distance below elbow
        return mass * 9.81 * r_com * np.sin(q)

    def _human_drive(self, q_target):
        """Biceps neural drive in [0, 1] from a weak PD + partial feed-forward."""
        tau = (
            self.human_ff * self._nom_gravity(q_target)
            + self.human_kp * (q_target - self.q)
            - self.human_kd * self.qd
        )
        return float(np.clip(tau / self.model.actuator_gainprm[self.act_biceps, 0], 0.0, 1.0))

    def _nom_gravity(self, q):
        # the human's internal model of their own (unloaded) arm
        return self._nom_mass * 9.81 * (-self._nom_ipos[2]) * np.sin(q)

    def _randomize(self):
        m = self.model
        if self.randomize:
            payload = self.rng.uniform(0.0, 0.8)          # kg held in hand at 0.31 m
            r_hand = 0.31
            mass = self._nom_mass + payload
            com_z = (self._nom_mass * self._nom_ipos[2] - payload * r_hand) / mass
            m.body_mass[self.forearm_body] = mass
            m.body_ipos[self.forearm_body] = [0.0, 0.0, com_z]
            m.actuator_gainprm[self.act_biceps, 0] = self._nom_biceps_gain * self.rng.uniform(0.6, 1.2)
            emg_gain = self.rng.uniform(0.6, 1.4)          # electrode placement / skin impedance
        else:
            m.body_mass[self.forearm_body] = self._nom_mass
            m.body_ipos[self.forearm_body] = self._nom_ipos
            m.actuator_gainprm[self.act_biceps, 0] = self._nom_biceps_gain
            emg_gain = 1.0
        return emg_gain

    def _obs(self):
        return np.array(
            [self.q, self.qd, *self.emg_buf, self.prev_action], dtype=np.float32
        )

    # --------------------------------------------------------------------- API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            if gym is not None:
                super().reset(seed=seed)
                self.rng = self.np_random
            else:
                self.rng = np.random.default_rng(seed)
            self.intent.rng = self.rng
            self.emg.rng = self.rng
        mujoco.mj_resetData(self.model, self.data)
        emg_gain = self._randomize()

        q0 = self.rng.uniform(0.1, 0.5)
        self.data.qpos[self.qadr] = q0
        mujoco.mj_forward(self.model, self.data)

        self.intent.reset(q0)
        self.emg.reset(emg_gain)
        self.emg_buf = deque([0.0] * self.emg_history, maxlen=self.emg_history)
        self.prev_action = 0.0
        self.steps = 0
        self.q_target = q0
        self.u_human = 0.0
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        self.data.ctrl[self.act_motor] = a

        emg_val = 0.0
        effort = 0.0
        for _ in range(self.frame_skip):
            self.q_target = self.intent(self.data.time)
            self.u_human = self._human_drive(self.q_target)
            self.data.ctrl[self.act_biceps] = self.u_human
            emg_val = self.emg.step(self.u_human)
            mujoco.mj_step(self.model, self.data)
            effort += self.u_human**2 / self.frame_skip
        self.emg_buf.append(emg_val)

        err = self.q_target - self.q
        r_track = -self.w["track"] * err**2
        r_effort = -self.w["effort"] * effort
        r_motor = -self.w["motor"] * a**2
        r_smooth = -self.w["smooth"] * (a - self.prev_action) ** 2
        reward = r_track + r_effort + r_motor + r_smooth

        self.prev_action = a
        self.steps += 1
        truncated = self.steps >= self.max_steps
        info = dict(
            q=self.q, q_target=self.q_target, u_human=self.u_human, emg=emg_val,
            motor_torque=a * self.motor_max,
            r_track=r_track, r_effort=r_effort, r_motor=r_motor, r_smooth=r_smooth,
        )
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), False, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._viewer.cam.azimuth = 90
                self._viewer.cam.elevation = -10
                self._viewer.cam.distance = 1.6
                self._viewer.cam.lookat[:] = [0.1, 0.0, 1.05]
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
