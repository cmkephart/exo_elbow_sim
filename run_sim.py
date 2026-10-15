"""Run the exo elbow sim with simple baseline controllers.

    py run_sim.py                      # headless, compare baselines
    py run_sim.py --viewer --policy emg_prop
"""

import argparse
import time

import numpy as np

from exo_env import ExoElbowEnv


def zero_policy(obs, env):
    """No assistance: the user moves the arm alone."""
    return np.array([0.0])


def random_policy(obs, env):
    return np.random.uniform(-1, 1, size=1)


def emg_prop_policy(obs, env, k=1.2, threshold=0.05):
    """Classic myoelectric control: motor torque proportional to EMG envelope."""
    emg = obs[2 + env.emg_history - 1]  # most recent envelope sample
    return np.array([k * max(emg - threshold, 0.0)])


def emg_gravcomp_policy(obs, env, k=1.2, threshold=0.05):
    """EMG-proportional assist plus gravity compensation of the nominal arm."""
    q = obs[0]
    emg = obs[2 + env.emg_history - 1]
    tau = env._nom_gravity(q) + k * env.motor_max * max(emg - threshold, 0.0)
    return np.array([tau / env.motor_max])


POLICIES = dict(zero=zero_policy, random=random_policy, emg_prop=emg_prop_policy,
                emg_gravcomp=emg_gravcomp_policy)


def run_episode(env, policy, seed=None, realtime=False):
    obs, _ = env.reset(seed=seed)
    total, log = 0.0, []
    done = False
    while not done:
        t0 = time.perf_counter()
        obs, r, term, trunc, info = env.step(policy(obs, env))
        total += r
        log.append(info)
        done = term or trunc
        if realtime:
            time.sleep(max(0.0, env.dt - (time.perf_counter() - t0)))
    err = np.array([i["q_target"] - i["q"] for i in log])
    effort = np.array([i["u_human"] for i in log])
    torque = np.array([i["motor_torque"] for i in log])
    return dict(
        ret=total,
        rms_err_deg=np.degrees(np.sqrt(np.mean(err**2))),
        mean_human_drive=effort.mean(),
        mean_abs_motor_Nm=np.abs(torque).mean(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=POLICIES, default=None)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()

    if args.viewer:
        env = ExoElbowEnv(render_mode="human")
        pol = POLICIES[args.policy or "emg_prop"]
        try:
            while env._viewer is None or env._viewer.is_running():
                print(run_episode(env, pol, realtime=True))
        finally:
            env.close()
        return

    names = [args.policy] if args.policy else list(POLICIES)
    env = ExoElbowEnv()
    print(f"{'policy':<14}{'return':>10}{'RMS err (deg)':>15}{'human drive':>13}{'|motor| N*m':>13}")
    for name in names:
        stats = [run_episode(env, POLICIES[name], seed=s) for s in range(args.episodes)]
        avg = {k: np.mean([s[k] for s in stats]) for k in stats[0]}
        print(f"{name:<14}{avg['ret']:>10.1f}{avg['rms_err_deg']:>15.1f}"
              f"{avg['mean_human_drive']:>13.2f}{avg['mean_abs_motor_Nm']:>13.2f}")


if __name__ == "__main__":
    main()
