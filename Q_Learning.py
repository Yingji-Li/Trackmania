import json
import math
import os
import pickle
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tmrl import get_environment  # TMRL Gymnasium env for TrackMania 2020


# -----------------------------
# Config helpers
# -----------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Observation -> small discrete state
# -----------------------------
def extract_features(obs):
    speed = float(np.array(obs[0]).reshape(-1)[0])  # 0..300-ish

    lidar_hist = np.array(obs[1])     # (4,19)
    lidar = lidar_hist.mean(axis=0)

    MAX_LIDAR = 100.0
    lidar = np.where(lidar == 0.0, MAX_LIDAR, lidar)

    left_raw   = float(np.min(lidar[:6]))
    center_raw = float(np.min(lidar[6:13]))
    right_raw  = float(np.min(lidar[13:]))

    return np.array([speed, left_raw, center_raw, right_raw], dtype=np.float32)

def make_bins():
    speed_bins = np.array([5, 15, 30, 50, 80, 120, 160, 220, 300], dtype=np.float32)
    lidar_bins = np.array([5, 10, 15, 25, 40, 60, 80, 95], dtype=np.float32)
    return speed_bins, lidar_bins

def discretize_features(feat: np.ndarray, bins) -> tuple:
    speed_bins, lidar_bins = bins
    speed, left, center, right = feat.tolist()

    dspeed = int(np.digitize(speed, speed_bins))
    dleft = int(np.digitize(left, lidar_bins))
    dcenter = int(np.digitize(center, lidar_bins))
    dright = int(np.digitize(right, lidar_bins))

    return (dspeed, dleft, dcenter, dright)

# -----------------------------
# Discrete action set for TrackMania
# -----------------------------
def build_action_set():
    steers = [-1.0, -0.5, 0.0, 0.5, 1.0]
    actions = []
    actions += [np.array([1.0, 0.0, s], dtype=np.float32) for s in steers]  # gas + steer
    actions += [np.array([0.0, 0.0, s], dtype=np.float32) for s in steers]  # coast + steer
    actions += [np.array([0.0, 1.0, s], dtype=np.float32) for s in steers]  # brake + steer
    return actions


# -----------------------------
# Q-learning
# -----------------------------
def epsilon_by_episode(ep, eps_start=1.0, eps_end=0.05, eps_decay=1500):
    # Keep full exploration early, then decay.
    if ep < 500:
        return 1.0
    ep2 = ep - 500
    return eps_end + (eps_start - eps_end) * math.exp(-ep2 / eps_decay)


def select_action(Q, state, actions, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(actions)))
    q = Q[state]
    mx = np.max(q)
    best = np.flatnonzero(q == mx)
    return int(rng.choice(best))


def train(
    config_path: str = "config.json",
    episodes: int = 4000,
    alpha: float = 0.10,
    gamma: float = 0.99,
    seed: int = 0,
    save_every: int = 50,
    out_dir: str = "qlearn_out",
):
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(config_path)
    # Episode length from your config (rtgym ep_max_length) :contentReference[oaicite:4]{index=4}
    ep_max_len = int(cfg["ENV"]["RTGYM_CONFIG"]["ep_max_length"])

    # Create env (TMRL env depends on the config.json used by your TMRL install) :contentReference[oaicite:5]{index=5}
    env = get_environment()

    rng = np.random.default_rng(seed)
    actions = build_action_set()
    bins = make_bins()

    # Q table: maps discrete state -> array(len(actions))
    Q = defaultdict(lambda: np.zeros(len(actions), dtype=np.float32))

    # Logging
    ep_returns = []
    ep_lengths = []
    moving_avg = []
    ma_window = 20
    recent = deque(maxlen=ma_window)

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        feat = extract_features(obs)
        state = discretize_features(feat, bins)

        eps = epsilon_by_episode(ep)
        total_rew = 0.0
        steps = 0

        for t in range(ep_max_len):
            a_idx = select_action(Q, state, actions, eps, rng)
            act = actions[a_idx]

            obs2, rew, terminated, truncated, info = env.step(act)
            total_rew += float(rew)
            steps += 1

            feat2 = extract_features(obs2)
            state2 = discretize_features(feat2, bins)

            # Q-learning update
            best_next = float(np.max(Q[state2]))
            td_target = float(rew) + (0.0 if (terminated or truncated) else gamma * best_next)
            td_error = td_target - float(Q[state][a_idx])
            Q[state][a_idx] += alpha * td_error

            state = state2
  
            if ep == 1:
                print(f"t={t} rew={rew}")

            if terminated or truncated:
                print("DONE at t=", t)
                break


        ep_returns.append(total_rew)
        ep_lengths.append(steps)

        recent.append(total_rew)
        ma = float(np.mean(recent))
        moving_avg.append(ma)

        if ep % 10 == 0:
            print(f"Episode {ep:4d}/{episodes} | return={total_rew:8.2f} | ma({ma_window})={ma:8.2f} | eps={eps:5.3f}")

        # Save checkpoints
        if save_every > 0 and ep % save_every == 0:
            with open(os.path.join(out_dir, f"Q_ep{ep}.pkl"), "wb") as f:
                pickle.dump(dict(Q), f)

            # Progress plot
            plot_progress(ep_returns, moving_avg, out_path=os.path.join(out_dir, f"progress_ep{ep}.png"))

    # Final save + plot
    with open(os.path.join(out_dir, "Q_final.pkl"), "wb") as f:
        pickle.dump(dict(Q), f)
    plot_progress(ep_returns, moving_avg, out_path=os.path.join(out_dir, "progress_final.png"))

    env.unwrapped.wait()  # rtgym pause convenience (optional)
    return ep_returns, moving_avg

def plot_progress(ep_returns, moving_avg, out_path: str):
    plt.figure()
    plt.plot(ep_returns, label="Episode return")
    plt.plot(moving_avg, label="Moving average")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Tabular Q-learning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    # If your config lives elsewhere, update this path:
    # This is your provided config file :contentReference[oaicite:6]{index=6}
    train(
        config_path = Path("C:/Users/Yingj/TmrlData/config/config.json"),
        episodes=4000,
        alpha=0.10,
        gamma=0.99,
        seed=0,
        save_every=50,
        out_dir="qlearn_out",
    )
