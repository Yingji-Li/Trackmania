import json
import math
import os
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from tmrl import get_environment  # TMRL Gymnasium env for TrackMania 2020


# -----------------------------
# Config helpers
# -----------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Observation -> small continuous state (features)
# -----------------------------
def extract_features(obs) -> np.ndarray:
    # speed -> scalar in [0, 1]
    speed = float(np.array(obs[0]).reshape(-1)[0])
    speed = np.clip(speed, 0.0, 300.0) / 300.0

    # lidar history -> (4, 19)
    lidar_hist = np.array(obs[1], dtype=np.float32)

    MAX_LIDAR = 100.0
    # 0.0 usually means "no hit": treat as far away
    lidar_hist = np.where(lidar_hist == 0.0, MAX_LIDAR, lidar_hist)

    # clip + normalize to [0, 1]
    lidar_hist = np.clip(lidar_hist, 0.0, MAX_LIDAR) / MAX_LIDAR

    # flatten (4, 19) -> (76,)
    lidar_flat = lidar_hist.reshape(-1)

    # final state: (77,) = [speed] + 76 lidar values
    return np.concatenate(([speed], lidar_flat)).astype(np.float32)


# -----------------------------
# Discrete action set for TrackMania (expanded)
# -----------------------------
def build_action_set():
    steers = [-1.0, -0.5, 0.0, 0.5, 1.0]
    actions = []
    actions += [np.array([1.0, 0.0, s], dtype=np.float32) for s in steers]  # gas + steer
    actions += [np.array([0.0, 0.0, s], dtype=np.float32) for s in steers]  # coast + steer
    actions += [np.array([0.0, 1.0, s], dtype=np.float32) for s in steers]  # brake + steer
    return actions


# -----------------------------
# Epsilon schedule (fixed)
# -----------------------------
def epsilon_by_step(step, eps_start=1.0, eps_end=0.05, decay_steps=150000):
    # Exponential decay
    return eps_end + (eps_start - eps_end) * math.exp(-step / decay_steps)


# -----------------------------
# Replay buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.buf = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.buf)

    def push(self, s, a, r, s2, done):
        self.buf.append((
            np.asarray(s, dtype=np.float32),
            int(a),
            float(r),
            np.asarray(s2, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int, rng: np.random.Generator):
        idxs = rng.integers(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idxs]

        s  = np.stack([b[0] for b in batch], axis=0)
        a  = np.array([b[1] for b in batch], dtype=np.int64)
        r  = np.array([b[2] for b in batch], dtype=np.float32)
        s2 = np.stack([b[3] for b in batch], axis=0)
        d  = np.array([b[4] for b in batch], dtype=np.float32)
        return s, a, r, s2, d


# -----------------------------
# DQN model
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def select_action_epsilon_greedy(
    q_net: nn.Module,
    state: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    # Explore
    if rng.random() < float(epsilon):
        # Use numpy RNG for consistency with your code
        # (Assumes q_net output size == n_actions)
        s = torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)
        n_actions = int(q_net(s).shape[1])
        return int(rng.integers(0, n_actions))

    # Exploit (greedy)
    s = torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)
    q = q_net(s).squeeze(0)
    return int(torch.argmax(q).item())

# -----------------------------
# Plotting
# -----------------------------
def plot_progress(ep_returns, moving_avg, out_path: str):
    plt.figure()
    plt.plot(ep_returns, label="Episode return")
    plt.plot(moving_avg, label="Moving average")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DDQN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# Synchronous learner (no actor)
# -----------------------------
def train_sync(
    config_path: str = "config.json",
    episodes: int = 6000,
    gamma: float = 0.99,
    seed: int = 0,
    save_every: int = 50,
    out_dir: str = "ddqn_out",
    # DQN hyperparams
    lr: float = 1e-4,
    buffer_size: int = 100_000,
    batch_size: int = 64,
    learning_starts: int = 2_000,   # in env steps
    train_every: int = 4,
    target_update_every: int = 1_000,  # in learner/env steps
    max_grad_norm: float = 1.0,
):
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(config_path)
    ep_max_len = int(cfg["ENV"]["RTGYM_CONFIG"]["ep_max_length"])

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actions = build_action_set()
    n_actions = len(actions)
    state_dim = 77

    env = get_environment()

    q_net = DQN(state_dim, n_actions).to(device)
    target_net = DQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    replay = ReplayBuffer(buffer_size, state_dim)

    ep_returns = []
    moving_avg = []
    ma_window = 20
    recent = deque(maxlen=ma_window)

    global_step = 0

    try:
        for ep in range(1, episodes + 1):
            obs, info = env.reset()
            state = extract_features(obs)
            total_rew = 0.0
            eps = float(epsilon_by_step(global_step))

            for t in range(ep_max_len):
                eps = float(epsilon_by_step(global_step))
                a_idx = select_action_epsilon_greedy(q_net, state, eps, rng, device)
                act = actions[a_idx]

                obs2, rew, terminated, truncated, info = env.step(act)
                rew = float(rew)
                done = bool(terminated or truncated)

                state2 = extract_features(obs2)
                replay.push(state, a_idx, rew, state2, done)

                total_rew += rew
                state = state2
                global_step += 1

                # Learn
                if len(replay) >= learning_starts and (global_step % train_every == 0):
                    s_b, a_b, r_b, s2_b, d_b = replay.sample(batch_size, rng)

                    s_t  = torch.from_numpy(s_b).to(device=device, dtype=torch.float32)
                    a_t  = torch.from_numpy(a_b).to(device=device, dtype=torch.int64).unsqueeze(1)
                    r_t  = torch.from_numpy(r_b).to(device=device, dtype=torch.float32).unsqueeze(1)
                    s2_t = torch.from_numpy(s2_b).to(device=device, dtype=torch.float32)
                    d_t  = torch.from_numpy(d_b).to(device=device, dtype=torch.float32).unsqueeze(1)

                    q_sa = q_net(s_t).gather(1, a_t)

                    with torch.no_grad():
                        next_actions = q_net(s2_t).argmax(dim=1, keepdim=True)
                        next_q = target_net(s2_t).gather(1, next_actions)
                        target = r_t + (1.0 - d_t) * gamma * next_q

                    loss = loss_fn(q_sa, target)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(q_net.parameters(), max_grad_norm)
                    optimizer.step()

                # Target net update
                if global_step % target_update_every == 0:
                    target_net.load_state_dict(q_net.state_dict())

                if done:
                    break

            ep_returns.append(float(total_rew))
            recent.append(float(total_rew))
            moving_avg.append(float(np.mean(recent)))

            if ep % 10 == 0:
                print(
                    f"Episode {ep:4d}/{episodes} | return={total_rew:8.2f} | "
                    f"ma({ma_window})={moving_avg[-1]:8.2f} | eps={eps:5.3f} | steps={global_step}"
                )

            if save_every > 0 and ep % save_every == 0:
                ckpt = {
                    "episode": ep,
                    "global_step": global_step,
                    "q_net": q_net.state_dict(),
                    "target_net": target_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "seed": seed,
                    "gamma": gamma,
                    "lr": lr,
                    "n_actions": n_actions,
                    "state_dim": state_dim,
                }
                torch.save(ckpt, os.path.join(out_dir, f"dqn_ep{ep}.pt"))
                plot_progress(ep_returns, moving_avg, out_path=os.path.join(out_dir, f"progress_ep{ep}.png"))

        ckpt = {
            "episode": episodes,
            "global_step": global_step,
            "q_net": q_net.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "seed": seed,
            "gamma": gamma,
            "lr": lr,
            "n_actions": n_actions,
            "state_dim": state_dim,
        }
        torch.save(ckpt, os.path.join(out_dir, "dqn_final.pt"))
        plot_progress(ep_returns, moving_avg, out_path=os.path.join(out_dir, "progress_final.png"))

    finally:
        try:
            env.close()
        except Exception:
            pass

    return ep_returns, moving_avg


if __name__ == "__main__":
    train_sync(
        config_path=str(Path("C:/Users/Yingj/TmrlData/config/config.json")),
        episodes=6000,
        gamma=0.99,
        seed=0,
        save_every=50,
        out_dir="ddqn_out",
        lr=1e-4,
        buffer_size=100_000,
        batch_size=64,
        learning_starts=2000,
        train_every=4,
        target_update_every=1000,
        max_grad_norm=1.0,
    )
