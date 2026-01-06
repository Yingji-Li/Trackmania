import json
import math
import os
import time
from collections import deque
from pathlib import Path
import multiprocessing as mp
from queue import Empty, Full

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from tmrl import get_environment


# -----------------------------
# Config helpers
# -----------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Observation -> small continuous state (features)
# -----------------------------
def log_scale_distance(d: float, max_d: float = 100.0) -> float:
    d = float(np.clip(d, 0.0, max_d))
    return float(np.log1p(d) / np.log1p(max_d))


def extract_features(obs) -> np.ndarray:
    """
    Normalised continuous feature vector for DQN:
      [speed, lidar_left, lidar_center, lidar_right] ∈ [0, 1]
    """
    speed = float(np.array(obs[0]).reshape(-1)[0])
    speed = np.clip(speed, 0.0, 300.0) / 300.0

    lidar_hist = np.array(obs[1])        # (4, 19)
    lidar = lidar_hist.mean(axis=0)      # (19,)

    MAX_LIDAR = 100.0
    lidar = np.where(lidar == 0.0, MAX_LIDAR, lidar)

    def sector_min(arr: np.ndarray) -> float:
        arr = np.where(arr == 0.0, MAX_LIDAR, arr)
        return float(np.min(arr))

    left_raw   = sector_min(lidar[:6])
    center_raw = sector_min(lidar[6:13])
    right_raw  = sector_min(lidar[13:])

    left   = log_scale_distance(left_raw, MAX_LIDAR)
    center = log_scale_distance(center_raw, MAX_LIDAR)
    right  = log_scale_distance(right_raw, MAX_LIDAR)

    return np.array([speed, left, center, right], dtype=np.float32)


# -----------------------------
# Discrete action set for TrackMania (expanded)
# -----------------------------
def build_action_set():
    """
    Action format: [gas, brake, steer]
    Adds:
      - coast + steer (0,0,steer)
      - brake + steer (0,1,steer)
    """
    # FIXED typo from your code: you had "-0.5 -0.25" which becomes -0.75.
    steers = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    actions = []
    actions += [np.array([1.0, 0.0, s], dtype=np.float32) for s in steers]  # gas + steer
    actions += [np.array([0.0, 0.0, s], dtype=np.float32) for s in steers]  # coast + steer
    actions += [np.array([0.0, 1.0, s], dtype=np.float32) for s in steers]  # brake + steer
    return actions


# -----------------------------
# Epsilon schedule
# -----------------------------
def epsilon_by_episode(ep, eps_start=1.0, eps_end=0.05, eps_decay=3000):
    return eps_end + (eps_start - eps_end) * math.exp(-ep / eps_decay)


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
def select_action_dqn(q_net: nn.Module, state: np.ndarray, n_actions: int, epsilon: float,
                      rng: np.random.Generator, device: torch.device) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    s = torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)
    q = q_net(s)
    return int(torch.argmax(q, dim=1).item())


# -----------------------------
# Plotting
# -----------------------------
def plot_progress(ep_returns, moving_avg, out_path: str):
    plt.figure()
    plt.plot(ep_returns, label="Episode return")
    plt.plot(moving_avg, label="Moving average")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DDQN progress (async actor-learner)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# Actor process
# -----------------------------
def actor_loop(
    config_path: str,
    traj_queue: mp.Queue,
    stats_queue: mp.Queue,
    weights_queue: mp.Queue,
    stop_event: mp.Event,
    seed: int,
    epsilon_value: mp.Value,
):
    """
    Runs TrackMania env and streams transitions to traj_queue.
    Receives latest q_net weights from weights_queue (non-blocking).
    """
    cfg = load_config(config_path)
    ep_max_len = int(cfg["ENV"]["RTGYM_CONFIG"]["ep_max_length"])

    env = get_environment()

    rng = np.random.default_rng(seed)
    actions = build_action_set()
    n_actions = len(actions)
    state_dim = 4

    # Actor does inference on CPU to avoid fighting learner's GPU
    device = torch.device("cpu")
    q_net = DQN(state_dim, n_actions).to(device)
    q_net.eval()

    # Wait for initial weights
    init = weights_queue.get()
    q_net.load_state_dict(init)

    episode = 0
    while not stop_event.is_set():
        episode += 1
        obs, info = env.reset()
        state = extract_features(obs)
        total_rew = 0.0

        for t in range(ep_max_len):
            # Pull latest weights if available (don’t block)
            try:
                while True:
                    w = weights_queue.get_nowait()
                    q_net.load_state_dict(w)
            except Empty:
                pass

            eps = float(epsilon_value.value)
            a_idx = select_action_dqn(q_net, state, n_actions, eps, rng, device)
            act = actions[a_idx]

            obs2, rew, terminated, truncated, info = env.step(act)
            rew = float(rew)
            done = bool(terminated or truncated)
            state2 = extract_features(obs2)

            # If queue is full, drop rather than blocking the env (keeps actor real-time)
            try:
                traj_queue.put_nowait((state, a_idx, rew, state2, done))
            except Full:
                pass

            total_rew += rew
            state = state2

            if done or stop_event.is_set():
                break

        # Send episode stats (non-blocking)
        try:
            stats_queue.put_nowait((episode, total_rew))
        except Full:
            pass

    try:
        env.close()
    except Exception:
        pass


# -----------------------------
# Async Learner (main)
# -----------------------------
def train_async(
    config_path: str = "config.json",
    episodes: int = 5000,
    gamma: float = 0.99,
    seed: int = 0,
    save_every: int = 50,
    out_dir: str = "ddqn_out_async",
    # DQN hyperparams
    lr: float = 1e-3,
    buffer_size: int = 100_000,
    batch_size: int = 64,
    learning_starts: int = 2_000,
    train_every: int = 4,
    target_update_every: int = 2_000,  # in learner steps (transitions consumed)
    max_grad_norm: float = 10.0,
    # async knobs
    broadcast_every_sec: float = 1.0,   # how often learner pushes fresh weights to actor
):
    os.makedirs(out_dir, exist_ok=True)

    # Spawn-safe
    mp.set_start_method("spawn", force=True)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Learner device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actions = build_action_set()
    n_actions = len(actions)
    state_dim = 4

    q_net = DQN(state_dim, n_actions).to(device)
    target_net = DQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer(buffer_size, state_dim)

    # Queues/events for actor
    traj_queue = mp.Queue(maxsize=50_000)
    stats_queue = mp.Queue(maxsize=1_000)
    weights_queue = mp.Queue(maxsize=4)
    stop_event = mp.Event()
    epsilon_value = mp.Value("d", 1.0)  # shared epsilon

    # Start actor
    actor = mp.Process(
        target=actor_loop,
        args=(str(config_path), traj_queue, stats_queue, weights_queue, stop_event, seed, epsilon_value),
        daemon=True,
    )
    actor.start()

    # Broadcast initial weights
    with torch.no_grad():
        sd0 = {k: v.detach().cpu() for k, v in q_net.state_dict().items()}
    weights_queue.put(sd0)

    # Logging
    ep_returns = []
    moving_avg = []
    ma_window = 20
    recent = deque(maxlen=ma_window)

    # We count “learner steps” as transitions consumed (like your global_step)
    global_step = 0
    last_broadcast = time.time()

    # Episode counter based on actor stats (not perfect, but matches your eps schedule intent)
    last_seen_episode = 0

    try:
        while last_seen_episode < episodes:
            # Read actor episode stats if any
            try:
                while True:
                    ep_id, ep_ret = stats_queue.get_nowait()
                    last_seen_episode = max(last_seen_episode, int(ep_id))

                    ep_returns.append(float(ep_ret))
                    recent.append(float(ep_ret))
                    moving_avg.append(float(np.mean(recent)))

                    # Update epsilon schedule (your original epsilon_by_episode)
                    epsilon_value.value = float(epsilon_by_episode(last_seen_episode))

                    if ep_id % 10 == 0:
                        print(
                            f"Episode {ep_id:4d}/{episodes} | return={ep_ret:8.2f} | "
                            f"ma({ma_window})={moving_avg[-1]:8.2f} | eps={epsilon_value.value:5.3f} | steps={global_step}"
                        )

                    # Save checkpoints (based on actor episode count)
                    if save_every > 0 and ep_id % save_every == 0:
                        ckpt = {
                            "episode": int(ep_id),
                            "global_step": int(global_step),
                            "q_net": q_net.state_dict(),
                            "target_net": target_net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "seed": seed,
                            "gamma": gamma,
                            "lr": lr,
                            "n_actions": n_actions,
                            "state_dim": state_dim,
                        }
                        torch.save(ckpt, os.path.join(out_dir, f"ddqn_ep{ep_id}.pt"))
                        plot_progress(ep_returns, moving_avg, out_path=os.path.join(out_dir, f"progress_ep{ep_id}.png"))
            except Empty:
                pass

            # Consume one transition (block briefly so we don’t spin)
            try:
                s, a, r, s2, d = traj_queue.get(timeout=0.05)
            except Empty:
                continue

            replay.push(s, a, r, s2, d)
            global_step += 1

            # Learn (same as your loop)
            if len(replay) >= learning_starts and (global_step % train_every == 0):
                s_b, a_b, r_b, s2_b, d_b = replay.sample(batch_size, rng)

                s_t  = torch.from_numpy(s_b).to(device=device, dtype=torch.float32)
                a_t  = torch.from_numpy(a_b).to(device=device, dtype=torch.int64).unsqueeze(1)
                r_t  = torch.from_numpy(r_b).to(device=device, dtype=torch.float32).unsqueeze(1)
                s2_t = torch.from_numpy(s2_b).to(device=device, dtype=torch.float32)
                d_t  = torch.from_numpy(d_b).to(device=device, dtype=torch.float32).unsqueeze(1)

                q_sa = q_net(s_t).gather(1, a_t)

                with torch.no_grad():
                    next_a = q_net(s2_t).argmax(dim=1, keepdim=True)
                    next_q = target_net(s2_t).gather(1, next_a)
                    target = r_t + (1.0 - d_t) * gamma * next_q

                loss = loss_fn(q_sa, target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), max_grad_norm)
                optimizer.step()

            # Update target network (same cadence idea as yours)
            if global_step % target_update_every == 0:
                target_net.load_state_dict(q_net.state_dict())

            # Broadcast latest weights periodically (time-based keeps it simple)
            if time.time() - last_broadcast >= broadcast_every_sec:
                with torch.no_grad():
                    sd = {k: v.detach().cpu() for k, v in q_net.state_dict().items()}
                # Don’t block if actor hasn’t consumed old weights yet: drop old and push new
                try:
                    while True:
                        _ = weights_queue.get_nowait()
                except Empty:
                    pass
                try:
                    weights_queue.put_nowait(sd)
                except Full:
                    pass
                last_broadcast = time.time()

        # Final save
        ckpt = {
            "episode": int(last_seen_episode),
            "global_step": int(global_step),
            "q_net": q_net.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "seed": seed,
            "gamma": gamma,
            "lr": lr,
            "n_actions": n_actions,
            "state_dim": state_dim,
        }
        torch.save(ckpt, os.path.join(out_dir, "ddqn_final.pt"))
        plot_progress(ep_returns, moving_avg, out_path=os.path.join(out_dir, "progress_final.png"))

    finally:
        stop_event.set()
        if actor.is_alive():
            actor.join(timeout=2)

    return ep_returns, moving_avg


if __name__ == "__main__":
    train_async(
        config_path=Path("C:/Users/Yingj/TmrlData/config/config.json"),
        episodes=5000,
        gamma=0.99,
        seed=0,
        save_every=50,
        out_dir="ddqn_out_async",
        lr=1e-3,
        buffer_size=100_000,
        batch_size=64,
        learning_starts=2_000,
        train_every=4,
        target_update_every=2_000,
        max_grad_norm=10.0,
        broadcast_every_sec=1.0,
    )