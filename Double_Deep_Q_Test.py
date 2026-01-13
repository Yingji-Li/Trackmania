import os
import re
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from tmrl import get_environment

# ---- import the SAME helpers you used in training (if your DQN uses them) ----
from Deep_Q_Learning import (
    extract_features,
    build_action_set,
)

# -----------------------------
# Config
# -----------------------------
CKPT_DIR = "dqn_out"
CKPT_GLOB = os.path.join(CKPT_DIR, "*.pt")

n_eval_episodes = 5
max_steps = 3000
base_seed = 123  # or None


# -----------------------------
# Helpers
# -----------------------------
def _ckpt_sort_key(path: str):
    """
    Sort by any integer found in filename (e.g., dqn_ep500.pt), otherwise put 'final' last.
    """
    name = os.path.basename(path).lower()
    if "final" in name:
        return (10**18, name)
    nums = re.findall(r"\d+", name)
    if nums:
        return (int(nums[-1]), name)
    return (10**17, name)


def device_from_env():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_q_net_from_checkpoint(pt_path: str, device: torch.device):
    """
    Loads your training checkpoint dict and returns the q_net state_dict.
    Expects keys like: ckpt["q_net"], ckpt["n_actions"], ckpt["state_dim"].
    """
    ckpt = torch.load(pt_path, map_location=device)
    if not isinstance(ckpt, dict) or "q_net" not in ckpt:
        raise ValueError(
            f"Expected a checkpoint dict with key 'q_net' (state_dict). Got: {type(ckpt)} from {pt_path}"
        )
    return ckpt


# --- Model definition must match training exactly ---
import torch.nn as nn

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
def evaluate_checkpoint(env, model, actions, n_episodes=5, max_steps=3000, seed=None, device=None):
    """
    Deterministic policy: argmax_a Q(s,a)
    Returns list of episode returns.
    """
    if device is None:
        device = next(model.parameters()).device

    returns = []
    for ep in range(n_episodes):
        if seed is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(seed=seed + ep)

        total = 0.0
        for _ in range(max_steps):
            s = extract_features(obs)
            s_t = torch.from_numpy(s).to(device=device, dtype=torch.float32).unsqueeze(0)  # [1, 77]

            q = model(s_t)  # [1, n_actions]
            a_idx = int(torch.argmax(q, dim=1).item())
            action = actions[a_idx]

            obs, rew, term, trunc, _ = env.step(action)
            total += float(rew)
            if term or trunc:
                break

        returns.append(total)
    return returns


# -----------------------------
# Main: load checkpoints, eval, plot
# -----------------------------
actions = build_action_set()

ckpt_paths = sorted(glob.glob(CKPT_GLOB), key=_ckpt_sort_key)
if not ckpt_paths:
    raise FileNotFoundError(f"No .pt checkpoints found in {CKPT_DIR!r} (glob: {CKPT_GLOB})")

print("Found checkpoints:")
for p in ckpt_paths:
    print(" -", p)

device = device_from_env()
env = get_environment()

ckpt_labels = [os.path.basename(p) for p in ckpt_paths]
x = np.arange(len(ckpt_paths))

mean_returns = []
all_returns = []

for i, pt_path in enumerate(ckpt_paths):
    ckpt = load_q_net_from_checkpoint(pt_path, device=device)

    # rebuild model exactly as in training
    state_dim = int(ckpt.get("state_dim", 77))
    n_actions = int(ckpt.get("n_actions", len(actions)))

    if n_actions != len(actions):
        raise ValueError(
            f"Action-set mismatch: checkpoint says n_actions={n_actions}, but build_action_set() gives {len(actions)}.\n"
            f"Fix by using the same build_action_set() as training, or load the action set from checkpoint."
        )

    model = DQN(state_dim, n_actions).to(device)
    model.load_state_dict(ckpt["q_net"])
    model.eval()

    rets = evaluate_checkpoint(
        env, model, actions,
        n_episodes=n_eval_episodes,
        max_steps=max_steps,
        seed=base_seed,
        device=device,
    )
    all_returns.append(rets)
    mean_returns.append(float(np.mean(rets)))

    print(f"[{i+1:02d}/{len(ckpt_paths)}] {os.path.basename(pt_path)} "
          f"return(mean±std)={mean_returns[-1]:.2f}±{np.std(rets):.2f}")

# Plot deterministic returns
plt.figure()
plt.plot(x, mean_returns, markersize=0.3)
plt.xticks(x[::10], ckpt_labels[::10], rotation=45, ha="right")
plt.xlabel("Checkpoint index")
plt.ylabel("Deterministic eval return")
plt.title("Deterministic Evaluation Returns Across DQN Checkpoints")
plt.tight_layout()
plt.show()

env.unwrapped.wait()






