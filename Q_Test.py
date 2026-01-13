import os
import re
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from tmrl import get_environment

# ---- import the SAME helpers you used in training ----
from Q_Learning import (
    extract_features,
    discretize_features,
    make_bins,
    build_action_set,
)

# -----------------------------
# Config
# -----------------------------
CKPT_DIR = "qlearn_out"
CKPT_GLOB = os.path.join(CKPT_DIR, "*.pkl")

n_eval_episodes = 5
max_steps = 3000

# If your environment supports seeding and you want repeatable eval:
# set a fixed seed here; otherwise leave as None.
base_seed = 123  # or None

# -----------------------------
# Helpers
# -----------------------------
def _ckpt_sort_key(path: str):
    """
    Sort by any integer found in filename (e.g., Q_10000.pkl), otherwise put 'final' last.
    """
    name = os.path.basename(path).lower()
    if "final" in name:
        return (10**18, name)
    nums = re.findall(r"\d+", name)
    if nums:
        return (int(nums[-1]), name)
    return (10**17, name)

def load_q_table(pkl_path: str, n_actions: int):
    with open(pkl_path, "rb") as f:
        q_dict = pickle.load(f)

    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    # Works whether q_dict is dict or defaultdict
    Q.update(q_dict)
    return Q

def evaluate_checkpoint(env, Q, actions, bins, n_episodes=5, max_steps=3000, seed=None):
    """
    Deterministic policy: argmax_a Q(s,a). Returns list of episode returns.
    """
    returns = []
    for ep in range(n_episodes):
        if seed is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(seed=seed + ep)

        total = 0.0
        for _ in range(max_steps):
            state = discretize_features(extract_features(obs), bins)
            a_idx = int(np.argmax(Q[state]))
            action = actions[a_idx]
            obs, rew, term, trunc, _ = env.step(action)
            total += float(rew)
            if term or trunc:
                break
        returns.append(total)
    return returns

def td_error_batch(Q, transitions, gamma):
    """
    Compute TD errors for a batch of transitions.
    transitions: list of (s, a_idx, r, s_next, done)
    Returns: np.array of TD errors
    """
    deltas = []
    for s, a_idx, r, s_next, done in transitions:
        q_sa = Q[s][a_idx]
        target = r
        if not done:
            target += gamma * np.max(Q[s_next])
        deltas.append(target - q_sa)
    return np.asarray(deltas, dtype=np.float32)

def collect_eval_transitions(env, Q, actions, bins, n_steps=5000, seed=None):
    """
    Collect transitions under greedy policy for TD-error evaluation.
    """
    transitions = []
    obs, _ = env.reset(seed=seed)
    for _ in range(n_steps):
        s = discretize_features(extract_features(obs), bins)
        a_idx = int(np.argmax(Q[s]))
        action = actions[a_idx]
        obs2, r, term, trunc, _ = env.step(action)
        s2 = discretize_features(extract_features(obs2), bins)
        done = term or trunc
        transitions.append((s, a_idx, float(r), s2, done))
        if done:
            obs, _ = env.reset()
        else:
            obs = obs2
    return transitions

def q_value_scale(Q):
    """
    Compute several scale summaries over the Q-table values.
    Assumes each Q[state] is a vector of actions.
    """
    if len(Q) == 0:
        return dict(max_abs=np.nan, mean_abs=np.nan, rms=np.nan, mean=np.nan)

    # Stack values into one big array for summary stats
    vals = np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in Q.values()])
    abs_vals = np.abs(vals)

    return {
        "max_abs": float(np.max(abs_vals)),
        "mean_abs": float(np.mean(abs_vals)),
        "rms": float(np.sqrt(np.mean(vals * vals))),
        "mean": float(np.mean(vals)),
    }

# -----------------------------
# Main: load checkpoints, eval, plot
# -----------------------------
actions = build_action_set()
bins = make_bins()

ckpt_paths = sorted(glob.glob(CKPT_GLOB), key=_ckpt_sort_key)
if not ckpt_paths:
    raise FileNotFoundError(f"No .pkl checkpoints found in {CKPT_DIR!r} (glob: {CKPT_GLOB})")

print("Found checkpoints:")
for p in ckpt_paths:
    print(" -", p)

env = get_environment()

ckpt_labels = [os.path.basename(p) for p in ckpt_paths]
x = np.arange(len(ckpt_paths))

# Bins
mean_returns = []
all_returns = []
max_abs_list = []
mean_abs_list = []
rms_list = []
q_mean_list = []
mean_td_list = []
rms_td_list = []

# Fixed transition set for TD-error evaluation
dummy_Q = load_q_table(ckpt_paths[0], n_actions=len(actions))
eval_transitions = collect_eval_transitions(
    env, dummy_Q, actions, bins,
    n_steps=3000,
    seed=base_seed
)

for i, pkl_path in enumerate(ckpt_paths):
    Q = load_q_table(pkl_path, n_actions=len(actions))

    # TD errors
    gamma = 0.99  # use the same gamma as training
    td = td_error_batch(Q, eval_transitions, gamma)

    mean_td_list.append(float(np.mean(td)))
    rms_td_list.append(float(np.sqrt(np.mean(td * td))))

    # Deterministic evaluation returns
    rets = evaluate_checkpoint(
        env, Q, actions, bins,
        n_episodes=n_eval_episodes,
        max_steps=max_steps,
        seed=base_seed
    )
    all_returns.append(rets)
    mean_returns.append(float(np.mean(rets)))

    # Q-value scale metrics
    scale = q_value_scale(Q)
    max_abs_list.append(scale["max_abs"])
    mean_abs_list.append(scale["mean_abs"])
    rms_list.append(scale["rms"])
    q_mean_list.append(scale["mean"])

    print(f"[{i+1:02d}/{len(ckpt_paths)}] {os.path.basename(pkl_path)} "
          f"return(mean±std)={mean_returns[-1]:.2f}"
          f"Q(max|.|)={max_abs_list[-1]:.4g} Q(mean|.|)={mean_abs_list[-1]:.4g}"
          f"TD(mean)={mean_td_list[-1]:.4g} TD(RMS)={rms_td_list[-1]:.4g}")
    


import matplotlib.pyplot as plt

# Deterministic returns
plt.figure()
plt.plot(x, mean_returns, markersize=0.3)
plt.xticks(x[::10], ckpt_labels[::10], rotation=45, ha='right')
plt.xlabel("Checkpoint index")
plt.ylabel(f"Deterministic eval return")
plt.title("Deterministic Evaluation Returns Across Checkpoints")
plt.tight_layout()

# Q-value scales
plt.figure()
plt.plot(x, max_abs_list, marker='o', markersize=0.3)
plt.xticks(x[::10], ckpt_labels[::10], rotation=45, ha='right')
plt.xlabel("Checkpoint index")
plt.title("max |Q|")
plt.xticks([])

plt.figure()
plt.plot(x, mean_abs_list, marker='o', label="mean |Q|", markersize=0.3)
plt.plot(x, rms_list, marker='o', label="RMS(Q)", markersize=0.3)
plt.xticks(x[::10], ckpt_labels[::10], rotation=45, ha='right')
plt.xlabel("Checkpoint index")
plt.legend()
plt.xticks([])

plt.show()

# TD error plots
plt.figure()
plt.plot(x, mean_td_list, marker='o', markersize=0.3, label="Mean TD error")
plt.plot(x, rms_td_list, marker='o', markersize=0.3, label="RMS TD error")
plt.xlabel("Checkpoint index")
plt.ylabel("TD error")
plt.title("TD Error and RMS(TD Error) Across Checkpoints")
plt.legend()
plt.xticks([])
plt.tight_layout()

env.unwrapped.wait()