import pickle
import numpy as np
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
# Load learned Q-table
# -----------------------------
with open("qlearn_out/Q_final.pkl", "rb") as f:
    Q_dict = pickle.load(f)

# Convert back to defaultdict for safety
actions = build_action_set()
Q = defaultdict(lambda: np.zeros(len(actions), dtype=np.float32))
Q.update(Q_dict)

bins = make_bins()

# -----------------------------
# Run policy (no exploration)
# -----------------------------
env = get_environment()

for ep in range(5):
    obs, _ = env.reset()
    total = 0.0
    for t in range(3000):
        state = discretize_features(extract_features(obs), bins)
        action = actions[int(np.argmax(Q[state]))]
        obs, rew, term, trunc, _ = env.step(action)
        total += float(rew)
        if term or trunc:
            break
    print(f"Eval episode {ep}: return = {total:.2f}")


print("Evaluation return:", total)

env.unwrapped.wait()
