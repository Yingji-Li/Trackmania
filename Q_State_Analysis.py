import pickle
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load pkl
# ----------------------------
PKL_PATH = "qlearn_out/Q_final.pkl"  # <- change this

with open(PKL_PATH, "rb") as f:
    obj = pickle.load(f)

print("Loaded object type:", type(obj))

# ----------------------------
# Helpers to normalize formats
# ----------------------------
def to_matrix_and_labels(obj):
    """
    Returns:
      M: 2D numpy array
      state_labels: list[str]
      action_labels: list[str]
    Supports:
      - numpy array shape (S, A)
      - pandas.DataFrame
      - dict[state] -> array/list length A
      - dict[state] -> dict[action] -> value
      - dict[(state, action)] -> value
    """
    first_val = obj[keys[0]]
    states = sorted(obj.keys(), key=str)
    lengths = [len(obj[s]) for s in states]
    if len(set(lengths)) != 1:
        raise ValueError(f"State->array dict has varying action lengths: {set(lengths)}")
    A = lengths[0]
    M = np.vstack([np.asarray(obj[s], dtype=float) for s in states])
    action_labels = [str(i) for i in range(A)]
    return M, [str(s) for s in states], action_labels

    raise TypeError(f"Unsupported Q-table format: {type(obj)}")


# ----------------------------
# Build a "visited" matrix
# ----------------------------
M, state_labels, action_labels = to_matrix_and_labels(obj)

# If you ALSO stored a visit-count table, you can load it and use it directly.
# Example: if your pkl contains {"Q": ..., "N": ...} then set:
# counts = to_matrix_and_labels(obj["N"])[0]
# For now we infer visited-ness from Q-table contents:

DEFAULT_UNVISITED_VALUE = 0.0  # change if your Q init isn't 0
treat_nan_as_unvisited = True

if treat_nan_as_unvisited:
    visited = (~np.isnan(M)) & (M != DEFAULT_UNVISITED_VALUE)
else:
    visited = (M != DEFAULT_UNVISITED_VALUE)

visit_int = visited.astype(int)  # 1 = visited, 0 = never visited (inferred)

# ----------------------------
# Plot heatmap
# ----------------------------
plt.figure(figsize=(12, 6))
im = plt.imshow(visit_int, aspect="auto")
plt.title("State Action Heatmap")
plt.xlabel("Action")
plt.ylabel("State")
plt.colorbar(im, label="Visited = 1, Not Visited = 0")

# Tick labels (if huge, it will be unreadable—see note below)
max_ticks = 50
if len(action_labels) <= max_ticks:
    plt.xticks(range(len(action_labels)), action_labels, rotation=90)
if len(state_labels) <= max_ticks:
    plt.yticks(range(len(state_labels)), state_labels)

plt.tight_layout()
plt.show()
