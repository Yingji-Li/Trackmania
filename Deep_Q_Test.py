import os
import torch
import numpy as np

from tmrl import get_environment

# ---- import the SAME helpers you used in Deep Q training ----
# (adjust module name if your training script file is named differently)
from Deep_Q_Learning import (
    extract_features,
    build_action_set,
    DQN,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def greedy_action(q_net: torch.nn.Module, state: np.ndarray, device: torch.device) -> int:
    """
    Greedy (epsilon=0) action selection for DQN.
    """
    s = torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1, state_dim)
    q = q_net(s)  # (1, n_actions)
    return int(torch.argmax(q, dim=1).item())


def load_dqn_checkpoint(ckpt_path: str, state_dim: int, n_actions: int, device: torch.device):
    """
    Loads a .pt checkpoint saved by your training loop.
    Expects keys: 'q_net' (and optionally others).
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    q_net = DQN(state_dim, n_actions).to(device)
    q_net.load_state_dict(ckpt["q_net"])
    q_net.eval()

    return q_net, ckpt


if __name__ == "__main__":
    # ---- point this at your saved checkpoint ----
    CKPT_PATH = os.path.join("dqn_out", "dqn_final.pt")
    # or e.g. os.path.join("dqn_out", "dqn_ep5000.pt")

    actions = build_action_set()
    n_actions = len(actions)
    state_dim = 4  # [speed, left, center, right] in your code

    q_net, ckpt = load_dqn_checkpoint(CKPT_PATH, state_dim, n_actions, DEVICE)
    print(f"Loaded checkpoint: {CKPT_PATH}")
    print(f"Checkpoint episode={ckpt.get('episode')} global_step={ckpt.get('global_step')}")

    env = get_environment()

    for ep in range(5):
        obs, _ = env.reset()
        total = 0.0

        for t in range(3000):
            state = extract_features(obs)
            a_idx = greedy_action(q_net, state, DEVICE)
            action = actions[a_idx]

            obs, rew, term, trunc, _ = env.step(action)
            total += float(rew)

            if term or trunc:
                break

        print(f"Eval episode {ep}: return = {total:.2f}")

    print("Evaluation return:", total)

    env.unwrapped.wait()
