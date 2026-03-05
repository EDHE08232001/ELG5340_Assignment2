# ELG5340 — Assignment 2: Policy Gradient vs Value-Based RL

Compares **REINFORCE** and **DQN** on a stochastic 8×8 GridWorld environment using JAX.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Outputs are written to:

| Directory       | Contents                                      |
|-----------------|-----------------------------------------------|
| `logs/`         | `results.json` — all episode rewards          |
| `checkpoints/`  | Trained model parameters (`.pkl`)             |
| `plots/`        | Learning curves and greedy policy heatmaps    |

## Project Structure

```
├── main.py              # Entry point
├── config.yaml          # Hyperparameters and environment settings
├── requirements.txt
└── src/
    ├── environment.py   # GridWorldEnv (gymnasium)
    ├── models.py        # JAX MLP utilities
    ├── replay_buffer.py # Experience replay buffer (DQN)
    ├── experiments.py   # Hyperparameter sweep runner
    ├── plotting.py      # Learning curves and policy visualisation
    ├── utils.py         # Seeding, one-hot encoding, cleanup
    └── agents/
        ├── dqn.py           # DQN agent
        ├── reinforce.py     # REINFORCE agent
        └── random_agent.py  # Random baseline
```
