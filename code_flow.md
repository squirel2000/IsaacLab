IsaacLab/
├── isaaclab.sh                    # Main entry script
├── scripts/
│   └── reinforcement_learning/
│       └── rsl_rl/                # RSL-RL framework implementation
│           ├── train.py           # The main training script
│           └── play.py            # Script for running trained policies
├── source/
    ├── isaaclab/
    │   ├── envs/                      # Environment definitions
    │   │   └── quadcopter/            # Quadcopter environments
    │   │       └── quadcopter.py      # Quadcopter environment implementation
    │   ├── tasks/                     # Task definitions
    │   └── rl/                        # RL components
    ├── isaaclab_assets/               # Assets for IsaacLab
    │   └── robots/                    # Gym interface for IsaacLab
    │       └── quadcopter/            # quadcopter robots
    ├── isaaclab_tasks/                # Assets for IsaacLab
    │   └── direct/                    # Direct task implementations
    │       └── quadcopter/            # quadcopter robots
    │           └── quadcopter_env.py  # Quadcopter task implementation
    │
    └── configs/                       # Configuration files
        └── rl/                        # RL training configs
            └── multicopter/           # Multicopter-specific configs
                └── quadcopter.yaml    # Quadcopter training configuration