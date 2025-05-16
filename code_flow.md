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


### Install latest CMake 3.30.2:

```bash
# Install build dependencies
sudo apt install -y \
    cmake git g++ libboost-all-dev libeigen3-dev \
    libassimp-dev liburdfdom-dev libode-dev
sudo apt install libssl-dev zlib1g-dev

sudo apt purge --auto-remove cmake
wget https://github.com/Kitware/CMake/releases/download/v3.30.2/cmake-3.30.2.tar.gz
tar -zxvf cmake-3.30.2.tar.gz
cd cmake-3.30.2
./bootstrap
make -j$(nproc)
sudo make install
# Verify installation
cmake --version  # Should show 3.30.2
```

### Install Pinocchio and related packages from source with Isaac Sim-compatible flags
```bash
cd ~/Downloads

# Build and install pinned versions
pip uninstall -y pinocchio hpp-fcl eigenpy pink-ik  # Remove existing installations

# Install from source with Isaac Sim-compatible flags
git clone --recursive https://github.com/stack-of-tasks/pinocchio.git
cd pinocchio
git checkout v2.6.9
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV \
         -DBUILD_PYTHON_INTERFACE=ON \
         -DBUILD_TESTING=OFF
make install
cd ../..

# Install HPP-FCL with compatible settings
git clone https://github.com/humanoid-path-planner/hpp-fcl.git
cd hpp-fcl
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV
make install
cd ../..

# Install pink-ik
pip install pink-ik --no-binary pink-ik
```