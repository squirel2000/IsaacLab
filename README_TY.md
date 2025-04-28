
# IsaacLab

## 1. Install IsaacLab

```bash
./isaaclab.sh

```
or 
```bash
docker exec -it isaaclab_0 /bin/bash
```

## 2. Train or Play a Task
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-G1-v0 --max_iterations=20 --headless
```
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Rough-G1-v0 --num_envs=20
```

## 3. Verifying & Monitoring in TensorBoard

TensorBoard reads the event files in your logs directory; start it with:

```bash
./isaaclab.sh -p -m tensorboard.main --logdir=logs
```

Open http://localhost:6006 in your browser

## 4. Run the Isaac-PickPlace-GR1T2-Abs-v0 Task

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-PickPlace-GR1T2-Abs-v0 \
  --algo bc --normalize_training_actions
```

dependencies:
```bash
pip install pin
pip install pin-pink
```
