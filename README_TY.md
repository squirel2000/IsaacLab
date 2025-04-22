
# IsaacLab

## 1. Install IsaacLab

```bash
./isaaclab.sh

```
or 
```bash
docker exec -it isaaclab_0 /bin/bash
```

## 2. Run the Training Script
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-G1-v0 --max_iterations=20 --headless
```

## 3. Verifying & Monitoring in TensorBoard

TensorBoard reads the event files in your logs directory; start it with:

```bash
./isaaclab.sh -p -m tensorboard.main --logdir=logs
```

Open http://localhost:6006 in your browser