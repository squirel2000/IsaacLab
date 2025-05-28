# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an automated SE(3) trajectory for G1 to grasp a cube."""

"""Launch Isaac Sim Simulator first."""

import argparse
from typing import cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Automated SE(3) trajectory for G1 grasping.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G1-IK-Abs-v0", help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
app_launcher_args = vars(args_cli)

# Import pinocchio before AppLauncher
import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


"""Rest everything follows."""
import gymnasium as gym
import numpy as np
import torch
import time

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from utils.trajectory_player import TrajectoryPlayer

def main():
    """Runs automated grasping trajectory with Isaac Lab G1 environment."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg).unwrapped)

    # Flag to trigger trajectory generation and playback
    trajectory_player = TrajectoryPlayer(env, steps_per_segment=30)    # 30 fps
    should_generate_and_play_trajectory = True

    print("\n--- Automated Grasping Agent ---")
    print("This script will automatically generate and play a trajectory for the G1 to grasp a cube.")
    print("The trajectory consists of moving the right EEF from its current pose to a calculated grasp pose.")
    print("The left arm will remain stationary.")
    print("------------------------------------\n")

    obs, _ = env.reset()
    iteration = 1
    # Simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            if should_generate_and_play_trajectory:
                print("Reset and generate new grasp trajectory...")
                obs, _ = env.reset() # Reset env to reset the cube and arm pose
                time.sleep(3.0) # Pause to allow environment to stabilize
                # 1. Generate the full trajectory by passing the current observation
                trajectory_player.generate_auto_grasp_pick_place_trajectory(obs=obs)
                # 2. Prepare the playback trajectory
                trajectory_player.prepare_playback_trajectory()
                # 3. Set to False to play this trajectory
                should_generate_and_play_trajectory = False

            actions = None
            if trajectory_player.is_playing_back:
                playback_action_tuple = trajectory_player.get_formatted_action_for_playback()
                if playback_action_tuple is not None:
                    action_array_28D_np = playback_action_tuple[0]
                    if not (isinstance(action_array_28D_np, np.ndarray) and action_array_28D_np.shape == (28,)):
                        raise ValueError(f"Unexpected action_array_28D_np format from TrajectoryPlayer: {action_array_28D_np}")
                    actions = torch.tensor(action_array_28D_np, dtype=torch.float, device=args_cli.device).repeat(env.num_envs, 1)
                else: # Playback finished
                    print(f"{iteration} trajectory playback finished, and next iteration will start.\n")
                    should_generate_and_play_trajectory = True
                    iteration += 1
                    actions = env.unwrapped.cfg.idle_action.to(args_cli.device).repeat(env.unwrapped.num_envs, 1) # type: ignore

            if actions is not None:
                obs, rewards, terminated, truncated, info = env.step(actions)
            else:
                # If not playing back and not generating, just render
                env.sim.render()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()