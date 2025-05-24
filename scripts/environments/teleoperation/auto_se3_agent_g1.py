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
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G1-IK-Rel-v0", help="Name of the task.")

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
from scipy.spatial.transform import Rotation as R, Slerp

import omni.log

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from trajectory_player import TrajectoryPlayer
from grasp_pose_calculator import GraspPoseCalculator
from isaaclab_tasks.manager_based.manipulation.pick_place_g1.mdp.observations import (
    get_right_eef_pos,
    get_right_eef_quat,
    get_left_eef_pos, # Though left is fixed, good to have for consistency if needed
    get_left_eef_quat,
)

# Constants for the left arm
DEFAULT_LEFT_ARM_POS_W = np.array([-0.14866172, 0.1997742, 0.9152355])
DEFAULT_LEFT_ARM_QUAT_WXYZ_W = np.array([0.7071744, 0.0000018, 0.00004074, 0.70703906]) # wxyz
DEFAULT_LEFT_HAND_BOOL = False # False for open

# Constants for RED_PLATE pose
RED_PLATE_XY_POS = np.array([0.250, 0.200])
# Z will be determined dynamically based on grasp height
RED_PLATE_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0]) # Identity quaternion (w,x,y,z)



def pre_process_actions(
    action_array_28D_np: np.ndarray,
    num_envs: int,
    device: str,
) -> torch.Tensor:
    """Convert 28D numpy action array from TrajectoryPlayer to the format expected by the environment action space."""
    # action_array_28D_np is [left_arm_eef(7), right_arm_eef(7), hand_joints(14)] in wxyz for quats
    if not (isinstance(action_array_28D_np, np.ndarray) and action_array_28D_np.shape == (28,)):
        raise ValueError(f"Unexpected action_array_28D_np format: {action_array_28D_np}")

    actions = torch.tensor(action_array_28D_np, dtype=torch.float, device=device).repeat(num_envs, 1)
    return actions


def main():
    """Runs automated grasping trajectory with Isaac Lab G1 environment."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg).unwrapped)
    print(f"The environment '{args_cli.task}' uses absolute 6D pose control for EEFs and hand joints.")

    trajectory_player = TrajectoryPlayer(env, args_cli.device, steps_per_segment=100)
    grasp_calculator = GraspPoseCalculator() # Uses default example grasp for relative transform

    # Flag to trigger trajectory generation and playback
    # Set to True to run the auto-grasp sequence once after initial reset.
    should_generate_and_play_trajectory = True

    print("\n--- Automated Grasping Agent ---")
    print("This script will automatically generate and play a trajectory for the G1 to grasp a cube.")
    print("The trajectory consists of moving the right EEF from its current pose to a calculated grasp pose.")
    print("The left arm will remain stationary.")
    print("------------------------------------\n")

    env.reset()

    # Simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            if should_generate_and_play_trajectory:
                omni.log.info("Generating new grasp trajectory...")
                env.reset() # Reset env to get new cube pose if desired, or ensure clean state
                # Add a 3-second pause for environment stabilization
                omni.log.info("Pausing for 3 seconds to allow environment to stabilize...")
                time.sleep(3.0)

                # 1. Get current poses
                current_right_eef_pos_w = get_right_eef_pos(env).cpu().numpy().squeeze()
                current_right_eef_quat_wxyz_w = get_right_eef_quat(env).cpu().numpy().squeeze()

                cube_red_pos_w = env.scene["object1"].data.root_pos_w[0].cpu().numpy()
                cube_red_quat_wxyz_w = env.scene["object1"].data.root_quat_w[0].cpu().numpy() # wxyz

                print(f"Current Right EEF Pose: pos={current_right_eef_pos_w}, quat_wxyz={current_right_eef_quat_wxyz_w}")
                print(f"Target Cube Pose: pos={cube_red_pos_w}, quat_wxyz={cube_red_quat_wxyz_w}")

                # 2. Calculate target grasp pose for the right EEF
                target_grasp_right_eef_pos_w, target_grasp_right_eef_quat_wxyz_w = \
                    grasp_calculator.calculate_target_ee_pose(cube_red_pos_w, cube_red_quat_wxyz_w)
                print(f"Calculated Target Grasp Right EEF Pose: pos={target_grasp_right_eef_pos_w}, quat_wxyz={target_grasp_right_eef_quat_wxyz_w}")

                # 3. Prepare waypoints for TrajectoryPlayer
                trajectory_player.clear_waypoints()

                # Waypoint 1: Current EEF pose (right hand open)
                wp1_left_arm_eef = np.concatenate([DEFAULT_LEFT_ARM_POS_W, DEFAULT_LEFT_ARM_QUAT_WXYZ_W])
                wp1_right_arm_eef = np.concatenate([current_right_eef_pos_w, current_right_eef_quat_wxyz_w])
                waypoint1 = {
                    "left_arm_eef": wp1_left_arm_eef,
                    "right_arm_eef": wp1_right_arm_eef,
                    "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL), # False = open
                    "right_hand_bool": 0  # 0 for open at the start of trajectory
                }
                trajectory_player.recorded_waypoints.append(waypoint1)

                # Waypoint 2: Target grasp EEF pose (right hand open)
                wp2_left_arm_eef = np.concatenate([DEFAULT_LEFT_ARM_POS_W, DEFAULT_LEFT_ARM_QUAT_WXYZ_W]) # Left arm remains same
                wp2_right_arm_eef = np.concatenate([target_grasp_right_eef_pos_w, target_grasp_right_eef_quat_wxyz_w])
                waypoint2 = {
                    "left_arm_eef": wp2_left_arm_eef,
                    "right_arm_eef": wp2_right_arm_eef,
                    "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL), # False = open
                    "right_hand_bool": 0  # 1 for closed at the grasp pose
                }
                trajectory_player.recorded_waypoints.append(waypoint2)
                
                # Waypoint 3: Target grasp EEF pose (right hand closed)
                waypoint3 = {
                    "left_arm_eef": wp2_left_arm_eef,
                    "right_arm_eef": wp2_right_arm_eef,
                    "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL), # False = open
                    "right_hand_bool": 1  # 1 for closed at the grasp pose
                }
                trajectory_player.recorded_waypoints.append(waypoint3)

                # Define RED_PLATE pose based on dynamic Z from grasp
                red_plate_pos_w = np.array([RED_PLATE_XY_POS[0], RED_PLATE_XY_POS[1], target_grasp_right_eef_pos_w[2]])
                red_plate_quat_wxyz_w = RED_PLATE_QUAT_WXYZ

                # Waypoint 4: Intermediate lift pose
                # Position: Midpoint in XY between grasp (wp3) and RED_PLATE, Z is 0.05m higher than their average Z
                lift_pos_x = (target_grasp_right_eef_pos_w[0] + red_plate_pos_w[0]) / 2
                lift_pos_y = (target_grasp_right_eef_pos_w[1] + red_plate_pos_w[1]) / 2
                lift_pos_z = (target_grasp_right_eef_pos_w[2] + red_plate_pos_w[2]) / 2 + 0.05
                lift_pos_w = np.array([lift_pos_x, lift_pos_y, lift_pos_z])

                # Orientation: Slerp between grasp (wp3) and RED_PLATE
                # Scipy Slerp needs quaternions in [x, y, z, w]
                quat_grasp_xyzw = target_grasp_right_eef_quat_wxyz_w[[1, 2, 3, 0]]
                quat_red_plate_xyzw = red_plate_quat_wxyz_w[[1, 2, 3, 0]]
                
                key_rots = R.from_quat([quat_grasp_xyzw, quat_red_plate_xyzw])
                slerp_interpolator = Slerp([0, 1], key_rots)
                lift_quat_xyzw = slerp_interpolator(0.5).as_quat() # Interpolate to midpoint
                lift_quat_wxyz_w = lift_quat_xyzw[[3, 0, 1, 2]] # Convert back to [w, x, y, z]

                waypoint4 = {
                    "left_arm_eef": wp2_left_arm_eef,
                    "right_arm_eef": np.concatenate([lift_pos_w, lift_quat_wxyz_w]),
                    "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL),
                    "right_hand_bool": 1 # Keep hand closed
                }
                trajectory_player.recorded_waypoints.append(waypoint4)

                # Waypoint 5: Move right arm EEF to RED_PLATE pose
                waypoint5 = {
                    "left_arm_eef": wp2_left_arm_eef,
                    "right_arm_eef": np.concatenate([red_plate_pos_w, red_plate_quat_wxyz_w]),
                    "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL),
                    "right_hand_bool": 1 # Keep hand closed
                }
                trajectory_player.recorded_waypoints.append(waypoint5)

                # Waypoint 6: Keep the same pose as Waypoint 5 (RED_PLATE), but open the right hand
                waypoint6 = {
                    "left_arm_eef": wp2_left_arm_eef,
                    "right_arm_eef": np.concatenate([red_plate_pos_w, red_plate_quat_wxyz_w]),  # Same pose as waypoint 5
                    "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL),
                    "right_hand_bool": 0  # Open the hand
                }
                trajectory_player.recorded_waypoints.append(waypoint6)

                # Waypoint 7: Move the right arm to the initial pose
                waypoint7 = {
                    "left_arm_eef": wp2_left_arm_eef,
                    "right_arm_eef": wp1_right_arm_eef,  # Back to initial pose
                    "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL),
                    "right_hand_bool": 0 # Keep hand open
                }
                trajectory_player.recorded_waypoints.append(waypoint7)

                print("Generated Waypoints:")
                print(f"  Waypoint 1 (Current): Right EEF Pos {current_right_eef_pos_w.round(3)}, Right Hand Open")
                print(f"  Waypoint 2 (Grasp): Right EEF Pos {target_grasp_right_eef_pos_w.round(3)}, Right Hand Open (Pre-Grasp)")
                print(f"  Waypoint 3 (Grasp): Right EEF Pos {target_grasp_right_eef_pos_w.round(3)}, Right Hand Closed (Grasping)")
                print(f"  Waypoint 4 (Lift): Right EEF Pos {lift_pos_w.round(3)}, Quat {lift_quat_wxyz_w.round(3)}, Right Hand Closed (Lifting)")
                print(f"  Waypoint 5 (RED_PLATE): Right EEF Pos {red_plate_pos_w.round(3)}, Quat {red_plate_quat_wxyz_w.round(3)}, Right Hand Closed (Moving to Plate)")
                print(f"  Waypoint 6 (Release): Right EEF Pos {red_plate_pos_w.round(3)}, Quat {red_plate_quat_wxyz_w.round(3)}, Right Hand Open (Releasing at Plate)")
                print(f"  Waypoint 7 (Initial): Right EEF Pos {current_right_eef_pos_w.round(3)}, Right Hand Open (Returning to Initial)")

                # Prepare the playback trajectory
                trajectory_player.prepare_playback_trajectory()

                should_generate_and_play_trajectory = False # Set to False to play this trajectory
                                                        # Set to True again if you want to loop after playback.

            actions_to_step = None
            if trajectory_player.is_playing_back:
                playback_action_tuple = trajectory_player.get_formatted_action_for_playback()
                if playback_action_tuple is not None:
                    action_array_28D_np = playback_action_tuple[0]
                    actions_to_step = pre_process_actions(
                        action_array_28D_np,
                        args_cli.num_envs, # Use parsed num_envs
                        args_cli.device    # Use parsed device
                    )
                else: # Playback finished
                    omni.log.info("Automated trajectory playback finished.")
                    # To make it run continuously, uncomment the next line:
                    # should_generate_and_play_trajectory = True
                    # For now, it will play once and then idle.
                    # You might want to add a small delay or a specific condition
                    # if you enable continuous looping.
                    pass # Do nothing, let it idle or wait for external reset/new logic

            if actions_to_step is not None:
                obs, rewards, terminated, truncated, info = env.step(actions_to_step)
            else:
                # If not playing back and not generating, just render
                env.sim.render()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()