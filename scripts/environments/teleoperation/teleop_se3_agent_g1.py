# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
from typing import cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G1-IK-Rel-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
app_launcher_args = vars(args_cli)

# Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
# not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the    
import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


"""Rest everything follows."""
import gymnasium as gym
import numpy as np
import torch

import omni.log

from isaaclab.devices import Se3Keyboard
from isaaclab.envs import ManagerBasedRLEnv # Import ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

from trajectory_player import TrajectoryPlayer
from isaaclab_tasks.manager_based.manipulation.pick_place_g1.mdp.observations import get_right_eef_pos, get_right_eef_quat, get_left_eef_pos, get_left_eef_quat
from scipy.spatial.transform import Rotation as R

def pre_process_actions(
    # LIVE TELEOP: (delta_pose_6D_numpy, gripper_command_bool)
    # PLAYBACK: tuple[np.ndarray] where np.ndarray is 14D [pos(3), quat_xyzw(4), hand_joints(7)]
    teleop_data: tuple[np.ndarray, bool] | list[tuple[np.ndarray, np.ndarray, np.ndarray]] | tuple[np.ndarray, ...],
    num_envs: int,
    device: str,
    previous_target_right_eef_pos_w: np.ndarray,
    previous_target_right_eef_quat_wxyz_w: np.ndarray,
    trajectory_player: TrajectoryPlayer
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Convert teleop data to the format expected by the environment action space."""
    # teleop_data can be one of two things:
    # 1. From TrajectoryPlayer (playback): a single 28D numpy array  [left_arm_eef(7), right_arm_eef(7), hand_joints(14)]) in wxyz
    # 2. From live device (e.g., keyboard): A tuple (delta_pose_6D_np, gripper_cmd_bool)
    if isinstance(teleop_data, tuple) and len(teleop_data) == 1 and isinstance(teleop_data[0], np.ndarray) and teleop_data[0].shape == (28,):
        # Case 1: Playback from TrajectoryPlayer
        action_array_28D_np = teleop_data[0]
        # Extract right arm data for position tracking
        target_right_eef_pos_w = action_array_28D_np[7:10]  # Right arm position starts at index 7
        target_right_eef_quat_wxyz_w = action_array_28D_np[10:14]  # Right arm quaternion starts at index 10
        
    elif isinstance(teleop_data, tuple) and len(teleop_data) == 2 and isinstance(teleop_data[0], np.ndarray):
        # Case 2: Live teleoperation (e.g., keyboard) for the right arm/hand
        delta_pose_6D_np, gripper_cmd_bool = teleop_data

        # Compose new orientation using axis-angle delta
        target_right_eef_pos_w = previous_target_right_eef_pos_w + delta_pose_6D_np[:3]
        delta_quat = R.from_rotvec(delta_pose_6D_np[3:6]).as_quat()  # xyzw
        target_quat = R.from_quat(delta_quat) * R.from_quat(previous_target_right_eef_quat_wxyz_w)
        target_right_eef_quat_wxyz_w = target_quat.as_quat()

        # Fill left arm with default values (constant pose)
        left_arm_pose = np.array([-0.14866172,  0.1997742,  0.9152355])
        left_arm_quat_wxyz = np.array([0.7071744, 0.0000018,  0.00004074, 0.70703906])  # wxyz
        
        # Create hand joint positions using TrajectoryPlayer's utility function
        hand_positions = trajectory_player.create_hand_joint_positions(
            left_hand_bool=False,  # Always False for left hand
            right_hand_bool=bool(gripper_cmd_bool) # Explicitly cast to bool
        )
        
        # Concatenate all components to form the final action array (wxyz)
        left_arm_eef = np.concatenate([left_arm_pose, left_arm_quat_wxyz])  # Already in wxyz format
        right_arm_eef = np.concatenate([target_right_eef_pos_w, target_right_eef_quat_wxyz_w])
        # [left_arm_eef (7), right_arm_eef (7), hand_positions (14)]
        action_array_28D_np = np.concatenate([left_arm_eef, right_arm_eef, hand_positions])
    else:
        raise ValueError(f"Unexpected teleop_data format for G1 task: {teleop_data}")

    actions = torch.tensor(action_array_28D_np, dtype=torch.float, device=device).repeat(num_envs, 1)
    return actions, target_right_eef_pos_w, target_right_eef_quat_wxyz_w



def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # Create environment and get unwrapped instance for direct access
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg).unwrapped)
    print(f"The environment '{args_cli.task}' uses absolute 6D pose control for the right arm eef and right hand.")

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True # Default to active for keyboard/spacemouse/gamepad
    allow_env_reset = True  # Set to False to disable all environment resets

    def reset_env_and_player(): # Renamed for clarity
        nonlocal should_reset_recording_instance
        if allow_env_reset:
            should_reset_recording_instance = True
            if trajectory_player.is_playing_back:
                trajectory_player.is_playing_back = False
                omni.log.info("Playback stopped due to environment reset request.")
        else:
            print("[INFO] Environment reset is currently disabled (allow_env_reset=False)")

    # Initialize TrajectoryPlayer and teleoperation interface
    trajectory_player = TrajectoryPlayer(env, args_cli.device, steps_per_segment=100)
    teleop_interface = Se3Keyboard(pos_sensitivity=0.005 * args_cli.sensitivity, rot_sensitivity=0.02 * args_cli.sensitivity)

    # Trajectory Player callbacks
    last_teleop_output = None   # Store the last teleop output for use in the callback
    teleop_interface.add_callback("P", lambda: trajectory_player.record_current_pose(last_teleop_output))      # Record Pose
    teleop_interface.add_callback("L", lambda: trajectory_player.load_and_playback()) # Start Playback (loads + plays)
    teleop_interface.add_callback("M", trajectory_player.clear_waypoints)          # Clear Waypoints
    teleop_interface.add_callback("N", lambda: trajectory_player.save_waypoints()) # Save
    teleop_interface.add_callback("R", reset_env_and_player)

    print("\n--- Teleoperation Interface Controls ---")
    print(teleop_interface)
    print("\n--- Trajectory Player Controls for Unitree G1 ---")
    print("  P: Record current EE pose as waypoint.")
    print("  L: Prepare and start playback of recorded trajectory.")
    print("  M: Clear all recorded waypoints from memory.")
    print("  N: Save current waypoints to 'waypoints.json'.")
    print("  R: Reset environment (also stops playback).")
    print("------------------------------------\n")

    # Initialize environment and teleop interface
    env.reset()
    teleop_interface.reset()

    # Get CubeRed's full pose (position and orientation)
    cube_red_pos = env.scene["object1"].data.root_pos_w[0].cpu().numpy()
    cube_red_rot = env.scene["object1"].data.root_quat_w[0].cpu().numpy() # wxyz
    print("Cube red pos and orient:", cube_red_pos, cube_red_rot)
    
    # Get initial EE pose to initialize previous target pose
    previous_target_right_eef_pos_w = get_right_eef_pos(env).cpu().numpy().squeeze()
    previous_target_right_eef_quat_wxyz_w = get_right_eef_quat(env).cpu().numpy().squeeze()
    print("Initial right EE pose:", previous_target_right_eef_pos_w, previous_target_right_eef_quat_wxyz_w)
    print("Initial left EE pose:", get_left_eef_pos(env).cpu().numpy().squeeze(), get_left_eef_quat(env).cpu().numpy().squeeze())

    # Simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            if should_reset_recording_instance:
                env.reset()
                teleop_interface.reset()
                # Re-initialize previous target pose on environment reset
                previous_target_right_eef_pos_w = get_right_eef_pos(env).cpu().numpy().squeeze()
                previous_target_right_eef_quat_wxyz_w = get_right_eef_quat(env).cpu().numpy().squeeze()
                should_reset_recording_instance = False

            raw_teleop_device_output = teleop_interface.advance()
            last_teleop_output = raw_teleop_device_output
            actions_to_step = None

            if trajectory_player.is_playing_back:
                playback_action_tuple = trajectory_player.get_formatted_action_for_playback()
                if playback_action_tuple is not None:
                    actions_to_step, previous_target_right_eef_pos_w, previous_target_right_eef_quat_wxyz_w = pre_process_actions(
                        playback_action_tuple,
                        getattr(env, 'num_envs', 1),
                        getattr(env, 'device', 'cpu'),
                        previous_target_right_eef_pos_w,
                        previous_target_right_eef_quat_wxyz_w,
                        trajectory_player
                    )
            elif teleoperation_active:
                processed_input_for_action_fn = raw_teleop_device_output
                if actions_to_step is None:
                     actions_to_step, previous_target_right_eef_pos_w, previous_target_right_eef_quat_wxyz_w = pre_process_actions(
                        processed_input_for_action_fn,
                        getattr(env, 'num_envs', 1),
                        getattr(env, 'device', 'cpu'),
                        previous_target_right_eef_pos_w,
                        previous_target_right_eef_quat_wxyz_w,
                        trajectory_player
                    )
            
            
            if actions_to_step is not None:
                # actions_to_step: [left_arm_eef(7), right_arm_eef(7), left_hand(7), right_hand(7)]
                obs, rewards, terminated, truncated, info = env.step(actions_to_step)
            else:
                env.sim.render()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
