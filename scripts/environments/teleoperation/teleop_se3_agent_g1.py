# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

from trajectory_player import TrajectoryPlayer # Import the new class

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
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg
# Import G1 hand joint configurations for live teleop processing
from isaaclab_tasks.manager_based.manipulation.pick_place_g1.pickplace_g1_env_cfg import (
    G1_RIGHT_HAND_JOINT_NAMES_ORDERED,
    G1_HAND_JOINTS_OPEN_DICT,
    G1_HAND_JOINTS_CLOSED_DICT
)



def pre_process_actions(
    # LIVE TELEOP: (delta_pose_6D_numpy, gripper_command_bool)
    # PLAYBACK: tuple[np.ndarray] where np.ndarray is 14D [pos(3), quat_xyzw(4), hand_joints(7)]
    teleop_data: tuple[np.ndarray, bool] | list[tuple[np.ndarray, np.ndarray, np.ndarray]] | tuple[np.ndarray, ...],
    num_envs: int,
    device: str
) -> torch.Tensor:
    """Convert teleop data to the format expected by the environment action space."""
    # G1 tasks like "PickPlace-G1"
    # teleop_data can be one of two things:
    # 1. From TrajectoryPlayer (playback): A tuple containing a single 14D numpy array
    #    (abs_palm_pos, abs_palm_quat_xyzw, abs_hand_joints)
    # 2. From live device (e.g., keyboard): A tuple (delta_pose_6D_np, gripper_cmd_bool)

    if isinstance(teleop_data, tuple) and len(teleop_data) == 1 and isinstance(teleop_data[0], np.ndarray) and teleop_data[0].shape == (14,):
        # Case 1: Playback from TrajectoryPlayer
        action_array_14D_np = teleop_data[0]
    elif isinstance(teleop_data, tuple) and len(teleop_data) == 2 and isinstance(teleop_data[0], np.ndarray):
        # Case 2: Live teleoperation (e.g., keyboard)
        delta_pose_6D_np, gripper_cmd_bool = teleop_data

        # Get current G1 right palm absolute pose from the environment
        robot = env.unwrapped.scene.articulations["robot"]
        right_palm_idx = robot.find_bodies(["right_palm_link"])[0]
        current_palm_pos_w = robot.data.body_states_w[0, right_palm_idx, :3].cpu().numpy()
        current_palm_quat_wxyz_w = robot.data.body_states_w[0, right_palm_idx, 3:7].cpu().numpy()
        current_palm_quat_xyzw_w = np.array([current_palm_quat_wxyz_w[1], current_palm_quat_wxyz_w[2], current_palm_quat_wxyz_w[3], current_palm_quat_wxyz_w[0]])

        # Apply delta to current absolute pose (simplified, use proper SE3 composition)
        target_palm_pos_w = current_palm_pos_w + delta_pose_6D_np[:3]
        target_palm_quat_xyzw_w = current_palm_quat_xyzw_w # Placeholder - needs proper update

        # Convert gripper command to hand joint positions
        hand_dict_to_use = G1_HAND_JOINTS_CLOSED_DICT if gripper_cmd_bool else G1_HAND_JOINTS_OPEN_DICT
        target_hand_joint_positions_np = np.array([hand_dict_to_use.get(name, 0.0) for name in G1_RIGHT_HAND_JOINT_NAMES_ORDERED])

        action_array_14D_np = np.concatenate([target_palm_pos_w, target_palm_quat_xyzw_w, target_hand_joint_positions_np])
    else:
        raise ValueError(f"Unexpected teleop_data format for G1 task: {teleop_data}")

    actions = torch.tensor(action_array_14D_np, dtype=torch.float, device=device).repeat(num_envs, 1)
    return actions



def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task

    # Safely access commands if the attribute exists
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "object_pose"):
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print(f"The environment '{args_cli.task}' uses absolute 6D pose control for the right palm link and target joint positions for the right hand.")


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

    # Initialize TrajectoryPlayer - Pass empty dicts for G1 hand joints for now
    # TODO: User might need to provide actual joint values for open/closed hand poses
    trajectory_player = TrajectoryPlayer(env, args_cli.device, steps_per_segment=100, g1_hand_joints_open={}, g1_hand_joints_closed={})

    teleop_interface = Se3Keyboard(pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity)

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

    # Simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            if should_reset_recording_instance:
                env.reset()
                if hasattr(teleop_interface, "reset"): teleop_interface.reset()
                should_reset_recording_instance = False

            raw_teleop_device_output = teleop_interface.advance()
            last_teleop_output = raw_teleop_device_output
            actions_to_step = None

            if trajectory_player.is_playing_back:
                playback_action_tuple = trajectory_player.get_formatted_action_for_playback()
                if playback_action_tuple is not None:
                    actions_to_step = pre_process_actions(playback_action_tuple, getattr(env, 'num_envs', 1), getattr(env, 'device', 'cpu'))
            elif teleoperation_active:
                processed_input_for_action_fn = raw_teleop_device_output
                # if "Reach" in args_cli.task:
                #     try:
                #         if hasattr(env.unwrapped.scene, 'sensors') and "ee_frame" in env.unwrapped.scene.sensors:
                #             ee_sensor_reach = env.unwrapped.scene.sensors["ee_frame"]
                #             if hasattr(ee_sensor_reach, 'data') and hasattr(ee_sensor_reach.data, 'target_pos_w'):
                #                 current_ee_pos_abs_tensor = ee_sensor_reach.data.target_pos_w[0, :3]
                #             else:
                #                 current_ee_pos_abs_tensor = env.unwrapped.scene.articulations["robot"].data.ee_state_w[0, :3]
                #         else:
                #             current_ee_pos_abs_tensor = env.unwrapped.scene.articulations["robot"].data.ee_state_w[0, :3]
                #         current_ee_pos_abs_np = current_ee_pos_abs_tensor.cpu().numpy().squeeze()
                #         delta_pos_from_device_np = raw_teleop_device_output[0][:3]
                #         target_abs_pos_for_reach_np = current_ee_pos_abs_np + delta_pos_from_device_np
                #         processed_input_for_action_fn = (target_abs_pos_for_reach_np, raw_teleop_device_output[1])
                #     except Exception as e:
                #         print(f"[MainLoop Reach ERROR] Error getting current EE state: {e}")
                #         actions_to_step = None
                # No specific processing needed for G1 tasks here, TrajectoryPlayer handles it
                # elif "G1" in args_cli.task:
                #     processed_input_for_action_fn = raw_teleop_device_output # Pass raw output to TrajectoryPlayer

                if actions_to_step is None:
                    actions_to_step = pre_process_actions(processed_input_for_action_fn, getattr(env, 'num_envs', 1), getattr(env, 'device', 'cpu'))

            if actions_to_step is not None:
                obs, rewards, terminated, truncated, info = env.step(actions_to_step)
            else:
                env.sim.render()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
