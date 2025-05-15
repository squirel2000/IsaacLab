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
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import numpy as np
import torch

import omni.log

if "handtracking" in args_cli.teleop_device.lower():
    from isaacsim.xr.openxr import OpenXRSpec

from isaaclab.devices import OpenXRDevice, Se3Gamepad, Se3Keyboard, Se3SpaceMouse

if args_cli.enable_pinocchio:
    from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import GR1T2Retargeter
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab.devices.openxr.retargeters.manipulator import GripperRetargeter, Se3AbsRetargeter, Se3RelRetargeter
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg



def pre_process_actions(
    # teleop_data format depends on task_name:
    # For "Reach": (abs_pos_3D_numpy, gripper_command_bool)
    # For "Lift" (IK-Rel) & others: (delta_pose_6D_numpy, gripper_command_bool)
    # For "PickPlace-GR1T2": list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    teleop_data: tuple[np.ndarray, bool] | list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_envs: int,
    device: str,
    task_name: str
) -> torch.Tensor:
    """Convert teleop data to the format expected by the environment action space."""
    if "Reach" in task_name:
        # Input: (abs_pos_3D_numpy, gripper_command_bool)
        # Output: Action is just the target absolute 3D position. Gripper command usually ignored.
        target_pos_abs_3D_np, _ = teleop_data
        actions = torch.tensor(target_pos_abs_3D_np, dtype=torch.float, device=device).repeat(num_envs, 1)
        return actions
    elif "PickPlace-GR1T2" in task_name:
        (left_wrist_pose, right_wrist_pose, hand_joints) = teleop_data[0]
        # Reconstruct actions_arms tensor with converted positions and rotations
        actions = torch.tensor(
            np.concatenate([
                left_wrist_pose,  # left ee pose
                right_wrist_pose,  # right ee pose
                hand_joints,  # hand joint angles
            ]),
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0) # Expects batch dim
        if num_envs > 1 and actions.shape[0] == 1: # repeat if single action for multiple envs
            actions = actions.repeat(num_envs, 1)
        return actions
    elif "G1" in task_name: # G1 tasks like "PickPlace-G1"
        # Input from TrajectoryPlayer: (action_array_13D,)
        # action_array_13D is [right palm pos (3), right palm quat (4), right hand joint pos (7)]
        action_array_13D = teleop_data[0] # Extract the numpy array from the tuple

        # Convert numpy array to torch tensor and repeat for num_envs
        actions = torch.tensor(action_array_13D, dtype=torch.float, device=device).repeat(num_envs, 1)
        return actions
    else: # IK-Rel tasks like "Lift"
        # Input: (delta_pose_6D_numpy, gripper_command_bool)
        # delta_pose_6D_numpy is [dx, dy, dz, d_ax, d_ay, d_az] (axis-angle for rotation)
        # gripper_command_bool: True for "grip" (close), False for "open" (release)
        delta_pose_6D_np, gripper_cmd_bool = teleop_data

        delta_pose_tensor = torch.tensor(delta_pose_6D_np, dtype=torch.float, device=device).repeat(num_envs, 1)

        # Gripper command: typically -1 for close (grip), 1 for open (release) in env actions
        gripper_val = -1.0 if gripper_cmd_bool else 1.0
        gripper_action_tensor = torch.full((delta_pose_tensor.shape[0], 1), gripper_val, dtype=torch.float, device=device)

        actions = torch.concat([delta_pose_tensor, gripper_action_tensor], dim=1)
        return actions


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # Safely set env_name if the attribute exists
    if hasattr(env_cfg, "env_name"):
        env_cfg.env_name = args_cli.task

    # Safely handle terminations if the attribute exists and is a dictionary
    if hasattr(env_cfg, "terminations") and isinstance(env_cfg.terminations, dict):
        # Clear existing terminations if present
        if env_cfg.terminations is not None:
             env_cfg.terminations = {}

        if "Lift" in args_cli.task:
            # set the resampling time range to large number to avoid resampling
            # Safely access commands if the attribute exists
            if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "object_pose"):
                env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
            # add termination condition for reaching the goal otherwise the environment won't reset
            # if hasattr(mdp, "object_reached_goal"):
            #     env_cfg.terminations["object_reached_goal"] = DoneTerm(func=mdp.object_reached_goal)
        elif "Reach" in args_cli.task:
            # Ensure there is a timeout for Reach tasks
            if hasattr(env_cfg.terminations, "time_out") and env_cfg.terminations.get("time_out") is None:
                 print(f"Reach task '{args_cli.task}' did not have a time_out termination. Adding a default one.")
                 # Assuming mdp.time_out is available and appropriate
                 if hasattr(mdp, "time_out"):
                     env_cfg.terminations["time_out"] = DoneTerm(func=mdp.time_out, time_out=True)

    print("Active terminations in env_cfg:", getattr(env_cfg, "terminations", None))

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        print(
            f"The environment '{args_cli.task}' uses absolute 3D position control for the end-effector. "
            "Orientation commands from SE3 devices will be used to calculate deltas but only position is sent."
        )
    elif "G1" in args_cli.task:
         print(
            f"The environment '{args_cli.task}' uses relative 6D pose control for the right palm link. "
            "Gripper commands from SE3 devices will control the right hand joints."
        )


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

    def start_teleoperation():
        nonlocal teleoperation_active
        teleoperation_active = True
        omni.log.info("Teleoperation Activated.")

    def stop_teleoperation():
        nonlocal teleoperation_active
        teleoperation_active = False
        omni.log.info("Teleoperation Deactivated.")

    trajectory_player = TrajectoryPlayer(env, args_cli.device, steps_per_segment=100)

    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.10 * args_cli.sensitivity, rot_sensitivity=0.10 * args_cli.sensitivity
        )
    elif "dualhandtracking_abs" in args_cli.teleop_device.lower() and "GR1T2" in args_cli.task:
        # The following block is commented out to avoid errors if env.scene or env.unwrapped.device is missing
        gr1t2_retargeter = GR1T2Retargeter(
            enable_visualization=True,
            num_open_xr_hand_joints=2 * (int(OpenXRSpec.HandJointEXT.XR_HAND_JOINT_LITTLE_TIP_EXT) + 1),
            device=getattr(env.unwrapped, 'device', 'cpu'),
            hand_joint_names=getattr(getattr(getattr(env, 'scene', None), 'articulations', {{}}), "robot", None).data.joint_names[-22:] if hasattr(getattr(getattr(env, 'scene', None), 'articulations', {{}}), "robot") else None,
        )
        teleop_interface = OpenXRDevice(env_cfg.xr, retargeters=[gr1t2_retargeter])
        teleop_interface.add_callback("RESET", reset_env_and_player)
        teleop_interface.add_callback("START", start_teleoperation)
        teleop_interface.add_callback("STOP", stop_teleoperation)
        teleoperation_active = False
        raise NotImplementedError("dualhandtracking_abs with GR1T2 is not supported in this minimal robust version.")
    elif "handtracking" in args_cli.teleop_device.lower():
        RetargeterCls = Se3AbsRetargeter if "_abs" in args_cli.teleop_device.lower() else Se3RelRetargeter
        retargeter_device = RetargeterCls(bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, zero_out_xy_rotation=True)
        grip_retargeter = GripperRetargeter(bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT)
        teleop_interface = OpenXRDevice(env_cfg.xr, retargeters=[retargeter_device, grip_retargeter])
        teleop_interface.add_callback("RESET", reset_env_and_player)
        teleop_interface.add_callback("START", start_teleoperation)
        teleop_interface.add_callback("STOP", stop_teleoperation)
        teleoperation_active = False
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse', 'gamepad',"
            " 'handtracking', 'handtracking_abs'."
        )

    # Common callback for environment reset
    teleop_interface.add_callback("R", reset_env_and_player) # Changed name

    # Store the last teleop output for use in the callback
    last_teleop_output = None

    # Trajectory Player callbacks
    teleop_interface.add_callback("P", lambda: trajectory_player.record_current_pose(last_teleop_output))      # Record Pose
    teleop_interface.add_callback("L", lambda: trajectory_player.load_and_playback()) # Start Playback (loads + plays)
    teleop_interface.add_callback("M", trajectory_player.clear_waypoints)          # Clear Waypoints
    teleop_interface.add_callback("N", lambda: trajectory_player.save_waypoints()) # Save

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
    obs, info = env.reset()
    teleop_interface.reset()

    # Simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            if should_reset_recording_instance:
                obs, info = env.reset()
                if hasattr(teleop_interface, "reset"): teleop_interface.reset()
                should_reset_recording_instance = False

            raw_teleop_device_output = teleop_interface.advance()
            last_teleop_output = raw_teleop_device_output
            actions_to_step = None

            if trajectory_player.is_playing_back:
                playback_action_tuple = trajectory_player.get_formatted_action_for_playback(args_cli.task)
                if playback_action_tuple is not None:
                    actions_to_step = pre_process_actions(
                        playback_action_tuple,
                        getattr(env, 'num_envs', 1),
                        getattr(env, 'device', 'cpu'),
                        args_cli.task
                    )
            elif teleoperation_active:
                processed_input_for_action_fn = raw_teleop_device_output
                if "Reach" in args_cli.task:
                    try:
                        if hasattr(env.unwrapped.scene, 'sensors') and "ee_frame" in env.unwrapped.scene.sensors:
                            ee_sensor_reach = env.unwrapped.scene.sensors["ee_frame"]
                            if hasattr(ee_sensor_reach, 'data') and hasattr(ee_sensor_reach.data, 'target_pos_w'):
                                current_ee_pos_abs_tensor = ee_sensor_reach.data.target_pos_w[0, :3]
                            else:
                                current_ee_pos_abs_tensor = env.unwrapped.scene.articulations["robot"].data.ee_state_w[0, :3]
                        else:
                            current_ee_pos_abs_tensor = env.unwrapped.scene.articulations["robot"].data.ee_state_w[0, :3]
                        current_ee_pos_abs_np = current_ee_pos_abs_tensor.cpu().numpy().squeeze()
                        delta_pos_from_device_np = raw_teleop_device_output[0][:3]
                        target_abs_pos_for_reach_np = current_ee_pos_abs_np + delta_pos_from_device_np
                        processed_input_for_action_fn = (target_abs_pos_for_reach_np, raw_teleop_device_output[1])
                    except Exception as e:
                        print(f"[MainLoop Reach ERROR] Error getting current EE state: {e}")
                        actions_to_step = None
                if actions_to_step is None:
                    actions_to_step = pre_process_actions(
                        processed_input_for_action_fn,
                        getattr(env, 'num_envs', 1),
                        getattr(env, 'device', 'cpu'),
                        args_cli.task
                    )

            if actions_to_step is not None:
                obs, rewards, terminated, truncated, info = env.step(actions_to_step)

            else:
                env.sim.render()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
