# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json # For saving/loading waypoints

import numpy as np
import scipy.interpolate
from scipy.spatial.transform import Rotation, Slerp
import torch

from isaaclab.app import AppLauncher

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


class TrajectoryPlayer:
    def __init__(self, env, device_for_torch, steps_per_segment=150):
        self.env = env
        self.torch_device = device_for_torch
        self.recorded_waypoints = []  # List of {"position": np.array, "orientation_wxyz": np.array}
        self.playback_trajectory_abs_poses = []  # List of {"position": np.array, "orientation_wxyz": np.array}
        self.current_playback_idx = 0
        self.is_playing_back = False
        self.steps_per_segment = steps_per_segment
        self.gripper_command_during_playback = False  # Default gripper state for playback

        # Get the name of the robot's end-effector frame if available, otherwise default
        # This depends on how the specific environment is structured.
        # For now, we assume the relevant EE data is in env.unwrapped.scene["robot"].data.ee_state_w
        # which is common for many Isaac Lab manipulation tasks.

    def record_current_pose(self):
        if self.is_playing_back:
            print("Cannot record waypoints while playback is active. Stop playback first.")
            return

        # Assuming env_idx=0 for recording. ee_state_w format: (pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
        # This comes from the robot's data, usually for the first environment instance.
        current_ee_state_w_tensor = self.env.unwrapped.scene["robot"].data.ee_state_w[0]
        current_ee_state_w = current_ee_state_w_tensor.cpu().numpy()
        
        pos = current_ee_state_w[0:3]
        orient_wxyz = current_ee_state_w[3:7]  # Stored as w,x,y,z from Isaac Sim

        self.recorded_waypoints.append({"position": pos, "orientation_wxyz": orient_wxyz})
        print(f"Waypoint {len(self.recorded_waypoints)} recorded: pos={pos}, orient_wxyz={orient_wxyz}")

    def clear_waypoints(self):
        self.recorded_waypoints = []
        self.playback_trajectory_abs_poses = []
        self.is_playing_back = False
        print("Waypoints cleared.")

    def prepare_playback_trajectory(self):
        if len(self.recorded_waypoints) < 2:
            print("Not enough waypoints (need at least 2). Playback not started.")
            self.is_playing_back = False
            return

        self.playback_trajectory_abs_poses = []
        positions = np.array([wp["position"] for wp in self.recorded_waypoints])
        
        orientations_xyzw = []
        for wp in self.recorded_waypoints:
            wxyz = wp["orientation_wxyz"] # w,x,y,z
            orientations_xyzw.append([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]) # x,y,z,w for SciPy
        
        scipy_rotations = Rotation.from_quat(orientations_xyzw)
        
        num_segments = len(self.recorded_waypoints) - 1
        
        for i in range(num_segments):
            segment_times = np.linspace(0, 1, self.steps_per_segment, endpoint=(i == num_segments - 1)) # include endpoint only for last segment
            
            # Slerp for orientation for the segment [i, i+1]
            # We need to handle the case of single segment Slerp carefully if using Rotation.concatenate
            key_rots = Rotation.concatenate([scipy_rotations[i], scipy_rotations[i+1]])
            key_times = [0, 1]
            slerp_interpolator = Slerp(key_times, key_rots)

            for t_sample in segment_times:
                # Linear interpolation for position in this segment
                interp_pos = positions[i] * (1 - t_sample) + positions[i+1] * t_sample
                
                interp_rot_scipy = slerp_interpolator([t_sample])[0]
                interp_orient_xyzw = interp_rot_scipy.as_quat() # x,y,z,w
                interp_orient_wxyz = np.array([interp_orient_xyzw[3], interp_orient_xyzw[0], interp_orient_xyzw[1], interp_orient_xyzw[2]]) # w,x,y,z

                self.playback_trajectory_abs_poses.append({
                    "position": interp_pos,
                    "orientation_wxyz": interp_orient_wxyz
                })
        
        self.current_playback_idx = 0
        self.is_playing_back = True
        print(f"Playback trajectory prepared with {len(self.playback_trajectory_abs_poses)} steps.")

    def get_formatted_action_for_playback(self, task_name: str):
        if not self.is_playing_back or self.current_playback_idx >= len(self.playback_trajectory_abs_poses):
            self.is_playing_back = False
            if self.current_playback_idx > 0: # Avoid printing on first call if no trajectory
                print("Playback finished.")
            return None

        target_abs_pose = self.playback_trajectory_abs_poses[self.current_playback_idx]
        self.current_playback_idx += 1

        target_pos_abs = target_abs_pose["position"]
        target_rot_abs_wxyz = target_abs_pose["orientation_wxyz"]
        current_gripper_command = self.gripper_command_during_playback

        if "Reach" in task_name:
            # "Reach" tasks expect absolute position target. Gripper command is often ignored.
            return (target_pos_abs, current_gripper_command)
        else:
            # For "Lift" and other IK-Rel tasks, action is (delta_pos, delta_rot_axis_angle)
            current_ee_state_w_tensor = self.env.unwrapped.scene["robot"].data.ee_state_w[0]
            current_ee_state_w = current_ee_state_w_tensor.cpu().numpy()
            current_pos_abs = current_ee_state_w[0:3]
            current_rot_abs_wxyz = current_ee_state_w[3:7]

            delta_pos = target_pos_abs - current_pos_abs
            
            R_current = Rotation.from_quat([current_rot_abs_wxyz[1], current_rot_abs_wxyz[2], current_rot_abs_wxyz[3], current_rot_abs_wxyz[0]])
            R_target = Rotation.from_quat([target_rot_abs_wxyz[1], target_rot_abs_wxyz[2], target_rot_abs_wxyz[3], target_rot_abs_wxyz[0]])
            
            R_delta = R_target * R_current.inv()
            delta_rot_axis_angle = R_delta.as_rotvec()

            delta_pose_command = np.concatenate([delta_pos, delta_rot_axis_angle])
            return (delta_pose_command, current_gripper_command)

    def save_waypoints(self, filepath="waypoints.json"):
        waypoints_to_save = []
        for wp in self.recorded_waypoints:
            waypoints_to_save.append({
                "position": wp["position"].tolist(),
                "orientation_wxyz": wp["orientation_wxyz"].tolist()
            })
        try:
            with open(filepath, 'w') as f:
                json.dump(waypoints_to_save, f, indent=4)
            print(f"Waypoints saved to {filepath}")
        except Exception as e:
            omni.log.error(f"Error saving waypoints: {e}")

    def load_waypoints(self, filepath="waypoints.json"):
        try:
            with open(filepath, 'r') as f:
                loaded_wps_list = json.load(f)
            self.recorded_waypoints = []
            for wp_dict in loaded_wps_list:
                self.recorded_waypoints.append({
                    "position": np.array(wp_dict["position"]),
                    "orientation_wxyz": np.array(wp_dict["orientation_wxyz"])
                })
            print(f"Waypoints loaded from {filepath}. {len(self.recorded_waypoints)} waypoints found.")
            if len(self.recorded_waypoints) > 1:
                self.prepare_playback_trajectory()
            elif len(self.recorded_waypoints) > 0:
                omni.log.warn("Loaded waypoints, but need at least 2 to form a trajectory.")
        except FileNotFoundError:
            omni.log.error(f"Waypoint file {filepath} not found.")
        except Exception as e:
            omni.log.error(f"Error loading waypoints: {e}")

    def toggle_playback_gripper(self):
        self.gripper_command_during_playback = not self.gripper_command_during_playback
        print(f"Gripper command during playback set to: {'Close (grip)' if self.gripper_command_during_playback else 'Open (release)'}")


def pre_process_actions(
    teleop_data: tuple[np.ndarray, bool] | list[tuple[np.ndarray, np.ndarray, np.ndarray]], num_envs: int, device: str, task_name: str
) -> torch.Tensor:
    """Convert teleop data to the format expected by the environment action space.

    Args:
        teleop_data: Data from the teleoperation device.
        num_envs: Number of environments.
        device: Device to create tensors on.

    Returns:
        Processed actions as a tensor.
    """
    # compute actions based on environment
    if "Reach" in task_name:
        # For Reach, teleop_data is (abs_pose_xyz_numpy, gripper_command_bool)
        # The gripper_command is usually ignored by Reach envs.
        # Action is just the target absolute position.
        target_pos_abs, _ = teleop_data # gripper_command ignored
        actions = torch.tensor(target_pos_abs, dtype=torch.float, device=device).repeat(num_envs, 1)
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
        ).unsqueeze(0)
        # Concatenate arm poses and hand joint angles
        return actions
    else:
        # For other tasks like Lift, teleop_data is (delta_pose_numpy, gripper_command_bool)
        # delta_pose_numpy is [dx, dy, dz, d_ax, d_ay, d_az] (axis-angle for rotation)
        delta_pose, gripper_command = teleop_data
        delta_pose_tensor = torch.tensor(delta_pose, dtype=torch.float, device=device).repeat(num_envs, 1)
        
        # Gripper command: typically -1 for close (grip), 1 for open (release) in many envs
        # The boolean from teleop_interface: True for "grip" (close), False for "open" (release)
        gripper_val = -1.0 if gripper_command else 1.0
        gripper_vel = torch.full((delta_pose_tensor.shape[0], 1), gripper_val, dtype=torch.float, device=device)
        
        actions = torch.concat([delta_pose_tensor, gripper_vel], dim=1)
        return actions


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    elif "Reach" in args_cli.task:
         # For Reach, time_out is often useful to reset if stuck
        if env_cfg.terminations.time_out is not None:
            print("Reach task, keeping existing time_out termination.")
        else: # If it was None, set a default to avoid running forever if stuck
            env_cfg.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)
            
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Instantiate TrajectoryPlayer
    trajectory_player = TrajectoryPlayer(env, args_cli.device, steps_per_segment=100) # 100 steps per segment

    # Callback handlers
    def reset_recording_instance():
        """Reset the environment to its initial state.

        This callback is triggered when the user presses the reset key (typically 'R').
        It's useful when:
        - The robot gets into an undesirable configuration
        - The user wants to start over with the task
        - Objects in the scene need to be reset to their initial positions

        The environment will be reset on the next simulation step.
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        if trajectory_player.is_playing_back:
            trajectory_player.is_playing_back = False
            print("Playback stopped due to environment reset request.")

    def start_teleoperation():
        """Activate teleoperation control of the robot.

        This callback enables active control of the robot through the input device.
        It's typically triggered by a specific gesture or button press and is used when:
        - Beginning a new teleoperation session
        - Resuming control after temporarily pausing
        - Switching from observation mode to control mode

        While active, all commands from the device will be applied to the robot.
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation Activated.")

    def stop_teleoperation():
        """Deactivate teleoperation control of the robot.

        This callback temporarily suspends control of the robot through the input device.
        It's typically triggered by a specific gesture or button press and is used when:
        - Taking a break from controlling the robot
        - Repositioning the input device without moving the robot
        - Pausing to observe the scene without interference

        While inactive, the simulation continues to render but device commands are ignored.
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation Deactivated.")


    # create controller
    # Sensitivity values might need tuning based on task and user preference
    pos_sens = 0.05 * args_cli.sensitivity # Increased for finer control if needed
    rot_sens = 0.05 * args_cli.sensitivity
    
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(pos_sensitivity=pos_sens, rot_sensitivity=rot_sens)
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(pos_sensitivity=pos_sens, rot_sensitivity=rot_sens)
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(pos_sensitivity=pos_sens * 2, rot_sensitivity=rot_sens * 2) # Gamepads often need higher base
    elif "dualhandtracking_abs" in args_cli.teleop_device.lower() and "GR1T2" in args_cli.task:
        # Create GR1T2 retargeter with desired configuration
        gr1t2_retargeter = GR1T2Retargeter(
            enable_visualization=True,
            num_open_xr_hand_joints=2 * (int(OpenXRSpec.HandJointEXT.XR_HAND_JOINT_LITTLE_TIP_EXT) + 1),
            device=env.unwrapped.device,
            hand_joint_names=env.scene["robot"].data.joint_names[-22:],
        )

        # Create hand tracking device with retargeter
        teleop_interface = OpenXRDevice(
            env_cfg.xr,
            retargeters=[gr1t2_retargeter],
        )
        teleop_interface.add_callback("RESET", reset_recording_instance)
        teleop_interface.add_callback("START", start_teleoperation)
        teleop_interface.add_callback("STOP", stop_teleoperation)

        # Hand tracking needs explicit start gesture to activate
        teleoperation_active = False

    elif "handtracking" in args_cli.teleop_device.lower():
        # Create EE retargeter with desired configuration
        if "_abs" in args_cli.teleop_device.lower():
            retargeter_device = Se3AbsRetargeter(
                bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, zero_out_xy_rotation=True
            )
        else:
            retargeter_device = Se3RelRetargeter(
                bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, zero_out_xy_rotation=True
            )

        grip_retargeter = GripperRetargeter(bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT)

        # Create hand tracking device with retargeter (in a list)
        teleop_interface = OpenXRDevice(
            env_cfg.xr,
            retargeters=[retargeter_device, grip_retargeter],
        )
        teleop_interface.add_callback("RESET", reset_recording_instance)
        teleop_interface.add_callback("START", start_teleoperation)
        teleop_interface.add_callback("STOP", stop_teleoperation)

        # Hand tracking needs explicit start gesture to activate
        teleoperation_active = False
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse', 'gamepad',"
            " 'handtracking', 'handtracking_abs'."
        )

    # add teleoperation key for env reset (for all devices)
    teleop_interface.add_callback("R", reset_recording_instance)
    print(teleop_interface)


    # Trajectory Player callbacks
    teleop_interface.add_callback("P", trajectory_player.record_current_pose)
    teleop_interface.add_callback("L", trajectory_player.prepare_playback_trajectory)
    teleop_interface.add_callback("C", trajectory_player.clear_waypoints)
    teleop_interface.add_callback("K", lambda: trajectory_player.save_waypoints("waypoints.json"))
    teleop_interface.add_callback("O", lambda: trajectory_player.load_waypoints("waypoints.json"))
    teleop_interface.add_callback("G", trajectory_player.toggle_playback_gripper) # Toggle gripper for playback

    print("\n--- Teleoperation Interface Controls ---")
    print(teleop_interface) # Prints device specific controls
    print("\n--- Trajectory Player Controls ---")
    print("  P: Record current EE pose as waypoint.")
    print("  L: Prepare and start playback of recorded trajectory.")
    print("  C: Clear all recorded waypoints from memory.")
    print("  K: Save current waypoints to 'waypoints.json'.")
    print("  O: Load waypoints from 'waypoints.json' and prepare for playback.")
    print("  G: Toggle gripper command for PLAYBACK (Open/Close).")
    print("  R: Reset environment (also stops playback).")
    print("------------------------------------\n")


    # reset environment
    env.reset()
    # Se3Keyboard/Mouse/Gamepad have reset(), OpenXRDevice might not.
    if hasattr(teleop_interface, "reset"):
        teleop_interface.reset()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if should_reset_recording_instance:
                env.reset()
                if hasattr(teleop_interface, "reset"): teleop_interface.reset()
                should_reset_recording_instance = False
                # trajectory_player.is_playing_back is already handled by reset_recording_instance callback

            # Advance teleop interface to process key presses for callbacks AND get raw teleop commands
            raw_teleop_data = teleop_interface.advance()

            if trajectory_player.is_playing_back:
                playback_action_tuple = trajectory_player.get_formatted_action_for_playback(args_cli.task)
                if playback_action_tuple is not None:
                    actions = pre_process_actions(playback_action_tuple, env.num_envs, env.device, args_cli.task)
                    env.step(actions)
                else: # Playback finished or issue
                    env.sim.render() # Just render, user can resume teleop or restart playback
            elif teleoperation_active: # Not playing back, and teleop is active
                actions = pre_process_actions(raw_teleop_data, env.num_envs, env.device, args_cli.task)
                env.step(actions)
            else: # Not playing back, and teleop is not active (e.g. handtracking waiting for start)
                env.sim.render()
                
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
