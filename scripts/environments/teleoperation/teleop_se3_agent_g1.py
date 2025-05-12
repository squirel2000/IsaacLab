# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json # For saving/loading waypoints

import numpy as np
import scipy.interpolate # Keep this for Slerp if needed, but Rotation object has Slerp too
from scipy.spatial.transform import Rotation, Slerp # Slerp specific import
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
    def __init__(self, env, device_for_torch, steps_per_segment=100): # Reduced default for faster playback
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
        print("[TrajectoryPlayer DEBUG] Inside record_current_pose.")
        if self.is_playing_back:
            print("Cannot record waypoints while playback is active. Stop playback first.")
            return

        try:
            print(f"[TrajectoryPlayer DEBUG] dir(self.env.unwrapped.scene): {dir(self.env.unwrapped.scene)}")
            
            # --- BEGIN SENSOR DEBUG ---
            if hasattr(self.env.unwrapped.scene, 'sensors'):
                print(f"[TrajectoryPlayer DEBUG] dir(self.env.unwrapped.scene.sensors): {dir(self.env.unwrapped.scene.sensors)}")
                if "ee_frame" in self.env.unwrapped.scene.sensors:
                    ee_sensor = self.env.unwrapped.scene.sensors["ee_frame"]
                    print(f"[TrajectoryPlayer DEBUG] type(ee_sensor): {type(ee_sensor)}")
                    print(f"[TrajectoryPlayer DEBUG] dir(ee_sensor): {dir(ee_sensor)}")
                    if hasattr(ee_sensor, 'data'):
                        print(f"[TrajectoryPlayer DEBUG] type(ee_sensor.data): {type(ee_sensor.data)}")
                        print(f"[TrajectoryPlayer DEBUG] dir(ee_sensor.data): {dir(ee_sensor.data)}")
                        
                        # Attempt to get pose from sensor data
                        pos_tensor = ee_sensor.data.target_pos_w[0]  # Use target_pos_w
                        orient_tensor_wxyz = ee_sensor.data.target_quat_w[0] # Use target_quat_w
                        
                        # Ensure pos and orient_wxyz are 1D arrays
                        pos = pos_tensor.cpu().numpy().squeeze()
                        orient_wxyz = orient_tensor_wxyz.cpu().numpy().squeeze()
                        
                        self.recorded_waypoints.append({"position": pos, "orientation_wxyz": orient_wxyz})
                        print(f"Waypoint {len(self.recorded_waypoints)} recorded (from sensor): pos={pos}, orient_wxyz={orient_wxyz}")
                        return # Successfully recorded from sensor
                    else:
                        print("[TrajectoryPlayer DEBUG] ee_sensor does not have 'data' attribute.")
                else:
                    print("[TrajectoryPlayer DEBUG] 'ee_frame' sensor not found in scene.sensors. Available sensors: {list(self.env.unwrapped.scene.sensors.keys())}")
            else:
                print("[TrajectoryPlayer DEBUG] self.env.unwrapped.scene does not have 'sensors' attribute.")
            # --- END SENSOR DEBUG ---

            # Fallback to articulation data if sensor method fails or is not available
            print("[TrajectoryPlayer DEBUG] Falling back to articulation data for EE pose.")
            if "robot" in self.env.unwrapped.scene.articulations:
                robot_object = self.env.unwrapped.scene.articulations["robot"]
                print(f"[TrajectoryPlayer DEBUG] type(robot_object): {type(robot_object)}")
                print(f"[TrajectoryPlayer DEBUG] dir(robot_object): {dir(robot_object)}")
                if hasattr(robot_object, 'data'):
                    robot_data = robot_object.data
                    print(f"[TrajectoryPlayer DEBUG] type(robot_data): {type(robot_data)}")
                    print(f"[TrajectoryPlayer DEBUG] dir(robot_data): {dir(robot_data)}")
                    # The original error was here: 'ArticulationData' object has no attribute 'ee_state_w'
                    # We need to find the correct attribute in robot_data for the EE pose.
                    # For now, this will still error out if ee_state_w is not the correct path.
                    # We will analyze dir(robot_data) from the output to find the correct attribute.
                    current_ee_state_w_tensor = robot_data.ee_state_w[0] # This line will likely still fail
                    current_ee_state_w = current_ee_state_w_tensor.cpu().numpy()
                    pos = current_ee_state_w[0:3]
                else:
                    print("[TrajectoryPlayer DEBUG] robot_object does not have attribute 'data'")
                    return # Cannot proceed without robot data
            else:
                print("[TrajectoryPlayer DEBUG] 'robot' not in self.env.unwrapped.scene.articulations. Available: {list(self.env.unwrapped.scene.articulations.keys())}")
                return # Cannot proceed if robot articulation is not found
            
            # This part will only be reached if the articulation fallback is attempted AND ee_state_w exists (which it doesn't)
            # or if we find the correct attribute later.
            orient_wxyz = current_ee_state_w[3:7]  # Stored as w,x,y,z from Isaac Sim

            self.recorded_waypoints.append({"position": pos, "orientation_wxyz": orient_wxyz})
            print(f"Waypoint {len(self.recorded_waypoints)} recorded: pos={pos}, orient_wxyz={orient_wxyz}")

        except AttributeError as e:
            print(f"[TrajectoryPlayer ERROR] AttributeError in record_current_pose: {e}")
            import traceback
            traceback.print_exc()
        except KeyError as e:
            print(f"[TrajectoryPlayer ERROR] KeyError in record_current_pose (likely 'robot' not in scene): {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"[TrajectoryPlayer ERROR] Unexpected error in record_current_pose: {e}")
            import traceback
            traceback.print_exc()

    def clear_waypoints(self):
        self.recorded_waypoints = []
        self.playback_trajectory_abs_poses = []
        if self.is_playing_back:
            self.is_playing_back = False
            print("Playback stopped and waypoints cleared.")
        else:
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
            # Create N points for the segment, where N = self.steps_per_segment
            # endpoint=True for the last segment, endpoint=False for others to avoid duplicating waypoints
            # (as the start of the next segment will be that waypoint)
            is_last_segment = (i == num_segments - 1)
            num_points_in_segment = self.steps_per_segment
            
            # linspace(start, stop, num, endpoint)
            # if endpoint=False, last point is stop - (stop-start)/num
            # if endpoint=True, last point is stop
            # We want to generate `steps_per_segment` steps *between* waypoints.
            # So, `steps_per_segment + 1` points in total for a segment if we include both ends.
            # The first point of the segment is waypoint[i].
            # The Nth step lands on waypoint[i+1].
            # So we interpolate N times.

            segment_times = np.linspace(0, 1, num_points_in_segment, endpoint=True) # Always include endpoint of sub-segment
            if not is_last_segment :
                 segment_times = segment_times[:-1] # Exclude the last point if not the final segment


            key_rots_segment = Rotation.concatenate([scipy_rotations[i], scipy_rotations[i+1]])
            slerp_interpolator = Slerp([0, 1], key_rots_segment)

            for t_sample_idx, t_sample in enumerate(segment_times):
                interp_pos = positions[i] * (1 - t_sample) + positions[i+1] * t_sample
                interp_rot_scipy = slerp_interpolator([t_sample])[0]
                interp_orient_xyzw = interp_rot_scipy.as_quat()
                interp_orient_wxyz = np.array([interp_orient_xyzw[3], interp_orient_xyzw[0], interp_orient_xyzw[1], interp_orient_xyzw[2]])

                self.playback_trajectory_abs_poses.append({
                    "position": interp_pos,
                    "orientation_wxyz": interp_orient_wxyz
                })
        
        # Add the very last waypoint explicitly if not perfectly included by linspace due to floating points
        if len(self.playback_trajectory_abs_poses) > 0 and not np.allclose(self.playback_trajectory_abs_poses[-1]["position"], positions[-1]):
             self.playback_trajectory_abs_poses.append({
                "position": positions[-1],
                "orientation_wxyz": np.array([scipy_rotations[-1].as_quat()[3], *scipy_rotations[-1].as_quat()[:3]])
            })


        self.current_playback_idx = 0
        self.is_playing_back = True
        print(f"Playback trajectory prepared with {len(self.playback_trajectory_abs_poses)} steps.")

    def get_formatted_action_for_playback(self, task_name: str):
        if not self.is_playing_back or self.current_playback_idx >= len(self.playback_trajectory_abs_poses):
            self.is_playing_back = False
            if self.current_playback_idx > 0 and len(self.playback_trajectory_abs_poses) > 0 : # Avoid print if never started
                print("Playback finished.")
            return None

        target_abs_pose = self.playback_trajectory_abs_poses[self.current_playback_idx]
        self.current_playback_idx += 1

        target_pos_abs_3D = target_abs_pose["position"] # This is 3D
        target_rot_abs_wxyz = target_abs_pose["orientation_wxyz"] # This is 4D (w,x,y,z)
        
        # Gripper command from playback setting
        # True for "grip" (close), False for "open" (release)
        current_gripper_command_bool = self.gripper_command_during_playback

        if "Reach" in task_name:
            # "Reach" tasks expect (absolute_3D_position_numpy, gripper_command_bool)
            return (target_pos_abs_3D, current_gripper_command_bool)
        else: # e.g. "Lift" (IK-Rel)
            # IK-Rel tasks expect (delta_pose_6D_numpy, gripper_command_bool)
            # delta_pose_6D_numpy is [dx, dy, dz, d_ax, d_ay, d_az] (axis-angle for rotation)
            
            # Accessing robot data for playback - ensure this is also robust or uses the corrected path once found
            try:
                # Try sensor first for playback as well
                if hasattr(self.env.unwrapped.scene, 'sensors') and "ee_frame" in self.env.unwrapped.scene.sensors:
                    ee_sensor = self.env.unwrapped.scene.sensors["ee_frame"]
                    if hasattr(ee_sensor, 'data') and hasattr(ee_sensor.data, 'target_pos_w') and hasattr(ee_sensor.data, 'target_quat_w'): # Check for target_ attributes
                        current_pos_abs = ee_sensor.data.target_pos_w[0].cpu().numpy().squeeze() # Use target_pos_w and squeeze
                        current_rot_abs_wxyz = ee_sensor.data.target_quat_w[0].cpu().numpy().squeeze() # Use target_quat_w and squeeze
                        print("[TrajectoryPlayer Playback DEBUG] Using sensor data for current EE pose.")
                    else: # Fallback if sensor data attributes are missing
                        print("[TrajectoryPlayer Playback DEBUG] Sensor 'ee_frame' found but data attributes missing, falling back to articulation for playback.")
                        robot_data_playback = self.env.unwrapped.scene.articulations["robot"].data
                        # This will fail if ee_state_w is not the correct attribute
                        current_ee_state_w_playback = robot_data_playback.ee_state_w[0].cpu().numpy()
                        current_pos_abs = current_ee_state_w_playback[0:3]
                        current_rot_abs_wxyz = current_ee_state_w_playback[3:7]
                else: # Fallback if sensor itself is missing
                    print("[TrajectoryPlayer Playback DEBUG] Sensor 'ee_frame' not found, falling back to articulation for playback.")
                    robot_data_playback = self.env.unwrapped.scene.articulations["robot"].data
                    # This will fail if ee_state_w is not the correct attribute
                    current_ee_state_w_playback = robot_data_playback.ee_state_w[0].cpu().numpy()
                    current_pos_abs = current_ee_state_w_playback[0:3]
                    current_rot_abs_wxyz = current_ee_state_w_playback[3:7] # Corrected this line

                delta_pos = target_pos_abs_3D - current_pos_abs
                
                R_current = Rotation.from_quat([current_rot_abs_wxyz[1], current_rot_abs_wxyz[2], current_rot_abs_wxyz[3], current_rot_abs_wxyz[0]]) # xyzw
                R_target = Rotation.from_quat([target_rot_abs_wxyz[1], target_rot_abs_wxyz[2], target_rot_abs_wxyz[3], target_rot_abs_wxyz[0]]) # xyzw
                
                R_delta = R_target * R_current.inv()
                delta_rot_axis_angle = R_delta.as_rotvec() # Returns 3D axis-angle

                delta_pose_command_6D = np.concatenate([delta_pos, delta_rot_axis_angle])
                return (delta_pose_command_6D, current_gripper_command_bool)
            except AttributeError as e:
                print(f"[TrajectoryPlayer Playback ERROR] AttributeError getting current EE state: {e}")
                # Potentially stop playback or return a neutral action if state can't be read
                self.is_playing_back = False 
                return None # Stop playback if we can't get current state
            except KeyError as e:
                print(f"[TrajectoryPlayer Playback ERROR] KeyError getting current EE state (likely 'robot' not in scene): {e}")
                self.is_playing_back = False
                return None
            except Exception as e:
                print(f"[TrajectoryPlayer Playback ERROR] Unexpected error getting current EE state: {e}")
                self.is_playing_back = False
                return None

    def save_waypoints(self, filepath="waypoints.json"):
        if not self.recorded_waypoints:
            print("No waypoints to save.")
            return
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
            print(f"[TrajectoryPlayer ERROR] Error saving waypoints to {filepath}: {e}") # Changed to print
            import traceback
            traceback.print_exc()

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
                # Optionally, directly prepare for playback after loading
                # self.prepare_playback_trajectory()
                print("Press 'L' to start playback with loaded waypoints.")
            elif len(self.recorded_waypoints) > 0:
                print("Loaded waypoints, but need at least 2 to form a trajectory.")
        except FileNotFoundError:
            print(f"Waypoint file {filepath} not found. No waypoints loaded.")
        except Exception as e:
            print(f"Error loading waypoints: {e}")

    def toggle_playback_gripper(self):
        self.gripper_command_during_playback = not self.gripper_command_during_playback
        state = "Close (grip)" if self.gripper_command_during_playback else "Open (release)"
        print(f"Gripper command during PLAYBACK set to: {state}")


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
    env_cfg.env_name = args_cli.task
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    elif "Reach" in args_cli.task:
        if env_cfg.terminations.time_out is None : # Ensure there is a timeout for Reach tasks
             print(f"Reach task '{args_cli.task}' did not have a time_out termination. Adding a default one.")
             env_cfg.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True) # Assuming mdp.time_out is available

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        print(
            f"The environment '{args_cli.task}' uses absolute 3D position control for the end-effector. "
            "Orientation commands from SE3 devices will be used to calculate deltas but only position is sent."
        )

    omni.log.warn(f"args_cli.task: {args_cli.task}")
    print(f"args_cli: {args_cli}")
    print(f"args_cli.task: {args_cli.task}")
    print(f"args_cli.teleop_device: {args_cli.teleop_device}")

    should_reset_recording_instance = False
    teleoperation_active = True # Default to active for keyboard/spacemouse/gamepad

    trajectory_player = TrajectoryPlayer(env, args_cli.device, steps_per_segment=100)

    def reset_env_and_player(): # Renamed for clarity
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        if trajectory_player.is_playing_back:
            trajectory_player.is_playing_back = False
            omni.log.info("Playback stopped due to environment reset request.")

    def start_teleoperation():
        nonlocal teleoperation_active
        teleoperation_active = True
        omni.log.info("Teleoperation Activated.")

    def stop_teleoperation():
        nonlocal teleoperation_active
        teleoperation_active = False
        omni.log.info("Teleoperation Deactivated.")

    pos_sens = 0.4 * args_cli.sensitivity
    rot_sens = 0.4 * args_cli.sensitivity

    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(pos_sensitivity=pos_sens, rot_sensitivity=rot_sens)
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(pos_sensitivity=pos_sens, rot_sensitivity=rot_sens)
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(pos_sensitivity=pos_sens * 2, rot_sensitivity=rot_sens * 2)
    elif "dualhandtracking_abs" in args_cli.teleop_device.lower() and "GR1T2" in args_cli.task:
        gr1t2_retargeter = GR1T2Retargeter(
            enable_visualization=True,
            num_open_xr_hand_joints=2 * (int(OpenXRSpec.HandJointEXT.XR_HAND_JOINT_LITTLE_TIP_EXT) + 1),
            device=env.unwrapped.device,
            hand_joint_names=env.scene.articulations["robot"].data.joint_names[-22:],
        )
        teleop_interface = OpenXRDevice(env_cfg.xr, retargeters=[gr1t2_retargeter])
        teleop_interface.add_callback("RESET", reset_env_and_player) # Changed name
        teleop_interface.add_callback("START", start_teleoperation)
        teleop_interface.add_callback("STOP", stop_teleoperation)
        teleoperation_active = False # Hand tracking needs explicit start
    elif "handtracking" in args_cli.teleop_device.lower():
        RetargeterCls = Se3AbsRetargeter if "_abs" in args_cli.teleop_device.lower() else Se3RelRetargeter
        retargeter_device = RetargeterCls(bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, zero_out_xy_rotation=True)
        grip_retargeter = GripperRetargeter(bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT)
        teleop_interface = OpenXRDevice(env_cfg.xr, retargeters=[retargeter_device, grip_retargeter])
        teleop_interface.add_callback("RESET", reset_env_and_player) # Changed name
        teleop_interface.add_callback("START", start_teleoperation)
        teleop_interface.add_callback("STOP", stop_teleoperation)
        teleoperation_active = False # Hand tracking needs explicit start
    else:
        raise ValueError(f"Invalid device: {args_cli.teleop_device}")

    # Common callback for environment reset
    teleop_interface.add_callback("R", reset_env_and_player) # Changed name

    # Trajectory Player callbacks - **USING NON-CONFLICTING KEYS**
    teleop_interface.add_callback("P", trajectory_player.record_current_pose)      # Record Pose
    teleop_interface.add_callback("L", trajectory_player.prepare_playback_trajectory) # Start Playback
    teleop_interface.add_callback("M", trajectory_player.clear_waypoints)          # Clear Waypoints (was C)
    teleop_interface.add_callback("N", lambda: trajectory_player.save_waypoints("waypoints.json")) # Save (was K)
    teleop_interface.add_callback("O", lambda: trajectory_player.load_waypoints("waypoints.json")) # Load
    teleop_interface.add_callback("B", trajectory_player.toggle_playback_gripper)  # Toggle Playback Gripper (was G)

    print("\n--- Teleoperation Interface Controls ---")
    # This print(teleop_interface) is fine, it shows the Se3Keyboard's own mapping
    # The duplicate print was removed from earlier in the code.
    print(teleop_interface)
    print("\n--- Trajectory Player Controls (Non-conflicting) ---")
    print("  P: Record current EE pose as waypoint.")
    print("  L: Prepare and start playback of recorded trajectory.")
    print("  M: Clear all recorded waypoints from memory. (was C)")
    print("  N: Save current waypoints to 'waypoints.json'. (was K)")
    print("  O: Load waypoints from 'waypoints.json'.")
    print("  B: Toggle gripper command for PLAYBACK (Open/Close). (was G)")
    print("  R: Reset environment (also stops playback).")
    print("------------------------------------\n")

    # Initialize environment and teleop interface
    obs, info = env.reset() # Get initial observation
    if hasattr(teleop_interface, "reset"):
        teleop_interface.reset()

    # Simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            if should_reset_recording_instance:
                obs, info = env.reset()
                if hasattr(teleop_interface, "reset"): teleop_interface.reset()
                should_reset_recording_instance = False
                # trajectory_player.is_playing_back is handled by reset_env_and_player callback

            # teleop_interface.advance() polls keys and triggers callbacks like P, L, M, N, O, B, R
            # AND returns raw delta commands from keyboard if no other callback consumed the event
            raw_teleop_device_output = teleop_interface.advance()

            actions_to_step = None

            if trajectory_player.is_playing_back:
                # Get (pose_data_np, gripper_bool) from player
                playback_action_tuple = trajectory_player.get_formatted_action_for_playback(args_cli.task)
                if playback_action_tuple is not None:
                    actions_to_step = pre_process_actions(playback_action_tuple, env.num_envs, env.device, args_cli.task)
            elif teleoperation_active:
                # Manual teleoperation is active
                # raw_teleop_device_output is (delta_pose_6D_np, gripper_bool_from_device_toggle)
                
                processed_input_for_action_fn = raw_teleop_device_output # Default for IK-Rel

                if "Reach" in args_cli.task:
                    # For "Reach", convert device's delta_pos to an absolute_pos target
                    # raw_teleop_device_output[0] is 6D delta [dx,dy,dz,dax,day,daz]
                    # raw_teleop_device_output[1] is gripper command from device (e.g. keyboard 'K')
                    
                    # Get current EE state (position part) for the first environment
                    # This must be done *after* any potential env.reset() and *before* env.step()

                    # Accessing robot data for Reach task - ensure this is also robust
                    try:
                        # Try sensor first for Reach task as well
                        if hasattr(env.unwrapped.scene, 'sensors') and "ee_frame" in env.unwrapped.scene.sensors:
                            ee_sensor_reach = env.unwrapped.scene.sensors["ee_frame"]
                            if hasattr(ee_sensor_reach, 'data') and hasattr(ee_sensor_reach.data, 'target_pos_w'): # Check for target_pos_w
                                current_ee_pos_abs_tensor = ee_sensor_reach.data.target_pos_w[0, :3] # on device, use target_pos_w
                                print("[MainLoop Reach DEBUG] Using sensor data for current EE pose.")
                            else: # Fallback if sensor data attributes are missing
                                print("[MainLoop Reach DEBUG] Sensor 'ee_frame' found but data attributes missing, falling back to articulation.")
                                current_ee_pos_abs_tensor = env.unwrapped.scene.articulations["robot"].data.ee_state_w[0, :3] # on device
                        else: # Fallback if sensor itself is missing
                            print("[MainLoop Reach DEBUG] Sensor 'ee_frame' not found, falling back to articulation.")
                            current_ee_pos_abs_tensor = env.unwrapped.scene.articulations["robot"].data.ee_state_w[0, :3] # on device
                        
                        current_ee_pos_abs_np = current_ee_pos_abs_tensor.cpu().numpy().squeeze() # Add squeeze here
                        delta_pos_from_device_np = raw_teleop_device_output[0][:3] # Take only dx,dy,dz
                        
                        target_abs_pos_for_reach_np = current_ee_pos_abs_np + delta_pos_from_device_np
                        
                        # pre_process_actions for "Reach" expects (abs_pos_3D_np, gripper_bool)
                        processed_input_for_action_fn = (target_abs_pos_for_reach_np, raw_teleop_device_output[1])
                    except AttributeError as e:
                        print(f"[MainLoop Reach ERROR] AttributeError getting current EE state: {e}")
                        actions_to_step = None # Don't attempt to step if we can't get state
                    except KeyError as e:
                        print(f"[MainLoop Reach ERROR] KeyError getting current EE state (likely 'robot' not in scene): {e}")
                        actions_to_step = None
                    except Exception as e:
                        print(f"[MainLoop Reach ERROR] Unexpected error getting current EE state: {e}")
                        actions_to_step = None
                
                if actions_to_step is not None: # Check if processed_input_for_action_fn was successfully created
                    actions_to_step = pre_process_actions(processed_input_for_action_fn, env.num_envs, env.device, args_cli.task)

            # Apply actions if any, otherwise just render
                    
                    target_abs_pos_for_reach_np = current_ee_pos_abs_np + delta_pos_from_device_np
                    
                    # pre_process_actions for "Reach" expects (abs_pos_3D_np, gripper_bool)
                    processed_input_for_action_fn = (target_abs_pos_for_reach_np, raw_teleop_device_output[1])

                actions_to_step = pre_process_actions(processed_input_for_action_fn, env.num_envs, env.device, args_cli.task)

            # Apply actions if any, otherwise just render
            if actions_to_step is not None:
                obs, rewards, terminated, truncated, info = env.step(actions_to_step)
                # Optional: print rewards or other info
                # if env.num_envs == 1 and rewards.item() != 0:
                #    print(f"Reward: {rewards.item():.4f}")
            else:
                # If no actions (e.g., playback finished or teleop not active), just render the simulation
                env.sim.render() # Ensures simulation view updates

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
