# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility class for recording and playing back end-effector trajectories."""

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import omni.log

# Default path for saving/loading waypoints
WAYPOINTS_JSON_PATH = os.path.join("logs", "teleoperation", "waypoints.json")

class TrajectoryPlayer:
    """
    Handles recording, saving, loading, and playing back end-effector trajectories.

    A trajectory is a sequence of waypoints, where each waypoint includes the
    end-effector's absolute position, orientation (as a quaternion), and the
    gripper command state.

    Playback interpolates linearly between positions and uses spherical linear
    interpolation (Slerp) for orientations. Gripper state is held constant
    between waypoints during playback.
    """
    def __init__(self, env, device_for_torch, steps_per_segment=100):
        """
        Initializes the TrajectoryPlayer.

        Args:
            env: The Isaac Lab environment instance. Used to access the robot's
                 current state (e.g., end-effector pose).
            device_for_torch: The torch device to use for tensor operations.
            steps_per_segment: The number of simulation steps to use for
                               interpolating between two consecutive waypoints
                               during playback.
        """
        self.env = env
        self.torch_device = device_for_torch
        # List of recorded waypoints. Each waypoint is a dict:
        # {"position": np.array, "orientation_wxyz": np.array, "gripper": bool}
        self.recorded_waypoints = []
        # List of interpolated poses for playback. Each element is a dict:
        # {"position": np.array, "orientation_wxyz": np.array, "gripper": bool}
        self.playback_trajectory_abs_poses = []
        self.current_playback_idx = 0
        self.is_playing_back = False
        self.steps_per_segment = steps_per_segment

    def record_current_pose(self, teleop_output=None):
        """
        Record the current end-effector pose and gripper command as a waypoint.

        If teleop_output is provided, extracts the gripper command from it.
        Requires the environment's scene to have an "ee_frame" sensor with
        'target_pos_w' and 'target_quat_w' data attributes.

        Args:
            teleop_output: Optional. The raw output from the teleoperation device.
                           Expected to be a tuple where the second element is
                           the boolean gripper command.
        """
        if self.is_playing_back:
            print("Cannot record waypoints while playback is active. Stop playback first.")
            return

        # Extract gripper command from teleop_output if available
        gripper_command = False
        if teleop_output is not None and isinstance(teleop_output, (tuple, list)) and len(teleop_output) > 1:
            gripper_command = teleop_output[1]

        try:
            # Access the end-effector sensor data to get the current pose
            if hasattr(self.env.unwrapped.scene, 'sensors') and "ee_frame" in self.env.unwrapped.scene.sensors:
                ee_sensor = self.env.unwrapped.scene.sensors["ee_frame"]
                if hasattr(ee_sensor, 'data') and hasattr(ee_sensor.data, 'target_pos_w') and hasattr(ee_sensor.data, 'target_quat_w'):
                    # Use target_pos_w and target_quat_w which represent the commanded pose
                    pos_tensor = ee_sensor.data.target_pos_w[0]
                    orient_tensor_wxyz = ee_sensor.data.target_quat_w[0]

                    # Convert tensors to numpy arrays and squeeze to remove batch dimension
                    pos = pos_tensor.cpu().numpy().squeeze()
                    orient_wxyz = orient_tensor_wxyz.cpu().numpy().squeeze()
                    gripper = gripper_command

                    # Store the waypoint
                    self.recorded_waypoints.append({
                        "position": pos,
                        "orientation_wxyz": orient_wxyz,
                        "gripper": gripper
                    })
                    print(f"Waypoint {len(self.recorded_waypoints)} recorded: pos={pos}, orient_wxyz={orient_wxyz}, gripper={gripper}")
                    return
                else:
                    omni.log.warn("[TrajectoryPlayer] EE sensor data or required attributes not found.")
            else:
                omni.log.warn("[TrajectoryPlayer] 'ee_frame' sensor not found in scene.sensors.")
        except Exception as e:
            omni.log.error(f"[TrajectoryPlayer ERROR] Unexpected error in record_current_pose: {e}")
            import traceback
            traceback.print_exc()

    def clear_waypoints(self):
        """
        Clears all recorded waypoints and stops playback if active.
        """
        self.recorded_waypoints = []
        self.playback_trajectory_abs_poses = []
        if self.is_playing_back:
            self.is_playing_back = False
            print("Playback stopped and waypoints cleared.")
        else:
            print("Waypoints cleared.")

    def load_and_playback(self, filepath=WAYPOINTS_JSON_PATH):
        """
        Loads waypoints from a file and prepares the trajectory for playback.

        Args:
            filepath: The path to the JSON file containing waypoints.
        """
        self.load_waypoints(filepath)
        self.prepare_playback_trajectory()

    def prepare_playback_trajectory(self):
        """
        Generates the interpolated trajectory steps from the recorded waypoints.

        Uses linear interpolation for position and Slerp for orientation
        between consecutive waypoints.
        """
        if len(self.recorded_waypoints) < 2:
            print("Not enough waypoints (need at least 2). Playback not started.")
            self.is_playing_back = False
            return

        self.playback_trajectory_abs_poses = []
        positions = np.array([wp["position"] for wp in self.recorded_waypoints])

        # Convert wxyz quaternions to xyzw format for SciPy Rotation
        orientations_xyzw = []
        for wp in self.recorded_waypoints:
            wxyz = wp["orientation_wxyz"]
            orientations_xyzw.append([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

        scipy_rotations = Rotation.from_quat(orientations_xyzw)

        num_segments = len(self.recorded_waypoints) - 1

        # Interpolate each segment
        for i in range(num_segments):
            # Create time points for interpolation within the segment
            num_points_in_segment = self.steps_per_segment
            # Exclude the last point for all but the final segment to avoid duplicates
            segment_times = np.linspace(0, 1, num_points_in_segment, endpoint=(i == num_segments - 1))

            # Interpolate rotation using Slerp
            key_rots_segment = Rotation.concatenate([scipy_rotations[i], scipy_rotations[i+1]])
            slerp_interpolator = Slerp([0, 1], key_rots_segment)
            interp_rot_scipy = slerp_interpolator(segment_times)

            # Interpolate position linearly
            interp_pos = positions[i, None] * (1 - segment_times[:, None]) + positions[i+1, None] * segment_times[:, None]

            # Get gripper state (held constant for the segment from the start waypoint)
            interp_gripper = self.recorded_waypoints[i]["gripper"]

            # Convert interpolated SciPy rotations back to wxyz numpy arrays
            interp_orient_xyzw = interp_rot_scipy.as_quat()
            interp_orient_wxyz = np.stack([interp_orient_xyzw[:, 3], interp_orient_xyzw[:, 0], interp_orient_xyzw[:, 1], interp_orient_xyzw[:, 2]], axis=1)

            # Append interpolated poses to the playback trajectory
            for j in range(len(segment_times)):
                 self.playback_trajectory_abs_poses.append({
                    "position": interp_pos[j],
                    "orientation_wxyz": interp_orient_wxyz[j],
                    "gripper": interp_gripper
                })

        self.current_playback_idx = 0
        self.is_playing_back = True
        print(f"Playback trajectory prepared with {len(self.playback_trajectory_abs_poses)} steps.")

    def get_formatted_action_for_playback(self, task_name: str):
        """
        Gets the next action command from the playback trajectory.

        Args:
            task_name: The name of the current task environment.

        Returns:
            A tuple representing the action command for the environment, or None
            if playback is finished or an error occurs.
        """
        if not self.is_playing_back or self.current_playback_idx >= len(self.playback_trajectory_abs_poses):
            self.is_playing_back = False
            if self.current_playback_idx > 0 and len(self.playback_trajectory_abs_poses) > 0 : # Avoid print if never started
                print("Playback finished.")
            return None

        # Get the target pose and gripper command for the current step
        target_abs_pose = self.playback_trajectory_abs_poses[self.current_playback_idx]
        self.current_playback_idx += 1
        target_pos_abs_3D = target_abs_pose["position"]
        target_rot_abs_wxyz = target_abs_pose["orientation_wxyz"]
        current_gripper_command_bool = target_abs_pose["gripper"]

        if "Reach" in task_name:
            # "Reach" tasks expect (absolute_3D_position_numpy, gripper_command_bool)
            # Note: Gripper command is often ignored by "Reach" tasks, but we pass it for consistency.
            return (target_pos_abs_3D, current_gripper_command_bool)
        else: # e.g. "Lift" (IK-Rel)
            # IK-Rel tasks expect (delta_pose_6D_numpy, gripper_command_bool)
            # delta_pose_6D_numpy is [dx, dy, dz, d_ax, d_ay, d_az] (axis-angle for rotation)

            # Need the current EE pose to calculate the delta pose
            try:
                if hasattr(self.env.unwrapped.scene, 'sensors') and "ee_frame" in self.env.unwrapped.scene.sensors:
                    ee_sensor = self.env.unwrapped.scene.sensors["ee_frame"]
                    if hasattr(ee_sensor, 'data') and hasattr(ee_sensor.data, 'target_pos_w') and hasattr(ee_sensor.data, 'target_quat_w'):
                        # Use target_pos_w and target_quat_w for current pose
                        current_pos_abs = ee_sensor.data.target_pos_w[0].cpu().numpy().squeeze()
                        current_rot_abs_wxyz = ee_sensor.data.target_quat_w[0].cpu().numpy().squeeze()
                        # print(f"Current_pos_abs({current_pos_abs}) and current_rot_abs_wxyz({current_rot_abs_wxyz})") # Debug print

                        # Calculate delta position
                        delta_pos = target_pos_abs_3D - current_pos_abs

                        # Calculate delta rotation using quaternion inversion and multiplication
                        R_current = Rotation.from_quat([current_rot_abs_wxyz[1], current_rot_abs_wxyz[2], current_rot_abs_wxyz[3], current_rot_abs_wxyz[0]]) # xyzw
                        R_target = Rotation.from_quat([target_rot_abs_wxyz[1], target_rot_abs_wxyz[2], target_rot_abs_wxyz[3], target_rot_abs_wxyz[0]]) # xyzw

                        R_delta = R_target * R_current.inv()
                        delta_rot_axis_angle = R_delta.as_rotvec() # Returns 3D axis-angle

                        # Combine delta position and delta rotation into a 6D pose command
                        delta_pose_command_6D = np.concatenate([delta_pos, delta_rot_axis_angle])
                        return (delta_pose_command_6D, current_gripper_command_bool)
                    else:
                         omni.log.warn("[TrajectoryPlayer Playback] EE sensor data or required attributes not found for delta calculation.")
                         self.is_playing_back = False
                         return None
                else:
                    omni.log.warn("[TrajectoryPlayer Playback] 'ee_frame' sensor not found for delta calculation.")
                    self.is_playing_back = False
                    return None
            except Exception as e:
                omni.log.error(f"[TrajectoryPlayer Playback ERROR] Unexpected error getting current EE state or calculating delta: {e}")
                self.is_playing_back = False # Stop playback on error
                return None

    def save_waypoints(self, filepath=WAYPOINTS_JSON_PATH):
        """
        Saves the recorded waypoints to a JSON file.

        Args:
            filepath: The path to save the JSON file. Directories will be created
                      if they don't exist.
        """
        if not self.recorded_waypoints:
            print("No waypoints to save.")
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        waypoints_to_save = []
        for wp in self.recorded_waypoints:
            # Convert numpy arrays to lists for JSON serialization
            waypoints_to_save.append({
                "position": wp["position"].tolist(),
                "orientation_wxyz": wp["orientation_wxyz"].tolist(),
                "gripper": bool(wp["gripper"]) # Ensure boolean type
            })
        try:
            with open(filepath, 'w') as f:
                json.dump(waypoints_to_save, f, indent=4)
            print(f"Waypoints saved to {filepath}")
        except Exception as e:
            omni.log.error(f"[TrajectoryPlayer ERROR] Error saving waypoints to {filepath}: {e}")
            import traceback
            traceback.print_exc()

    def load_waypoints(self, filepath=WAYPOINTS_JSON_PATH):
        """
        Loads waypoints from a JSON file.

        Args:
            filepath: The path to the JSON file to load.
        """
        try:
            with open(filepath, 'r') as f:
                loaded_wps_list = json.load(f)
            self.recorded_waypoints = []
            for wp_dict in loaded_wps_list:
                # Convert lists back to numpy arrays
                self.recorded_waypoints.append({
                    "position": np.array(wp_dict["position"]),
                    "orientation_wxyz": np.array(wp_dict["orientation_wxyz"]),
                    "gripper": bool(wp_dict.get("gripper", False)) # Handle missing gripper key with default False
                })
            print(f"Waypoints loaded from {filepath}. {len(self.recorded_waypoints)} waypoints found.")
            if len(self.recorded_waypoints) > 1:
                print("Press 'L' to start playback with loaded waypoints.")
            elif len(self.recorded_waypoints) > 0:
                print("Loaded waypoints, but need at least 2 to form a trajectory.")
        except FileNotFoundError:
            print(f"Waypoint file {filepath} not found. No waypoints loaded.")
        except json.JSONDecodeError:
            omni.log.error(f"Error decoding JSON from {filepath}. File might be corrupted.")
        except Exception as e:
            omni.log.error(f"Error loading waypoints from {filepath}: {e}")
            import traceback
            traceback.print_exc()
