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
import torch # Import torch

# Default paths for saving/loading data
WAYPOINTS_JSON_PATH = os.path.join("logs", "teleoperation", "waypoints.json")
JOINT_TRACKING_LOG_PATH = os.path.join("logs", "teleoperation", "joint_tracking_log.json")

class TrajectoryPlayer:
    """
    Handles recording, saving, loading, and playing back end-effector trajectories for G1.

    A trajectory is a sequence of waypoints, where each waypoint includes the
    right palm link's absolute position, orientation (as a quaternion), and the
    target joint positions for the right hand.

    Playback interpolates linearly between positions and uses spherical linear
    interpolation (Slerp) for orientations. Hand joint positions are held constant
    between waypoints during playback.
    Also records joint angles during trajectory playback for analysis.
    """
    def __init__(self, env, device_for_torch, steps_per_segment=100,
                 g1_hand_joints_open: dict = {}, g1_hand_joints_closed: dict = {}):
        """
        Initializes the TrajectoryPlayer for G1.

        Args:
            env: The Isaac Lab environment instance. Used to access the robot's
                 current state (e.g., end-effector pose).
            device_for_torch: The torch device to use for tensor operations.
            steps_per_segment: The number of simulation steps to use for
                               interpolating between two consecutive waypoints
                               during playback.
            g1_hand_joints_open: Dictionary of joint names and positions for open hand.
            g1_hand_joints_closed: Dictionary of joint names and positions for closed hand.
        """
        self.env = env
        self.torch_device = device_for_torch
        # List of recorded waypoints. Each waypoint is a dict:
        # {"palm_position": np.array, "palm_orientation_wxyz": np.array, "hand_joint_positions": np.array}
        self.recorded_waypoints = []
        # List of interpolated poses for playback. Each element is a dict:
        # {"palm_position": np.array, "palm_orientation_wxyz": np.array, "hand_joint_positions": np.array}
        self.playback_trajectory_abs_poses = []
        self.current_playback_idx = 0
        self.is_playing_back = False
        self.steps_per_segment = steps_per_segment

        # G1 Hand joint configurations
        self.g1_hand_joints_open = g1_hand_joints_open
        self.g1_hand_joints_closed = g1_hand_joints_closed
        # List of right hand joint names for G1.
        # Based on the provided link list and common hand structures, these are
        # the likely controllable joints for the right hand.
        # IMPORTANT: The order of these joint names must match the order expected
        # by the G1 environment's action space for the hand joints.
        self.right_hand_joint_names = [
            "right_five_joint", "right_three_joint", "right_six_joint",
            "right_four_joint", "right_zero_joint", "right_one_joint",
            "right_two_joint",
        ]

        # Joint tracking data
        self.joint_tracking_records = []
        self.sim_time = 0.0
        self.joint_tracking_active = False

    def record_current_pose(self, teleop_output=None):
        """
        Record the current right palm link pose and right hand joint positions as a waypoint.

        If teleop_output is provided, extracts the boolean gripper command from it
        to determine the target hand joint positions (open or closed).
        For G1, the gripper command (True/False) is used to select between
        pre-defined open or closed hand joint configurations.

        Args:
            teleop_output: Optional. The raw output from the teleoperation device.
                           Expected to be a tuple where the second element is
                           the boolean gripper command (True for close, False for open).
        """
        if self.is_playing_back:
            print("Cannot record waypoints while playback is active. Stop playback first.")
            return

        # Extract gripper command from teleop_output if available
        # For keyboard, this comes from the gripper button (e.g., 'G')
        gripper_command_bool = False
        if teleop_output is not None and isinstance(teleop_output, (tuple, list)) and len(teleop_output) > 1:
            gripper_command_bool = teleop_output[1]

        try:
            # Access the robot articulation
            if hasattr(self.env.unwrapped, 'scene') and hasattr(self.env.unwrapped.scene, 'articulations') and "robot" in self.env.unwrapped.scene.articulations:
                robot = self.env.unwrapped.scene.articulations["robot"]

                # Get the right palm link pose
                # Assuming body_states_w is available and indexed by body index
                if hasattr(robot.data, 'body_states_w'):
                    # Find the body index for the right_palm_link
                    right_palm_idx = robot.find_bodies(["right_palm_link"])[0]
                    palm_pos_tensor = robot.data.body_states_w[0, right_palm_idx, :3] # Assuming batch size 1
                    palm_orient_tensor_wxyz = robot.data.body_states_w[0, right_palm_idx, 3:7] # Assuming batch size 1

                    # Determine target hand joint positions based on gripper command
                    target_hand_joint_positions = np.zeros(len(self.right_hand_joint_names))
                    if gripper_command_bool: # Gripper command is True (close)
                        if self.g1_hand_joints_closed:
                            # Use closed positions, ensuring correct order
                            target_hand_joint_positions = np.array([self.g1_hand_joints_closed.get(name, 0.0) for name in self.right_hand_joint_names])
                        else:
                             # Fallback to current positions if closed config not provided
                             if hasattr(robot.data, 'joint_pos'):
                                right_hand_joint_indices = robot.find_joints(self.right_hand_joint_names)
                                target_hand_joint_positions = robot.data.joint_pos[0, right_hand_joint_indices].cpu().numpy().squeeze()
                                omni.log.warn("[TrajectoryPlayer] G1_HAND_JOINTS_CLOSED not provided. Recording current hand joint positions for 'close' command.")
                             else:
                                 omni.log.warn("[TrajectoryPlayer] Robot articulation data 'joint_pos' not found. Cannot record current hand joint positions.")
                                 # Keep target_hand_joint_positions as zeros if current pos not available
                    else: # Gripper command is False (open)
                        if self.g1_hand_joints_open:
                             # Use open positions, ensuring correct order
                             target_hand_joint_positions = np.array([self.g1_hand_joints_open.get(name, 0.0) for name in self.right_hand_joint_names])
                        else:
                             # Fallback to current positions if open config not provided
                             if hasattr(robot.data, 'joint_pos'):
                                right_hand_joint_indices = robot.find_joints(self.right_hand_joint_names)
                                target_hand_joint_positions = robot.data.joint_pos[0, right_hand_joint_indices].cpu().numpy().squeeze()
                                omni.log.warn("[TrajectoryPlayer] G1_HAND_JOINTS_OPEN not provided. Recording current hand joint positions for 'open' command.")
                             else:
                                 omni.log.warn("[TrajectoryPlayer] Robot articulation data 'joint_pos' not found. Cannot record current hand joint positions.")
                                 # Keep target_hand_joint_positions as zeros if current pos not available


                    # Convert tensors to numpy arrays and squeeze
                    palm_pos = palm_pos_tensor.cpu().numpy().squeeze()
                    palm_orient_wxyz = palm_orient_tensor_wxyz.cpu().numpy().squeeze()

                    # Store the waypoint
                    self.recorded_waypoints.append({
                        "palm_position": palm_pos,
                        "palm_orientation_wxyz": palm_orient_wxyz,
                        "hand_joint_positions": target_hand_joint_positions
                    })
                    print(f"Waypoint {len(self.recorded_waypoints)} recorded: palm_pos={palm_pos}, palm_orient_wxyz={palm_orient_wxyz}, hand_joint_positions={target_hand_joint_positions}")
                    return
                else:
                     omni.log.warn("[TrajectoryPlayer] Robot articulation data 'body_states_w' not found.")
            else:
                omni.log.warn("[TrajectoryPlayer] Robot articulation not found in scene.")

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

    def record_joint_state(self, reference_joints, current_joints):
        """
        Records joint states during trajectory playback.

        Args:
            reference_joints: List or array of reference/target joint angles
            current_joints: List or array of current joint angles
        """
        if not self.joint_tracking_active or not self.is_playing_back:
            return

        entry = {
            "timestamp": float(self.sim_time),
            "reference_joint_angles": [float(x) for x in reference_joints],
            "current_joint_angles": [float(x) for x in current_joints]
        }

        # Add joint names if available
        try:
            if hasattr(self.env.unwrapped, 'scene') and hasattr(self.env.unwrapped.scene, 'articulations') and "robot" in self.env.unwrapped.scene.articulations:
                robot_art = self.env.unwrapped.scene.articulations["robot"]
                if hasattr(robot_art.data, 'joint_names'):
                    entry["joint_names"] = list(robot_art.data.joint_names)
        except Exception as e:
             omni.log.warn(f"[TrajectoryPlayer Tracking] Could not get joint names for tracking: {e}")


        self.joint_tracking_records.append(entry)

    def clear_joint_tracking_data(self):
        """
        Clears recorded joint tracking data and resets timing.
        """
        self.joint_tracking_records = []
        self.sim_time = 0.0
        self.joint_tracking_active = False

    def save_joint_tracking_data(self):
        """
        Saves the recorded joint tracking data to a JSON file.
        """
        if not self.joint_tracking_records:
            print("No joint tracking data to save.")
            return

        os.makedirs(os.path.dirname(JOINT_TRACKING_LOG_PATH), exist_ok=True)
        try:
            with open(JOINT_TRACKING_LOG_PATH, 'w') as f:
                json.dump(self.joint_tracking_records, f, indent=2)
            print(f"Joint tracking data saved to {JOINT_TRACKING_LOG_PATH}")
        except Exception as e:
            omni.log.error(f"[TrajectoryPlayer ERROR] Error saving joint tracking data: {e}")
            import traceback
            traceback.print_exc()

    def prepare_playback_trajectory(self):
        """
        Generates the interpolated trajectory steps from the recorded waypoints.

        Uses linear interpolation for palm position and Slerp for palm orientation.
        Hand joint positions are held constant between waypoints.
        """
        if len(self.recorded_waypoints) < 2:
            print("Not enough waypoints (need at least 2). Playback not started.")
            self.is_playing_back = False
            return

        self.playback_trajectory_abs_poses = []
        palm_positions = np.array([wp["palm_position"] for wp in self.recorded_waypoints])

        # Convert wxyz quaternions to xyzw format for SciPy Rotation
        palm_orientations_xyzw = []
        for wp in self.recorded_waypoints:
            wxyz = wp["palm_orientation_wxyz"]
            palm_orientations_xyzw.append([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

        scipy_rotations = Rotation.from_quat(palm_orientations_xyzw)

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
            interp_pos = palm_positions[i, None] * (1 - segment_times[:, None]) + palm_positions[i+1, None] * segment_times[:, None]

            # Hand joint positions (held constant for the segment from the start waypoint)
            interp_hand_joints = self.recorded_waypoints[i]["hand_joint_positions"]

            # Convert interpolated SciPy rotations back to wxyz numpy arrays
            interp_orient_xyzw = interp_rot_scipy.as_quat()
            interp_orient_wxyz = np.stack([interp_orient_xyzw[:, 3], interp_orient_xyzw[:, 0], interp_orient_xyzw[:, 1], interp_orient_xyzw[:, 2]], axis=1)

            # Append interpolated poses to the playback trajectory
            for j in range(len(segment_times)):
                 self.playback_trajectory_abs_poses.append({
                    "palm_position": interp_pos[j],
                    "palm_orientation_wxyz": interp_orient_wxyz[j],
                    "hand_joint_positions": interp_hand_joints
                })

        self.current_playback_idx = 0
        self.is_playing_back = True
        self.clear_joint_tracking_data()  # Clear any previous tracking data
        self.joint_tracking_active = True  # Start joint tracking for this playback
        print(f"Playback trajectory prepared with {len(self.playback_trajectory_abs_poses)} steps.")

    def get_formatted_action_for_playback(self, task_name: str):
        """
        Gets the next action command from the playback trajectory for G1.

        Args:
            task_name: The name of the current task environment.

        Returns:
            A numpy array representing the action command for the environment,
            or None if playback is finished.
            Action format: [right palm pos (3), right palm quat (4), right hand joint pos (7)]
        """
        if not self.is_playing_back or self.current_playback_idx >= len(self.playback_trajectory_abs_poses):
            if self.joint_tracking_active:
                self.save_joint_tracking_data()  # Save data when playback ends
                self.joint_tracking_active = False
            self.is_playing_back = False
            if self.current_playback_idx > 0 and len(self.playback_trajectory_abs_poses) > 0 : # Avoid print if never started
                print("Playback finished.")
            return None

        # Get the target pose and hand joint positions for the current step
        target_abs_pose = self.playback_trajectory_abs_poses[self.current_playback_idx]
        self.current_playback_idx += 1
        target_palm_pos_3D = target_abs_pose["palm_position"]
        target_palm_orient_wxyz = target_abs_pose["palm_orientation_wxyz"]
        target_hand_joint_positions = target_abs_pose["hand_joint_positions"]

        # Record joint state for tracking (optional, can be removed if not needed)
        try:
            if hasattr(self.env.unwrapped, 'scene') and hasattr(self.env.unwrapped.scene, 'articulations'):
                if "robot" in self.env.unwrapped.scene.articulations:
                    robot_art = self.env.unwrapped.scene.articulations["robot"]
                    # Get current joint positions from observation
                    if hasattr(robot_art.data, 'joint_pos'):
                         current_joints = robot_art.data.joint_pos[0].cpu().numpy().tolist()
                         # For reference joints, use the target/commanded joints from actions
                         # Note: This might not be accurate if the environment's action mapping is complex.
                         # A more robust approach might involve storing the commanded joint targets during recording.
                         reference_joints = getattr(robot_art.data, 'joint_pos_target', torch.zeros_like(robot_art.data.joint_pos))[0].cpu().numpy().tolist()
                         # Get accurate simulation time from robot data
                         self.sim_time = float(getattr(robot_art.data, '_sim_timestamp', self.sim_time))

                         # Record the joint state
                         self.record_joint_state(reference_joints, current_joints)
                    else:
                         omni.log.warn("[TrajectoryPlayer Playback] Robot articulation data 'joint_pos' not found for tracking.")

        except Exception as e:
            omni.log.error(f"[TrajectoryPlayer Playback ERROR] Error recording joint state: {e}")
            import traceback
            traceback.print_exc()

        # Convert target palm orientation from WXYZ to XYZW for the action
        target_palm_orient_xyzw = np.array([
            target_palm_orient_wxyz[1],
            target_palm_orient_wxyz[2],
            target_palm_orient_wxyz[3],
            target_palm_orient_wxyz[0],
        ])

        # Combine into the 13-element action tensor
        action = np.concatenate([
            target_palm_pos_3D,
            target_palm_orient_xyzw,
            target_hand_joint_positions
        ])

        # Return as a tuple containing the single action array
        # The pre_process_actions function in teleop_se3_agent_g1.py expects a tuple
        # for playback data, even if it's just one element.
        return (action,)


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
                "palm_position": wp["palm_position"].tolist(),
                "palm_orientation_wxyz": wp["palm_orientation_wxyz"].tolist(),
                "hand_joint_positions": wp["hand_joint_positions"].tolist()
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
                    "palm_position": np.array(wp_dict["palm_position"]),
                    "palm_orientation_wxyz": np.array(wp_dict["palm_orientation_wxyz"]),
                    "hand_joint_positions": np.array(wp_dict.get("hand_joint_positions", np.zeros(len(self.right_hand_joint_names)))) # Handle missing key with default
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
