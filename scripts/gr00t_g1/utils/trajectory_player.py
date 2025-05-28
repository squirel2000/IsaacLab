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
from isaaclab_tasks.manager_based.manipulation.pick_place_g1.mdp.observations import get_right_eef_pos, get_right_eef_quat, get_left_eef_pos, get_left_eef_quat

# GraspPoseCalculator is now a direct runtime dependency
from .grasp_pose_calculator import GraspPoseCalculator


# === Quaternion conversion utilities === 
def quat_xyzw_to_wxyz(q):
    """Convert quaternion from [x, y, z, w] to [w, x, y, z] order."""
    q = np.asarray(q)
    if q.shape[-1] != 4:
        raise ValueError("Quaternion must have 4 elements.")
    return np.array([q[3], q[0], q[1], q[2]]) if q.ndim == 1 else np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)

def quat_wxyz_to_xyzw(q):
    """Convert quaternion from [w, x, y, z] to [x, y, z, w] order."""
    q = np.asarray(q)
    if q.shape[-1] != 4:
        raise ValueError("Quaternion must have 4 elements.")
    return np.array([q[1], q[2], q[3], q[0]]) if q.ndim == 1 else np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)


# === Constants for G1 Trajectory Generation ===
# Default left arm pose
DEFAULT_LEFT_ARM_POS_W = np.array([-0.14866172, 0.1997742, 0.9152355])
DEFAULT_LEFT_ARM_QUAT_WXYZ_W = np.array([0.7071744, 0.0000018, 0.00004074, 0.70703906])  # wxyz
DEFAULT_LEFT_HAND_BOOL = False  # False for open

# Constants for RED_PLATE pose
RED_PLATE_XY_POS = np.array([0.100, 0.200])
# Calculate 80-degree yaw quaternion (wxyz) for RED_PLATE
yaw_degrees = 80.0
RED_PLATE_QUAT_WXYZ = quat_xyzw_to_wxyz(Rotation.from_euler('z', yaw_degrees, degrees=True).as_quat())

# Default paths for saving waypoints and joint tracking logs
WAYPOINTS_JSON_PATH = os.path.join("logs", "teleoperation", "waypoints.json")
JOINT_TRACKING_LOG_PATH = os.path.join("logs", "teleoperation", "joint_tracking_log.json")


class TrajectoryPlayer:
    """
    Handles recording, saving, loading, and playing back end-effector trajectories for G1.

    A trajectory is a sequence of waypoints, where each waypoint includes the
    right palm link's absolute position, orientation (as a quaternion), and the
    target joint positions for the right hand.
    """
    def __init__(self, env, steps_per_segment=100):
        """
        Initializes the TrajectoryPlayer for G1.

        Args:
            env: The Isaac Lab environment instance. Used to access the robot's
                 current state (e.g., end-effector pose).
            device_for_torch: The torch device to use for tensor operations.
            steps_per_segment: The number of simulation steps to use for
                               interpolating between two consecutive waypoints
        """
        self.env = env
        # {"left_arm_eef"(7), "right_arm_eef"(7), "left_hand", "right_hand"}
        self.recorded_waypoints = []
        # {"palm_position": np.array, "palm_orientation_wxyz": np.array, "hand_joint_positions": np.array}
        self.playback_trajectory_actions = []
        self.current_playback_idx = 0
        self.is_playing_back = False
        self.grasp_calculator = GraspPoseCalculator() # Instantiate GraspPoseCalculator
        self.steps_per_segment = steps_per_segment

        # Get hand joint names from the action manager
        self.pink_hand_joint_names = self.env.action_manager._terms["pink_ik_cfg"].cfg.hand_joint_names
        # print("[TrajectoryPlayer] Initialized with hand joint names:", self.pink_hand_joint_names)
        # ['left_hand_index_0_joint', 'left_hand_middle_0_joint', 'left_hand_thumb_0_joint', 'right_hand_index_0_joint', 'right_hand_middle_0_joint', 'right_hand_thumb_0_joint', 'left_hand_index_1_joint', 'left_hand_middle_1_joint', 'left_hand_thumb_1_joint', 'right_hand_index_1_joint', 'right_hand_middle_1_joint', 'right_hand_thumb_1_joint', 'left_hand_thumb_2_joint', 'right_hand_thumb_2_joint']
        
        # Joint tracking data
        self.joint_tracking_records = []
        self.joint_tracking_active = False

    def record_current_pose(self, teleop_output=None):
        """
        Record the current end-effector link pose and orientation for both right and left, and gripper bools.
        Concatenate [right_arm_eef_pos, right_arm_eef_orient_wxyz, right_hand_bool, left_arm_eef_pos, left_arm_eef_orient_wxyz, left_hand_bool].
        """
        # Get the end-effector link pose and orientation using observation helpers
        left_arm_eef_pos = get_left_eef_pos(self.env).cpu().numpy().squeeze()
        left_arm_eef_orient_wxyz = get_left_eef_quat(self.env).cpu().numpy().squeeze()
        
        right_arm_eef_pos = get_right_eef_pos(self.env).cpu().numpy().squeeze()
        right_arm_eef_orient_wxyz = get_right_eef_quat(self.env).cpu().numpy().squeeze()

        # Extract right gripper command from teleop_output
        left_gripper_bool = 1  # Always 1 for left gripper
        right_gripper_bool = 0
        if teleop_output is not None and isinstance(teleop_output, (tuple, list)) and len(teleop_output) > 1:
            right_gripper_bool = int(teleop_output[1])

        # Store as structured dict per user request
        waypoint = {
            "left_arm_eef": np.concatenate([left_arm_eef_pos.flatten(), left_arm_eef_orient_wxyz.flatten()]),
            "right_arm_eef": np.concatenate([right_arm_eef_pos.flatten(), right_arm_eef_orient_wxyz.flatten()]),
            "left_hand_bool": int(left_gripper_bool),
            "right_hand_bool": int(right_gripper_bool)
        }

        self.recorded_waypoints.append(waypoint)
        print(f"Waypoint {len(self.recorded_waypoints)} recorded: {waypoint}")
        return


    def clear_waypoints(self):
        """
        Clears all recorded waypoints and stops playback if active.
        """
        self.recorded_waypoints = []
        self.playback_trajectory_actions = []
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


    def record_joint_state(self, sim_time, reference_joints, current_joints):
        """
        Records joint states during trajectory playback.

        Args:
            reference_joints: List or array of reference/target joint angles
            current_joints: List or array of current joint angles
        """
        if not self.joint_tracking_active or not self.is_playing_back:
            return

        entry = {
            "timestamp": float(sim_time),
            "reference_joints": [float(x) for x in reference_joints],
            "current_joints": [float(x) for x in current_joints]
        }

        self.joint_tracking_records.append(entry)


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
                # Convert all entries to JSON strings with indentation and join with commas
                json_lines = ',\n'.join(f'  {json.dumps(entry)}' for entry in self.joint_tracking_records)
                f.write('[\n' + json_lines + '\n]\n')
            print(f"Joint tracking data saved to {JOINT_TRACKING_LOG_PATH}")
        except Exception as e:
            omni.log.error(f"[TrajectoryPlayer ERROR] Error saving joint tracking data: {e}")
            import traceback
            traceback.print_exc()

    def clear_joint_tracking_data(self):
        """
        Clears recorded joint tracking data and resets timing.
        """
        self.joint_tracking_records = []
        self.joint_tracking_active = False
        
    def prepare_playback_trajectory(self):
        """
        Generates the interpolated trajectory steps from the recorded waypoints.

        Uses linear interpolation for end-effector position and Slerp for end-effector orientation.
        Uses linear interpolation for hand joints.
        """
        if len(self.recorded_waypoints) < 2:
            print("Not enough waypoints (need at least 2). Playback not started.")
            self.is_playing_back = False
            return

        self.playback_trajectory_actions = []
        left_arm_eef_pos = np.array([wp["left_arm_eef"][:3] for wp in self.recorded_waypoints])
        left_arm_eef_orient_wxyz = np.array([wp["left_arm_eef"][3:7] for wp in self.recorded_waypoints])
        left_hand_bools = [wp["left_hand_bool"] for wp in self.recorded_waypoints]
        right_arm_eef_pos = np.array([wp["right_arm_eef"][:3] for wp in self.recorded_waypoints])
        right_arm_eef_orient_wxyz = np.array([wp["right_arm_eef"][3:7] for wp in self.recorded_waypoints])
        right_hand_bools = [wp["right_hand_bool"] for wp in self.recorded_waypoints]

        # Convert wxyz quaternions to xyzw format for SciPy Rotation
        left_orient_xyzw = quat_wxyz_to_xyzw(left_arm_eef_orient_wxyz)
        left_rotations = Rotation.from_quat(left_orient_xyzw)
        
        right_orient_xyzw = quat_wxyz_to_xyzw(right_arm_eef_orient_wxyz)
        right_rotations = Rotation.from_quat(right_orient_xyzw)

        # Interpolate each segment
        num_segments = len(self.recorded_waypoints) - 1
        for i in range(num_segments):
            # Create time points for interpolation within the segment
            num_points_in_segment = self.steps_per_segment
            # Exclude the last point for all but the final segment to avoid duplicates
            segment_times = np.linspace(0, 1, num_points_in_segment, endpoint=(i == num_segments - 1))

            # Interpolate right arm end-effector (Slerp for orientation and linear for position)
            right_key_rots = Rotation.concatenate([right_rotations[i], right_rotations[i+1]])
            right_slerp = Slerp([0, 1], right_key_rots)
            interp_right_orient_xyzw = right_slerp(segment_times).as_quat() # xyzw format in SciPy
            interp_right_orient_wxyz = quat_xyzw_to_wxyz(interp_right_orient_xyzw)
            interp_right_pos = right_arm_eef_pos[i, None] * (1 - segment_times[:, None]) + right_arm_eef_pos[i+1, None] * segment_times[:, None]
            
            # Interpolate left arm end-effector
            left_key_rots = Rotation.concatenate([left_rotations[i], left_rotations[i+1]])
            left_slerp = Slerp([0, 1], left_key_rots)
            interp_left_orient_xyzw = left_slerp(segment_times).as_quat()
            interp_left_orient_wxyz = quat_xyzw_to_wxyz(interp_left_orient_xyzw)
            interp_left_pos = left_arm_eef_pos[i, None] * (1 - segment_times[:, None]) + left_arm_eef_pos[i+1, None] * segment_times[:, None]
            
            
            # Interpolate hand joint states base on the order of the pink_hand_joint_names
            hand_joint_positions = np.zeros(len(self.pink_hand_joint_names))
            next_hand_joint_positions = np.zeros(len(self.pink_hand_joint_names))

            # Set initial positions using create_hand_joint_positions
            hand_joint_positions = self.create_hand_joint_positions(
                left_hand_bool=left_hand_bools[i],
                right_hand_bool=right_hand_bools[i]
            )
            next_hand_joint_positions = self.create_hand_joint_positions(
                left_hand_bool=left_hand_bools[i+1],
                right_hand_bool=right_hand_bools[i+1]
            )

            # Store the interpolated 28D data for this segment [left_arm_eef(7), right_arm_eef(7), hand_joints(14)]
            for j in range(len(segment_times)):
                interp_hand_positions = hand_joint_positions * (1 - segment_times[j]) + next_hand_joint_positions * segment_times[j]

                action_array = np.concatenate([
                    np.concatenate([interp_left_pos[j], interp_left_orient_wxyz[j]]),   # left_arm_eef (7)
                    np.concatenate([interp_right_pos[j], interp_right_orient_wxyz[j]]), # right_arm_eef (7)
                    interp_hand_positions  # hand_joints (14)
                ])
                self.playback_trajectory_actions.append(action_array)

        self.current_playback_idx = 0
        self.is_playing_back = True
        self.clear_joint_tracking_data()  # Clear any previous tracking data
        self.joint_tracking_active = True  # Start joint tracking for this playback
        print(f"Playback trajectory prepared with {len(self.playback_trajectory_actions)} steps.")


    def get_formatted_action_for_playback(self):
        """
        Gets the next action command from the playback trajectory for G1.

        Returns:
            [left_arm_eef(7), right_arm_eef(7), hand_joints(14)]
        """
        if not self.is_playing_back or self.current_playback_idx >= len(self.playback_trajectory_actions):
            if self.joint_tracking_active:
                self.save_joint_tracking_data()  # Save data when playback ends
                self.joint_tracking_active = False
            self.is_playing_back = False
            if self.current_playback_idx > 0 and len(self.playback_trajectory_actions) > 0:
                print("Playback finished.")
            return None

        # Get the action array for the current step
        action_array = self.playback_trajectory_actions[self.current_playback_idx]
        self.current_playback_idx += 1
        # print(f"Playback step {self.current_playback_idx}/{len(self.playback_trajectory_actions)}: {action_array}")

        # Get current simulation time, reference/current joint positions from observation
        try:
            robot_art = self.env.unwrapped.scene.articulations["robot"]
            sim_time = float(robot_art.data._sim_timestamp)
            reference_joints = robot_art.data.joint_pos_target[0].cpu().numpy().tolist()
            current_joints = robot_art.data.joint_pos[0].cpu().numpy().tolist()
            
            self.record_joint_state(sim_time, reference_joints, current_joints)
                    
        except Exception as e:
            omni.log.error(f"[TrajectoryPlayer ERROR] Error recording joint state: {e}")
            import traceback
            traceback.print_exc()

        return (action_array,)


    def save_waypoints(self, filepath=WAYPOINTS_JSON_PATH):
        """
        Saves the recorded waypoints to a JSON file.

        Args:
            filepath: The path to save the JSON file. Directories will be created if they don't exist.
        """
        if not self.recorded_waypoints:
            print("No waypoints to save.")
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        waypoints_to_save = []
        for wp in self.recorded_waypoints:
            # Convert numpy arrays to lists for JSON serialization
            waypoints_to_save.append({
                "left_arm_eef": wp["left_arm_eef"].tolist(),
                "right_arm_eef": wp["right_arm_eef"].tolist(),
                "left_hand_bool": int(wp["left_hand_bool"]),
                "right_hand_bool": int(wp["right_hand_bool"])
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
        """ Loads waypoints from a JSON file. """
        try:
            with open(filepath, 'r') as f:
                loaded_wps_list = json.load(f)
        except FileNotFoundError:
            print(f"Waypoint file {filepath} not found. No waypoints loaded.")
            return
        except json.JSONDecodeError:
            omni.log.error(f"Error decoding JSON from {filepath}. File might be corrupted.")
            return
        except Exception as e:
            omni.log.error(f"Error loading waypoints from {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return
        
        self.recorded_waypoints = []
        for wp_dict in loaded_wps_list:
            # Convert lists back to numpy arrays, load in the new order
            self.recorded_waypoints.append({
                "left_arm_eef": np.array(wp_dict["left_arm_eef"]),
                "right_arm_eef": np.array(wp_dict["right_arm_eef"]),
                "left_hand_bool": int(wp_dict["left_hand_bool"]),
                "right_hand_bool": int(wp_dict["right_hand_bool"])
            })
        print(f"Waypoints loaded from {filepath}. {len(self.recorded_waypoints)} waypoints found.")

    def create_hand_joint_positions(self, left_hand_bool: bool, right_hand_bool: bool) -> np.ndarray:
        """Creates a hand joint positions array following the order of pink_hand_joint_names.
        
        Args:
            left_hand_bool: Boolean indicating if left hand should be closed (True) or open (False)
            right_hand_bool: Boolean indicating if right hand should be closed (True) or open (False)
            
        Returns:
            numpy.ndarray: Array of joint positions based on the open/closed states of the hands.
        """
        # Define joint positions for open and closed states
        joint_positions = {
            "left_hand_index_0_joint":   {"open": 0.0, "closed": 0.8},
            "left_hand_middle_0_joint":  {"open": 0.0, "closed": 0.8},
            "left_hand_thumb_0_joint":   {"open": 0.0, "closed": 0.0},
            "right_hand_index_0_joint":  {"open": 0.0, "closed": 0.8},
            "right_hand_middle_0_joint": {"open": 0.0, "closed": 0.8},
            "right_hand_thumb_0_joint":  {"open": 0.0, "closed": 0.0},
            "left_hand_index_1_joint":   {"open": 0.0, "closed": 0.8},
            "left_hand_middle_1_joint":  {"open": 0.0, "closed": 0.8},
            "left_hand_thumb_1_joint":   {"open": 0.0, "closed": 0.8},
            "right_hand_index_1_joint":  {"open": 0.0, "closed": 0.8},
            "right_hand_middle_1_joint": {"open": 0.0, "closed": 0.8},
            "right_hand_thumb_1_joint":  {"open": 0.0, "closed": -0.8},
            "left_hand_thumb_2_joint":   {"open": 0.0, "closed": 0.8},
            "right_hand_thumb_2_joint":  {"open": 0.0, "closed": -0.8},
        }

        hand_joint_positions = np.zeros(len(self.pink_hand_joint_names))
        for idx, joint_name in enumerate(self.pink_hand_joint_names):
            if "right" in joint_name:
                hand_joint_positions[idx] = joint_positions[joint_name]["closed"] if right_hand_bool else joint_positions[joint_name]["open"]
            elif "left" in joint_name:
                hand_joint_positions[idx] = joint_positions[joint_name]["closed"] if left_hand_bool else joint_positions[joint_name]["open"]
        return hand_joint_positions

    def _extract_poses_from_obs(self, obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extracts relevant poses from the observation dictionary."""
        current_right_eef_pos_w = obs["policy"]["right_eef_pos"][0].cpu().numpy()
        current_right_eef_quat_wxyz_w = obs["policy"]["right_eef_quat"][0].cpu().numpy()
        cube_pos_w = obs["policy"]["cube_pos"][0].cpu().numpy()
        cube_quat_wxyz_w = obs["policy"]["cube_rot"][0].cpu().numpy()
        # cube_color_rgb = obs["policy"]["cube_color"][0].cpu().numpy() # If needed in the future

        print(f"Current Right EEF Pose: pos={current_right_eef_pos_w}, quat_wxyz={current_right_eef_quat_wxyz_w}")
        print(f"Target Cube Pose: pos={cube_pos_w}, quat_wxyz={cube_quat_wxyz_w}")
        # print(f"Target Cube Pose: pos={cube_pos_w}, quat_wxyz={cube_quat_wxyz_w}, color={cube_color_rgb}")
        return current_right_eef_pos_w, current_right_eef_quat_wxyz_w, cube_pos_w, cube_quat_wxyz_w

    def generate_auto_grasp_pick_place_trajectory(self, obs: dict):
        """
        Generates a predefined 7-waypoint trajectory for grasping a cube and placing it.
        The waypoints are stored in self.recorded_waypoints.

        Args:
            obs: The observation dictionary from the environment, containing current robot and object states.
        """
        current_right_eef_pos_w, current_right_eef_quat_wxyz_w, cube_pos_w, cube_quat_wxyz_w = self._extract_poses_from_obs(obs)
        self.clear_waypoints()
        # 1. Calculate target grasp pose for the right EEF
        target_grasp_right_eef_pos_w, target_grasp_right_eef_quat_wxyz_w = \
            self.grasp_calculator.calculate_target_ee_pose(cube_pos_w, cube_quat_wxyz_w)
        print(f"Calculated Target Grasp Right EEF Pose: pos={target_grasp_right_eef_pos_w}, quat_wxyz={target_grasp_right_eef_quat_wxyz_w}")

        # Waypoint 1: Current EEF pose (right hand open)
        wp1_left_arm_eef = np.concatenate([DEFAULT_LEFT_ARM_POS_W, DEFAULT_LEFT_ARM_QUAT_WXYZ_W])
        wp1_right_arm_eef = np.concatenate([current_right_eef_pos_w, current_right_eef_quat_wxyz_w])
        waypoint1 = {
            "left_arm_eef": wp1_left_arm_eef,
            "right_arm_eef": wp1_right_arm_eef,
            "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL),
            "right_hand_bool": 0
        }
        self.recorded_waypoints.append(waypoint1)

        # Waypoint 2: Target grasp EEF pose (right hand open - pre-grasp)
        wp2_left_arm_eef = np.concatenate([DEFAULT_LEFT_ARM_POS_W, DEFAULT_LEFT_ARM_QUAT_WXYZ_W])
        wp2_right_arm_eef = np.concatenate([target_grasp_right_eef_pos_w, target_grasp_right_eef_quat_wxyz_w])
        waypoint2 = {
            "left_arm_eef": wp2_left_arm_eef,
            "right_arm_eef": wp2_right_arm_eef,
            "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL),
            "right_hand_bool": 0
        }
        self.recorded_waypoints.append(waypoint2)

        # Waypoint 3: Target grasp EEF pose (right hand closed - grasp)
        waypoint3 = {**waypoint2, "right_hand_bool": 1}
        self.recorded_waypoints.append(waypoint3)

        # Define RED_PLATE pose based on dynamic Z from grasp
        red_plate_target_pos_w = np.array([RED_PLATE_XY_POS[0], RED_PLATE_XY_POS[1], target_grasp_right_eef_pos_w[2]])

        # Waypoint 4: Intermediate lift pose
        lift_pos_x = (target_grasp_right_eef_pos_w[0] + red_plate_target_pos_w[0]) / 2
        lift_pos_y = (target_grasp_right_eef_pos_w[1] + red_plate_target_pos_w[1]) / 2
        lift_pos_z = (target_grasp_right_eef_pos_w[2] + red_plate_target_pos_w[2]) / 2 + 0.05
        lift_pos_w = np.array([lift_pos_x, lift_pos_y, lift_pos_z])

        quat_grasp_xyzw = target_grasp_right_eef_quat_wxyz_w[[1, 2, 3, 0]]
        quat_red_plate_xyzw = RED_PLATE_QUAT_WXYZ[[1, 2, 3, 0]]
        key_rots = Rotation.from_quat([quat_grasp_xyzw, quat_red_plate_xyzw])
        slerp_interpolator = Slerp([0, 1], key_rots)
        lift_quat_xyzw = slerp_interpolator(0.5).as_quat()
        lift_quat_wxyz_w = lift_quat_xyzw[[3, 0, 1, 2]]

        waypoint4 = {
            "left_arm_eef": wp2_left_arm_eef,
            "right_arm_eef": np.concatenate([lift_pos_w, lift_quat_wxyz_w]),
            "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL),
            "right_hand_bool": 1
        }
        self.recorded_waypoints.append(waypoint4)

        # Waypoint 5: Move right arm EEF to RED_PLATE pose (hand closed)
        waypoint5 = {
            "left_arm_eef": wp2_left_arm_eef,
            "right_arm_eef": np.concatenate([red_plate_target_pos_w, RED_PLATE_QUAT_WXYZ]),
            "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL),
            "right_hand_bool": 1
        }
        self.recorded_waypoints.append(waypoint5)

        # Waypoint 6: At RED_PLATE pose, open right hand
        waypoint6 = {**waypoint5, "right_hand_bool": 0}
        self.recorded_waypoints.append(waypoint6)

        # Waypoint 7: Move the right arm to the initial pose (hand open)
        waypoint7 = {**waypoint1, "right_hand_bool": 0} # Uses wp1's arm poses, ensures hand is open
        self.recorded_waypoints.append(waypoint7)

        omni.log.info(f"Generated {len(self.recorded_waypoints)} waypoints for auto grasp and place.")
