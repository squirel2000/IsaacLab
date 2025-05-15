# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation terms for G1 pick-and-place."""

import torch
import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedEnv

def get_left_eef_pos(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the left end-effector position in world frame."""
    # Access articulation data from scene
    robot = env.scene.articulations["robot"]
    # G1's left hand link is the EEF
    left_hand_idx = robot.find_bodies(["left_hand_link"])[0]
    return robot.data.body_states_w[left_hand_idx, :3]

def get_left_eef_quat(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the left end-effector orientation in world frame."""
    robot = env.scene.articulations["robot"]
    left_hand_idx = robot.find_bodies(["left_hand_link"])[0]
    return robot.data.body_states_w[left_hand_idx, 3:7]

def get_right_eef_pos(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the right end-effector position in world frame."""
    robot = env.scene.articulations["robot"]
    right_hand_idx = robot.find_bodies(["right_hand_link"])[0]
    return robot.data.body_states_w[right_hand_idx, :3]

def get_right_eef_quat(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the right end-effector orientation in world frame."""
    robot = env.scene.articulations["robot"]
    right_hand_idx = robot.find_bodies(["right_hand_link"])[0]
    return robot.data.body_states_w[right_hand_idx, 3:7]

def get_hand_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the hand joint states (positions and velocities)."""
    robot = env.scene.articulations["robot"]
    # Get indices for all hand joints
    hand_joint_indices = robot.find_joints([
        ".*_zero_joint", ".*_one_joint", ".*_two_joint",
        ".*_three_joint", ".*_four_joint", ".*_five_joint",
        ".*_six_joint"
    ])
    # Return concatenated pos and vel
    joint_pos = robot.data.joint_pos[:, hand_joint_indices]
    joint_vel = robot.data.joint_vel[:, hand_joint_indices]
    return torch.cat([joint_pos, joint_vel], dim=-1)

def get_all_robot_link_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Get all robot link states for full state estimation."""
    robot = env.scene.articulations["robot"]
    # Combine positions and orientations of all bodies
    return robot.data.body_states_w.reshape(robot.data.body_states_w.shape[0], -1)

def object_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Get object state relative to right hand frame."""
    # Get object state in world frame
    object_pos_w = env.scene.rigid_objects["object"].data.root_pos_w
    object_rot_w = env.scene.rigid_objects["object"].data.root_quat_w
    # Get right hand state
    robot = env.scene.articulations["robot"]
    right_hand_idx = robot.find_bodies(["right_hand_link"])[0]
    hand_pos_w = robot.data.body_states_w[right_hand_idx, :3]
    hand_rot_w = robot.data.body_states_w[right_hand_idx, 3:7]
    # Transform object pose to hand frame (you'll need transformation utility)
    # For now, return world frame states
    return torch.cat([object_pos_w - hand_pos_w, object_rot_w], dim=-1)
