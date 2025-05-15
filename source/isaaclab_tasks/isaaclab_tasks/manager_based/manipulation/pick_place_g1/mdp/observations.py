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
    # G1's right palm link is the EEF
    right_palm_idx = robot.find_bodies(["right_palm_link"])[0]
    return robot.data.body_states_w[right_palm_idx, :3]

def get_right_eef_quat(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the right end-effector orientation in world frame."""
    robot = env.scene.articulations["robot"]
    right_palm_idx = robot.find_bodies(["right_palm_link"])[0]
    return robot.data.body_states_w[right_palm_idx, 3:7]

def get_hand_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the right hand joint states (positions and velocities)."""
    robot = env.scene.articulations["robot"]
    # Get indices for right hand joints only
    right_hand_joint_indices = robot.find_joints([
        "right_zero_joint", "right_one_joint", "right_two_joint",
        "right_three_joint", "right_four_joint", "right_five_joint",
        "right_six_joint"
    ])
    # Return concatenated pos and vel for right hand joints
    joint_pos = robot.data.joint_pos[:, right_hand_joint_indices]
    joint_vel = robot.data.joint_vel[:, right_hand_joint_indices]
    return torch.cat([joint_pos, joint_vel], dim=-1)

def get_all_robot_link_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Get all robot link states for full state estimation."""
    robot = env.scene.articulations["robot"]
    # Combine positions and orientations of all bodies
    return robot.data.body_states_w.reshape(robot.data.body_states_w.shape[0], -1)

def object_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Get object state relative to right palm link frame."""
    # Get object state in world frame
    object_pos_w = env.scene.rigid_objects["object"].data.root_pos_w
    object_rot_w = env.scene.rigid_objects["object"].data.root_quat_w
    # Get right palm link state
    robot = env.scene.articulations["robot"]
    right_palm_idx = robot.find_bodies(["right_palm_link"])[0]
    palm_pos_w = robot.data.body_states_w[right_palm_idx, :3]
    palm_rot_w = robot.data.body_states_w[right_palm_idx, 3:7]
    # Transform object pose to palm link frame (you'll need transformation utility)
    # For now, return world frame states relative to palm position
    # Note: Full transformation requires quaternion math, which is not implemented here.
    # This observation might need refinement based on how the environment uses it.
    return torch.cat([object_pos_w - palm_pos_w, object_rot_w], dim=-1)
