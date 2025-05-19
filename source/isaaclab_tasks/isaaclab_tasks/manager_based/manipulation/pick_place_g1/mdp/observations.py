# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation terms for G1 pick-and-place."""

from turtle import left, right
import torch
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils

# from .pickplace_g1_env_cfg import _G1_RIGHT_HAND_JOINT_NAMES_ORDERED
# If that causes circular import, define it here:
_G1_RIGHT_HAND_JOINT_NAMES_ORDERED_OBS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]

def object_obs(env:ManagerBasedRLEnv) -> torch.Tensor:
    """ Object Observations (in world frame):
        object pos, object quat, left_eef to object, right_eef to object
    """
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_wrist_yaw_link")
    right_eef_idx = env.scene["robot"].data.body_names.index("right_wrist_yaw_link")
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins
    
    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    object_rot = env.scene["object"].data.root_quat_w
    
    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos
    
    return torch.cat((
        object_pos, object_rot, left_eef_to_object, right_eef_to_object
        ), dim=-1)


def get_right_eef_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the right end-effector (palm) position in world frame."""
    right_eef_idx = env.scene["robot"].data.body_names.index("right_wrist_yaw_link")
    body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = body_pos_w[:, right_eef_idx] - env.scene.env_origins
    
    return right_eef_idx

def get_right_eef_quat(env: ManagerBasedRLEnv) -> torch.Tensor:    
    body_quat_w = env.scene["robot"].data.body_quat_w
    right_eef_idx = env.scene["robot"].data.body_names.index("right_wrist_yaw_link")
    right_eef_quat = body_quat_w[:, right_eef_idx]

    return right_eef_quat

def get_left_eef_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    left_eef_idx = env.scene["robot"].data.body_names.index("left_wrist_yaw_link")
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    
    return left_eef_idx

def get_left_eef_quat(env: ManagerBasedRLEnv) -> torch.Tensor:    
    body_quat_w = env.scene["robot"].data.body_quat_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_wrist_yaw_link")
    left_eef_quat = body_quat_w[:, left_eef_idx]

    return left_eef_quat

def get_hand_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    hand_joint_states = env.scene["robot"].data.joint_pos[:, -14:]  # Hand joints are last 14 entries of joint state
    return hand_joint_states


def get_all_robot_link_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_link_state_w[:, :, :]
    all_robot_link_pos = body_pos_w

    return all_robot_link_pos


# Available strings: [
# 'pelvis', 
# 'left_hip_pitch_link', 'right_hip_pitch_link', 'waist_yaw_link', 
# 'left_hip_roll_link', 'right_hip_roll_link', 'waist_roll_link', 
# 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 
# 'left_knee_link', 'right_knee_link', 
# 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_ankle_pitch_link', 'right_ankle_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_elbow_link', 'right_elbow_link', 
# 'left_wrist_roll_link', 
# 'right_wrist_roll_link', 'left_wrist_pitch_link', 'right_wrist_pitch_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link', 
# 'left_hand_index_0_link', 'left_hand_middle_0_link', 'left_hand_thumb_0_link', 'right_hand_index_0_link', 'right_hand_middle_0_link', 'right_hand_thumb_0_link', 'left_hand_index_1_link', 'left_hand_middle_1_link', 'left_hand_thumb_1_link', 'right_hand_index_1_link', 'right_hand_middle_1_link', 'right_hand_thumb_1_link', 'left_hand_thumb_2_link', 'right_hand_thumb_2_link']
