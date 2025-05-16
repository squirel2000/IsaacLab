# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation terms for G1 pick-and-place."""

import torch

import isaaclab.utils.torch as torch_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.common import last_action # For last_action obs

# Assuming _G1_RIGHT_HAND_JOINT_NAMES_ORDERED is accessible here or defined
# For simplicity, let's redefine it or ensure it's imported.
# from .pickplace_g1_env_cfg import _G1_RIGHT_HAND_JOINT_NAMES_ORDERED
# If that causes circular import, define it here:
_G1_RIGHT_HAND_JOINT_NAMES_ORDERED_OBS = [
    "right_five_joint", "right_three_joint", "right_six_joint",
    "right_four_joint", "right_zero_joint", "right_one_joint",
    "right_two_joint",
]


def get_right_eef_pos(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the right end-effector (palm) position in world frame."""
    robot = env.scene.articulations["robot"]
    # Ensure "right_palm_link" is a valid body name in G1's USD/URDF
    right_palm_indices = robot.find_bodies("right_palm_link")[0] # Get first element if it returns a list of lists
    return robot.data.body_pos_w[:, right_palm_indices].squeeze(-2) # (num_envs, 3)

def get_right_eef_quat(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the right end-effector (palm) orientation (quat XYZW) in world frame."""
    robot = env.scene.articulations["robot"]
    right_palm_indices = robot.find_bodies("right_palm_link")[0]
    return robot.data.body_rot_w[:, right_palm_indices].squeeze(-2) # (num_envs, 4) XYZW

def get_right_hand_joint_pos(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the G1 right hand joint positions."""
    robot = env.scene.articulations["robot"]
    # Order of joints must match _G1_RIGHT_HAND_JOINT_NAMES_ORDERED_OBS
    joint_ids = robot.find_joints(_G1_RIGHT_HAND_JOINT_NAMES_ORDERED_OBS)[0]
    return robot.data.joint_pos[:, joint_ids]

def get_right_hand_joint_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the G1 right hand joint velocities."""
    robot = env.scene.articulations["robot"]
    joint_ids = robot.find_joints(_G1_RIGHT_HAND_JOINT_NAMES_ORDERED_OBS)[0]
    return robot.data.joint_vel[:, joint_ids]

# Example for relative observation (more advanced, requires careful implementation)
def eef_to_object_pos_relative(env: ManagerBasedEnv):
    """Object position relative to EEF, expressed in EEF frame."""
    object_pos_w = env.scene.rigid_objects["object"].data.root_pos_w
    eef_pos_w = get_right_eef_pos(env)
    eef_quat_w = get_right_eef_quat(env) # XYZW

    obj_pos_in_eef_frame = torch_utils.quat_rotate_inverse(eef_quat_w, object_pos_w - eef_pos_w)
    return obj_pos_in_eef_frame

def eef_to_object_rot_relative(env: ManagerBasedEnv):
    """Object orientation relative to EEF orientation."""
    object_rot_w = env.scene.rigid_objects["object"].data.root_quat_w # XYZW
    eef_quat_w = get_right_eef_quat(env) # XYZW

    # Relative_rotation = q_object * conjugate(q_eef)
    relative_quat = torch_utils.quat_mul(object_rot_w, torch_utils.quat_conjugate(eef_quat_w))
    return relative_quat

# You might need these if you had left arm observations enabled in the config
# def get_left_eef_pos(env: ManagerBasedEnv) -> torch.Tensor:
#     robot = env.scene.articulations["robot"]
#     left_palm_indices = robot.find_bodies("left_palm_link")[0]
#     return robot.data.body_pos_w[:, left_palm_indices].squeeze(-2)

# def get_left_eef_quat(env: ManagerBasedEnv) -> torch.Tensor:
#     robot = env.scene.articulations["robot"]
#     left_palm_indices = robot.find_bodies("left_palm_link")[0]
#     return robot.data.body_rot_w[:, left_palm_indices].squeeze(-2)

def get_all_robot_link_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Get all robot link states (pos and quat) flattened."""
    # This can be very high dimensional.
    # body_pos_w is (num_envs, num_bodies, 3)
    # body_rot_w is (num_envs, num_bodies, 4)
    robot = env.scene.articulations["robot"]
    pos = robot.data.body_pos_w.reshape(env.num_envs, -1)
    rot = robot.data.body_rot_w.reshape(env.num_envs, -1)
    # Optionally add velocities too
    # lin_vel = robot.data.body_lin_vel_w.reshape(env.num_envs, -1)
    # ang_vel = robot.data.body_ang_vel_w.reshape(env.num_envs, -1)
    return torch.cat([pos, rot], dim=-1)

# The 'object_obs' from the original prompt can be replaced by the relative ones above or world frame ones.
# If you need it as originally defined:
def object_obs_original_style(env: ManagerBasedEnv) -> torch.Tensor:
    """Get object state relative to right palm link frame (simplified)."""
    object_pos_w = env.scene.rigid_objects["object"].data.root_pos_w
    object_rot_w = env.scene.rigid_objects["object"].data.root_quat_w # XYZW
    palm_pos_w = get_right_eef_pos(env)
    # This is a simplified relative observation. True relative pose needs frame transformation.
    return torch.cat([object_pos_w - palm_pos_w, object_rot_w], dim=-1)