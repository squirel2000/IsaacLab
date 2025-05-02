# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from isaaclab.sensors import RayCaster # Make sure RayCaster is imported
import isaaclab.envs.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


# --- Helper ---
def _compute_terrain_roughness_raw(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Uses the height scanner to detect flat terrain, returns shape (num_envs,)."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # Assuming variance handles it or non-hits are filtered/ignored by sensor processing.
    height_diffs = sensor.data.pos_w[:, 2].unsqueeze(-1) - sensor.data.ray_hits_w[..., 2]  # (N, num_rays) = (N, 1) - (N, num_rays)
    roughness = torch.var(height_diffs, dim=1, unbiased=False) # (N,)
    return roughness

def straight_leg_bonus_on_flat(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"), flat_var_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalizes bending the specified joints (knees) when the terrain is flat."""
    # # Access the height scanner sensor
    height_scan_var = _compute_terrain_roughness_raw(env, sensor_cfg) # Return (N,)
    # Detect flat terrain
    is_flat = height_scan_var < flat_var_threshold
    
    # Get joint positions (assume knees are named "knee_joint")
    asset = env.scene[asset_cfg.name]
    knee_indices = [i for i, n in enumerate(asset.data.joint_names) if "knee" in n.lower()]
    if len(knee_indices) == 0:
        # If no knee joints match the pattern, return zero penalty
        # print(f"WARNING: No joints matching 'knee' found for penalty.") # Uncomment for debugging
        return torch.zeros(asset.data.joint_pos.shape[0], device=asset.data.joint_pos.device)
    knee_angles = asset.data.joint_pos[:, knee_indices]
    # Penalize bent knees, reward straight legs on flat
    penalty = torch.sum(torch.abs(knee_angles), dim=1)
    # Only apply when on flat
    return penalty * is_flat.float()


def adaptive_joint_deviation(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, flat_threshold: float, flat_penalty: float, rough_penalty: float
) -> torch.Tensor:
    """Penalize joint deviations more on flat terrain and less on rough. Returns shape (num_envs,)."""
    deviation = mdp.joint_deviation_l1(env, asset_cfg) # Shape (N,)
    height_scan_var = _compute_terrain_roughness_raw(env, sensor_cfg) # Shape (N,)

    penalty = torch.where(
        height_scan_var < flat_threshold, # Condition (N,)
        flat_penalty * deviation,  # (N,)
        rough_penalty * deviation  # (N,)
    ) # Result (N,)
    return penalty