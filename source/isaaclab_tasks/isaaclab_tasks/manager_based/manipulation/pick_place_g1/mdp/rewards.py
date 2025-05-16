# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for G1 pick-and-place task."""

import torch
from isaaclab.envs import ManagerBasedEnv
# if you use observations from your mdp.observations, import them
from .observations import get_right_eef_pos

def tracking_object_reward(env: ManagerBasedEnv, std: float = 0.1) -> torch.Tensor:
    """Reward for tracking/reaching the object with right hand, based on distance."""
    object_pos = env.scene.rigid_objects["object"].data.root_pos_w
    # Use the observation function for consistency if available and it returns world pos
    hand_pos = get_right_eef_pos(env) # This gets world position of right_palm_link

    distance = torch.norm(object_pos - hand_pos, p=2, dim=-1)
    # Gaussian-like reward: exp(-distance^2 / (2 * std^2))
    reward = torch.exp(-torch.square(distance) / (2 * std**2))
    return reward

def lifting_reward(env: ManagerBasedEnv, target_lift_height: float = 0.15) -> torch.Tensor:
    """Reward for lifting the object above a certain height from its initial spawn."""
    # Assuming initial height is related to table height or a known spawn Z.
    # For simplicity, let's use current height relative to a dynamic "lifted" threshold.
    # A better way would be to store initial object height at reset.
    # Here, we check if object is above its spawn Z + target_lift_height
    
    # This requires knowing the object's initial Z. If object is reset to a fixed height:
    initial_height_on_table = env.scene.cfg.object.init_state.pos[2] # Example: 0.8 for cube on table
                                                                    # Or get from observations if it's randomized

    object_current_height = env.scene.rigid_objects["object"].data.root_pos_w[..., 2]
    
    is_lifted = object_current_height > (initial_height_on_table + target_lift_height / 2) # Small lift
    bonus_for_higher_lift = torch.clamp(
        (object_current_height - (initial_height_on_table + target_lift_height / 2)) / target_lift_height,
        0.0, 1.0 # Normalize bonus
    )
    
    reward = torch.where(is_lifted, 0.5 + 0.5 * bonus_for_higher_lift, torch.zeros_like(object_current_height))
    return reward


def reaching_target_reward(env: ManagerBasedEnv, target_pos_tensor: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """Reward for placing the object near a target position."""
    # target_pos_tensor should be defined and passed, e.g., from a curriculum or fixed goal
    # Example: target_pos = torch.tensor([0.3, 0.4, 0.8], device=env.device)
    
    object_pos = env.scene.rigid_objects["object"].data.root_pos_w
    distance = torch.norm(object_pos - target_pos_tensor.unsqueeze(0).repeat(env.num_envs, 1), p=2, dim=-1)
    
    reward = torch.exp(-torch.square(distance) / (2 * std**2))
    return reward

# Add other rewards like:
# - Hand orientation reward (e.g., for grasping)
# - Smoothness rewards for actions
# - Penalty for excessive joint movement or torques
# - Penalty for object dropping (can also be a termination)