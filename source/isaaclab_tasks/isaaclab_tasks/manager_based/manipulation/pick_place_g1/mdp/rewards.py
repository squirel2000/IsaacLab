# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for G1 pick-and-place task."""

import torch
from isaaclab.envs import ManagerBasedEnv

def tracking_object_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward for tracking/reaching the object with right hand."""
    # Get object and hand positions
    object_pos = env.scene.rigid_objects["object"].data.root_pos_w
    robot = env.scene.articulations["robot"]
    right_hand_idx = robot.find_bodies(["right_hand_link"])[0]
    hand_pos = robot.data.body_states_w[right_hand_idx, :3]
    
    # Calculate distance-based reward
    distance = torch.norm(object_pos - hand_pos, dim=-1)
    reward = 1.0 / (1.0 + distance)  # Normalized [0,1]
    return reward

def lifting_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward for lifting the object."""
    object_height = env.scene.rigid_objects["object"].data.root_pos_w[..., 2]
    initial_height = 1.0413  # From scene config
    min_lift = 0.1  # Minimum lift height for reward
    
    # Reward for lifting above minimum height
    lift_amount = object_height - initial_height
    reward = torch.where(lift_amount > min_lift, 
                        1.0 + 0.5 * lift_amount,  # Bonus for higher lifts
                        0.0)
    return reward

def reaching_target_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward for reaching target placement position."""
    # Get current object position
    object_pos = env.scene.rigid_objects["object"].data.root_pos_w
    
    # Define target position (customize based on task)
    target_pos = torch.tensor([0.3, 0.4, 1.0], device=env.device)
    
    # Distance-based reward
    distance = torch.norm(object_pos - target_pos, dim=-1)
    reward = 1.0 / (1.0 + distance)
    
    # Bonus for very close placement
    close_threshold = 0.05
    reward = torch.where(distance < close_threshold,
                        reward + 1.0,  # Bonus for close placement
                        reward)
    return reward
