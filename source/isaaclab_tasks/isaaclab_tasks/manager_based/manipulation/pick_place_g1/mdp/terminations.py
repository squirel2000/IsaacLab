# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination terms for G1 pick-and-place."""

import torch
from isaaclab.envs import ManagerBasedEnv

def time_out(env: ManagerBasedEnv) -> torch.Tensor:
    """Check if episode has timed out."""
    return env.episode_length_buf >= env.max_episode_length

def task_done(env: ManagerBasedEnv) -> torch.Tensor:
    """Check if object has been successfully picked and placed."""
    # Get current object position
    object_pos = env.scene.rigid_objects["object"].data.root_pos_w
    
    # Define target position/region (customize based on your task)
    target_pos = torch.tensor([0.3, 0.4, 1.0], device=env.device)
    distance_threshold = 0.05
    
    # Check if object is close to target
    distance = torch.norm(object_pos - target_pos, dim=-1)
    success = distance < distance_threshold
    
    # Can add additional conditions (stability, orientation, etc.)
    return success

def object_dropped(env: ManagerBasedEnv) -> torch.Tensor:
    """Check if object has been dropped or is in invalid state."""
    # Get object height
    object_height = env.scene.rigid_objects["object"].data.root_pos_w[..., 2]
    # Threshold for considering object dropped
    min_height = 0.5  # Adjust based on your scene
    return object_height < min_height
