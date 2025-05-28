# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym # Corrected import

##
# Register Gym environments.
##

##
# Joint Position Control for G1 (using Pink IK)
##

gym.register(
    id="Isaac-PickPlace-G1-IK-Abs-v0", # Retained "Rel" in ID for consistency, though action space is absolute
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_g1_env_cfg:PickPlaceG1EnvCfg",
    },
    disable_env_checker=True,
)

# Play environment can be added if specific config variations are needed for playback
# For now, the main environment can be used for both live teleop and playback.
# gym.register(
#     id="Isaac-PickPlace-G1-IK-Rel-Play-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.pickplace_g1_env_cfg:PickPlaceG1EnvCfg_PLAY", # If you create a _PLAY config
#     },
#     disable_env_checker=True,
# )