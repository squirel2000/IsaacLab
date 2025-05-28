# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation terms for G1 pick-and-place."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

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


# import random
# from pxr import UsdShade, Gf  # For USD material manipulation
# import carb  # For logging

# import isaaclab.envs.mdp as base_mdp # Import base_mdp for existing functions


# def reset_cube_with_random_color(
#     env: ManagerBasedRLEnv,
#     env_ids: torch.Tensor, # Add env_ids to accept it from EventManager
#     event_params: dict
# ):
#     """Resets the root state of the specified asset and applies a random color from the list
#     by directly modifying the diffuseColor input of its UsdPreviewSurface shader."""
#     # Extract parameters from the event_params dictionary
#     # The env_ids received here could be used to target specific environments if needed.
#     pose_range: dict = event_params["pose_range"]
#     colors: list[tuple[float, float, float]] = event_params["colors"]
#     target_object_id: str = event_params["target_object_id"]
#     velocity_range: dict | None = event_params.get("velocity_range")

#     # Get the configured prim_path expression from the asset's configuration
#     asset_prim_path_expr = env.scene[target_object_id].cfg.prim_path
#     # Create a temporary SceneEntityCfg. This is a bit of a workaround.
#     temp_asset_cfg = SceneEntityCfg(target_object_id, asset_prim_path_expr)

#     if velocity_range is None:
#         velocity_range = {}

#     # Reset pose for all instances of the asset
#     base_mdp.reset_root_state_uniform(
#         env,
#         asset_cfg=temp_asset_cfg, # Use the dynamically created cfg
#         pose_range=pose_range,
#         velocity_range=velocity_range,
#         # Use the provided env_ids if you want the event to be targeted,
#         # or use all envs if the event is meant to be global for this call.
#         env_ids=env_ids # Or torch.arange(env.num_envs, device=env.device) if always all
#     )

#     asset = env.scene[target_object_id]
#     stage = env.sim.stage

#     for i in range(env.num_envs):
#         selected_color_tuple = random.choice(colors)
#         selected_color_gf = Gf.Vec3f(selected_color_tuple[0], selected_color_tuple[1], selected_color_tuple[2])

#         prim_path_str = asset.prim_paths[i]
#         prim = stage.GetPrimAtPath(prim_path_str)

#         if not prim.IsValid():
#             carb.log_warn(f"In reset_cube_red_with_random_color: Prim at path {prim_path_str} is not valid. Cannot set color.")
#             continue

#         material_api = UsdShade.MaterialBindingAPI(prim)
#         binding_rel = material_api.GetDirectBindingRel()

#         if not binding_rel.HasAuthoredTargets():
#             carb.log_warn(f"In reset_cube_red_with_random_color: No direct material binding found for prim {prim_path_str}. Cannot set color.")
#             continue

#         material_path = binding_rel.GetTargets()[0]
#         material_prim = stage.GetPrimAtPath(material_path)

#         if not material_prim.IsValid():
#             carb.log_warn(f"In reset_cube_red_with_random_color: Material prim at path {material_path} for {prim_path_str} is not valid.")
#             continue

#         found_shader = None
#         for child_prim_in_mat in material_prim.GetChildren():
#             if child_prim_in_mat.IsA(UsdShade.Shader):
#                 shader_candidate = UsdShade.Shader(child_prim_in_mat)
#                 if shader_candidate.GetIdAttr().Get() == "UsdPreviewSurface":
#                     found_shader = shader_candidate
#                     break
        
#         if not found_shader: # Fallback for shaders not identified by ID but by common name
#             shader_at_default_path = UsdShade.Shader(stage.GetPrimAtPath(material_prim.GetPath().AppendChild("Shader")))
#             if shader_at_default_path and shader_at_default_path.GetPrim().IsValid() and shader_at_default_path.GetIdAttr().Get() == "UsdPreviewSurface":
#                 found_shader = shader_at_default_path

#         if found_shader:
#             diffuse_input = found_shader.GetInput("diffuseColor")
#             if diffuse_input:
#                 diffuse_input.Set(selected_color_gf)
#             else:
#                 carb.log_warn(f"In reset_cube_red_with_random_color: Could not find 'diffuseColor' input on shader {found_shader.GetPath()} for {prim_path_str}")
#         else:
#             carb.log_warn(f"In reset_cube_red_with_random_color: Could not find UsdPreviewSurface shader for material {material_path} of {prim_path_str}")


# def get_object1_color_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """
#     Retrieves the current diffuse color of the asset (e.g., 'object1')
#     by reading its UsdPreviewSurface shader's diffuseColor input.
#     Returns a tensor of shape (num_envs, 3) with RGB values.
#     """
#     asset = env.scene[asset_cfg.name]

#     # Defensive check for prim_paths, especially during ObservationManager._prepare_terms
#     if not hasattr(asset, "prim_paths"):
#         carb.log_warn(
#             f"Asset '{asset_cfg.name}' of type {type(asset)} does not have 'prim_paths' attribute "
#             f"during observation function call (likely shape inference). Returning default color tensor."
#         )
#         # For shape inference, we must return a tensor of the expected shape (num_envs, 3)
#         return torch.full((env.num_envs, 3), 0.5, device=env.device, dtype=torch.float32)

#     # Check if prim_paths exists but is empty, and we need to iterate up to num_envs
#     # This can happen if the prim_path expression in the asset config doesn't match any prims.
#     if not asset.prim_paths and env.num_envs > 0:
#         carb.log_warn(
#             f"Asset '{asset_cfg.name}' has an empty prim_paths list. "
#             f"This might be due to its prim_path ('{asset.cfg.prim_path}') not matching any prims in the scene. "
#             f"Returning default color tensor."
#         )
#         return torch.full((env.num_envs, 3), 0.5, device=env.device, dtype=torch.float32)

#     stage = env.sim.stage
#     colors_list = []

#     for i in range(env.num_envs):
#         # Default color if issues arise for a specific environment instance
#         current_color_rgb = [0.5, 0.5, 0.5]  # Default color if not found

#         if i < len(asset.prim_paths):
#             prim_path_str = asset.prim_paths[i]
#             prim = stage.GetPrimAtPath(prim_path_str)

#             if prim.IsValid():
#                 material_api = UsdShade.MaterialBindingAPI(prim)
#                 binding_rel = material_api.GetDirectBindingRel()
#                 if binding_rel.HasAuthoredTargets():
#                     material_path = binding_rel.GetTargets()[0]
#                     material_prim = stage.GetPrimAtPath(material_path)
#                     if material_prim.IsValid():
#                         found_shader = None
#                         for child_prim_in_mat in material_prim.GetChildren():
#                             if child_prim_in_mat.IsA(UsdShade.Shader):
#                                 shader_candidate = UsdShade.Shader(child_prim_in_mat)
#                                 if shader_candidate.GetIdAttr().Get() == "UsdPreviewSurface":
#                                     found_shader = shader_candidate
#                                     break
                        
#                         if not found_shader: # Fallback for shaders not identified by ID but by common name
#                             # Common convention is that the shader prim under material is named "Shader"
#                             shader_prim_at_common_path = material_prim.GetPrim().GetChild("Shader")
#                             if shader_prim_at_common_path.IsValid():
#                                 shader_candidate = UsdShade.Shader(shader_prim_at_common_path)
#                                 if shader_candidate.GetIdAttr().Get() == "UsdPreviewSurface":
#                                     found_shader = shader_candidate

#                         if found_shader:
#                             diffuse_input = found_shader.GetInput("diffuseColor")
#                             if diffuse_input and diffuse_input.IsDefined():
#                                 color_val_gf = diffuse_input.Get()
#                                 if isinstance(color_val_gf, Gf.Vec3f):
#                                     current_color_rgb = [color_val_gf[0], color_val_gf[1], color_val_gf[2]]
#                                 else:
#                                     carb.log_warn(f"Shader {found_shader.GetPath()} 'diffuseColor' input has unexpected type: {type(color_val_gf)}")
#                             else:
#                                 carb.log_warn(f"Could not find 'diffuseColor' input on shader {found_shader.GetPath()} for {prim_path_str}")
#                         else:
#                             carb.log_warn(f"Could not find UsdPreviewSurface shader for material {material_path} of {prim_path_str}")
#                     else:
#                         carb.log_warn(f"Material prim at path {material_path} for {prim_path_str} is not valid.")
#                 else:
#                     carb.log_warn(f"No direct material binding found for prim {prim_path_str}.")
#             else:
#                 carb.log_warn(f"Prim at path {prim_path_str} is not valid for color retrieval.")
#         else:
#             carb.log_warn(
#                 f"Index {i} is out of bounds for asset.prim_paths (len: {len(asset.prim_paths)}) "
#                 f"for asset '{asset_cfg.name}'. Using default color for this environment instance."
#             )
#         colors_list.append(current_color_rgb)

#     return torch.tensor(colors_list, device=env.device, dtype=torch.float32)