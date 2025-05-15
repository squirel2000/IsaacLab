# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import torch

from pink.tasks import FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

# Import G1_CFG instead of GR1T2_CFG
from isaaclab_assets.robots.unitree import G1_CFG  # isort: skip

# Pre-defined G1 Hand joint positions for open/closed state
# IMPORTANT: Verify these joint values and order based on G1's actual model and desired grip
G1_HAND_JOINTS_OPEN = {
    # Left Hand (match order in hand_joint_names)
    "left_zero_joint": 0.0, "left_one_joint": 0.0, "left_two_joint": 0.0,
    "left_three_joint": 0.0, "left_four_joint": 0.0, "left_five_joint": 0.0,
    "left_six_joint": 0.0,
    # Right Hand (match order in hand_joint_names)
    "right_zero_joint": 0.0, "right_one_joint": 0.0, "right_two_joint": 0.0,
    "right_three_joint": 0.0, "right_four_joint": 0.0, "right_five_joint": 0.0,
    "right_six_joint": 0.0,
}
G1_HAND_JOINTS_CLOSED = {
    # Left Hand (match order in hand_joint_names) - Keep open as we control right
    "left_zero_joint": 0.0, "left_one_joint": 0.0, "left_two_joint": 0.0,
    "left_three_joint": 0.0, "left_four_joint": 0.0, "left_five_joint": 0.0,
    "left_six_joint": 0.0,
    # Right Hand (match order in hand_joint_names) - Define a closed pose
    "right_zero_joint": 1.5, "right_one_joint": 1.5, "right_two_joint": 1.5,
    "right_three_joint": 1.5, "right_four_joint": 1.5, "right_five_joint": 1.5,
    "right_six_joint": 1.5, # Adjust these values for desired grip
}
# Convert to ordered lists based on hand_joint_names defined later in ActionsCfg
# We will define the ordered lists inside ActionsCfg where hand_joint_names is defined.

##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.55, 0.0), rot=(1.0, 0.0, 0.0, 0.0)), # Changed to tuple
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, 0.40, 1.0413), rot=(1.0, 0.0, 0.0, 0.0)), # Changed to tuple
        spawn=sim_utils.CylinderCfg(
            radius=0.018,
            height=0.35,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), metallic=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )

    # Humanoid robot w/ arms higher
    # Use G1_CFG instead of GR1T2_CFG
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=G1_CFG.spawn, # Use spawn config from G1_CFG
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74), # Adjusted height for G1 based on unitree.py (as tuple)
            rot=(0.7071, 0.0, 0.0, 0.7071), # Keep orientation (as tuple)
            joint_pos={
                # right-arm (G1 joints)
                "right_shoulder_pitch_joint": 0.35, # Initial pose from G1_CFG
                "right_shoulder_roll_joint": -0.16, # Initial pose from G1_CFG
                "right_shoulder_yaw_joint": 0.0, # Assuming 0 for yaw
                "right_elbow_pitch_joint": 0.87, # Initial pose from G1_CFG
                "right_elbow_roll_joint": 0.0, # Assuming 0 for roll
                # left-arm (G1 joints)
                "left_shoulder_pitch_joint": 0.35, # Initial pose from G1_CFG
                "left_shoulder_roll_joint": 0.16, # Initial pose from G1_CFG
                "left_shoulder_yaw_joint": 0.0, # Assuming 0 for yaw
                "left_elbow_pitch_joint": 0.87, # Initial pose from G1_CFG
                "left_elbow_roll_joint": 0.0, # Assuming 0 for roll
                # hands (G1 joints)
                "left_one_joint": 1.0,
                "right_one_joint": -1.0,
                "left_two_joint": 0.52,
                "right_two_joint": -0.52,
                # -- other joints from G1_CFG
                ".*_hip_pitch_joint": -0.20,
                ".*_knee_joint": 0.42,
                ".*_ankle_pitch_joint": -0.23,
                "left_shoulder_roll_joint": 0.16,
                "left_shoulder_pitch_joint": 0.35,
                "right_shoulder_roll_joint": -0.16,
                "right_shoulder_pitch_joint": 0.35,
            },
            joint_vel={".*": 0.0},
        ),
        actuators=G1_CFG.actuators, # Use actuators from G1_CFG
        soft_joint_pos_limit_factor=G1_CFG.soft_joint_pos_limit_factor, # Use from G1_CFG
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pink_ik_cfg = PinkInverseKinematicsActionCfg(
        # Update controlled joint names for G1 arms
        pink_controlled_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_elbow_roll_joint", # Added elbow roll for G1
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_elbow_roll_joint", # Added elbow roll for G1
        ],
        # Joints to be locked in URDF (updated for G1)
        ik_urdf_fixed_joint_names=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
            "torso_joint",
            # Hand joints (all hand joints are fixed for IK control of the arm pose)
            "left_five_joint",
            "left_three_joint",
            "left_six_joint",
            "left_four_joint",
            "left_zero_joint",
            "left_one_joint",
            "left_two_joint",
            "right_five_joint",
            "right_three_joint",
            "right_six_joint",
            "right_four_joint",
            "right_zero_joint",
            "right_one_joint",
            "right_two_joint",
        ],
        # Hand joint names for direct control (G1 hand joints)
        hand_joint_names=[
            "left_five_joint",
            "left_three_joint",
            "left_six_joint",
            "left_four_joint",
            "left_zero_joint",
            "left_one_joint",
            "left_two_joint",
            "right_five_joint",
            "right_three_joint",
            "right_six_joint",
            "right_four_joint",
            "right_zero_joint",
            "right_one_joint",
            "right_two_joint",
        ],
        # the robot in the sim scene we are controlling
        asset_name="robot",
        # Configuration for the IK controller
        # The frames names are the ones present in the URDF file
        # The urdf has to be generated from the USD that is being used in the scene
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link",
            # Need to update num_hand_joints for G1 (14 joints per hand)
            num_hand_joints=14, # G1 has 14 hand joints per hand
            show_ik_warnings=False,
            variable_input_tasks=[
                # Need to update link names for G1 hands
                FrameTask(
                    "left_hand_link", # Placeholder - need to find actual link name
                    position_cost=1.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.1,
                ),
                FrameTask(
                    "right_hand_link", # Placeholder - need to find actual link name
                    position_cost=1.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.1,
                ),
            ],
            fixed_input_tasks=[
                # COMMENT OUT IF LOCKING WAIST/HEAD
                # FrameTask(
                #     "GR1T2_fourier_hand_6dof_head_yaw_link", # Need to update for G1
                #     position_cost=1.0,  # [cost] / [m]
                #     orientation_cost=0.05,  # [cost] / [rad]
                # ),
            ],
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        # Need to update these observation terms for G1
        left_eef_pos = ObsTerm(func=mdp.get_left_eef_pos) # Need to check if these functions work for G1
        left_eef_quat = ObsTerm(func=mdp.get_left_eef_quat) # Need to check if these functions work for G1
        right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos) # Need to check if these functions work for G1
        right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat) # Need to check if these functions work for G1

        hand_joint_state = ObsTerm(func=mdp.get_hand_state) # Need to check if this function works for G1
        head_joint_state = ObsTerm(func=mdp.get_head_state) # Need to check if this function works for G1

        object = ObsTerm(func=mdp.object_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.task_done)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.05, 0.0],
                "y": [0.0, 0.05],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickPlaceG1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg() # EventCfg is defined above

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(0.0, 0.0, 0.0), # Changed to 3 elements as per error message
    )

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action to hold robot in default pose (Need to update for G1)
    # Action format: [right arm pos (3), right arm quat (4), right hand joint pos (14)]
    # Assuming we only control the right arm and hand for this task
    idle_action = torch.tensor([
        0.22878, # Example pos_x (need to verify)
        0.2536,  # Example pos_y (need to verify)
        1.0953,  # Example pos_z (need to verify)
        0.5,     # Example quat_w (need to verify)
        0.5,     # Example quat_x (need to verify)
        -0.5,    # Example quat_y (need to verify)
        0.5,     # Example quat_z (need to verify)
        # Right hand joint positions (14 joints) - need to verify order and values
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 60  # 100Hz
        self.sim.render_interval = 2

        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )
        ControllerUtils.change_revolute_to_fixed(
            temp_urdf_output_path, self.actions.pink_ik_cfg.ik_urdf_fixed_joint_names
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
        self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path

        # Print the path to the generated URDF file
        print(f"Generated G1 URDF path: {temp_urdf_output_path}")
