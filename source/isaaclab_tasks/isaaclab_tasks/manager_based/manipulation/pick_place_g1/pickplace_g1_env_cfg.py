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
 
from isaaclab_assets.robots.unitree import G1_WITH_HAND_CFG  # isort: skip
from isaaclab.sensors import CameraCfg
 
import carb
carb_settings_iface = carb.settings.get_settings()


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.15], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.40, 1.0413], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CylinderCfg(
            radius=0.018,
            height=0.15,
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
    # Object 1: Red Cube
    object1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeRed",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.10, 0.35, 1.0413], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )
    # Object 2: Green Cube
    object2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeGreen",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, 1.0413], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )

    # Object 3: Blue Cube
    object3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeYellow",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.30, 0.40, 1.0413], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )

    """
    object_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/ShopTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 1.6, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Samples/Examples/FrankaNutBolt/SubUSDs/Shop_Table/Shop_Table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            scale=(0.01, 0.01, 0.01),
        ),
    )
    object_can = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ChefCan",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.1, 1.0, 0.9], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path="D:/chef_can.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), metallic=1.0),
        ),
    )
    test_env = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TestEnv",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path="D:/test_env.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )"""

    # Humanoid robot (Unitree G1 with hand)
    robot: ArticulationCfg = G1_WITH_HAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.82),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": -0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": -0.0,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": -0.0,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # legs
                ".*_hip_pitch_joint": -0.0,
                ".*_hip_roll_joint": 0.0,
                ".*_hip_yaw_joint": 0.0,
                ".*_knee_joint": 0.0,
                ".*_ankle_pitch_joint": -0.0,
                ".*_ankle_roll_joint": 0.0,
                # waist
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
                # hands
                "left_hand_index_0_joint": 0.0,
                "left_hand_index_1_joint": 0.0,
                "left_hand_middle_0_joint": 0.0,
                "left_hand_middle_1_joint": 0.0,
                "left_hand_thumb_0_joint": 0.0,
                "left_hand_thumb_1_joint": 0.0,
                "left_hand_thumb_2_joint": 0.0,
                "right_hand_index_0_joint": 0.0,
                "right_hand_index_1_joint": 0.0,
                "right_hand_middle_0_joint": 0.0,
                "right_hand_middle_1_joint": 0.0,
                "right_hand_thumb_0_joint": 0.0,
                "right_hand_thumb_1_joint": 0.0,
                "right_hand_thumb_2_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Sensors
    if carb_settings_iface.get("/isaaclab/cameras_enabled"):
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_camera",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0, focus_distance=5.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.55), rot=(0.67439, 0.21263, -0.21263, -0.67439), convention="opengl"),
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
        pink_controlled_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ],
        # Joints to be locked in URDF
        ik_urdf_fixed_joint_names=[
            'left_hip_pitch_joint', 
            'right_hip_pitch_joint', 
            'waist_yaw_joint', 
            'left_hip_roll_joint', 
            'right_hip_roll_joint', 
            'waist_roll_joint', 
            'left_hip_yaw_joint', 
            'right_hip_yaw_joint', 
            'waist_pitch_joint', 
            'left_knee_joint', 
            'right_knee_joint', 
            'left_ankle_pitch_joint', 
            'right_ankle_pitch_joint', 
            'left_ankle_roll_joint', 
            'right_ankle_roll_joint', 
            'left_hand_index_0_joint', 
            'left_hand_middle_0_joint', 
            'left_hand_thumb_0_joint', 
            'right_hand_index_0_joint', 
            'right_hand_middle_0_joint', 
            'right_hand_thumb_0_joint', 
            'left_hand_index_1_joint', 
            'left_hand_middle_1_joint', 
            'left_hand_thumb_1_joint', 
            'right_hand_index_1_joint', 
            'right_hand_middle_1_joint', 
            'right_hand_thumb_1_joint', 
            'left_hand_thumb_2_joint', 
            'right_hand_thumb_2_joint',
        ],
        hand_joint_names=[
            "left_hand_index_0_joint",
            "left_hand_middle_0_joint",
            "left_hand_thumb_0_joint",
            "right_hand_index_0_joint",
            "right_hand_middle_0_joint",
            "right_hand_thumb_0_joint",
            "left_hand_index_1_joint",
            "left_hand_middle_1_joint",
            "left_hand_thumb_1_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_1_joint",
            "right_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "right_hand_thumb_2_joint",
        ],
        # the robot in the sim scene we are controlling
        asset_name="robot",
        # Configuration for the IK controller
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=14,
            show_ik_warnings=False,
            variable_input_tasks=[
                FrameTask(
                    "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
                    position_cost=1.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTask(
                    "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
                    position_cost=1.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
            ],
            fixed_input_tasks=[],
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

        left_eef_pos = ObsTerm(func=mdp.get_left_eef_pos)
        left_eef_quat = ObsTerm(func=mdp.get_left_eef_quat)
        right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos)
        right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)

        hand_joint_state = ObsTerm(func=mdp.get_hand_state)

        object = ObsTerm(func=mdp.object_obs)
        
        if carb_settings_iface.get("/isaaclab/cameras_enabled"):
            rgb_image = ObsTerm(
                func=base_mdp.image, 
                params={
                    "sensor_cfg": SceneEntityCfg("camera"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
            )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # time_out = DoneTerm(func=mdp.time_out, time_out=False)

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
    """Configuration for the Unitree G1 pick-and-place environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action to hold robot in default pose
    # Action format: [left arm pos (3), left arm quat (4), right arm pos (3), right arm quat (4),
    #                 left hand joint pos (7), right hand joint pos (7)]
    idle_action = torch.tensor([
        0.22878,  # left arm pos x
        0.2536,    # left arm pos y
        1.0953,    # left arm pos z
        0.5,       # left arm quat
        0.5,
        -0.5,
        0.5,
        0.22878,   # right arm pos x
        0.2536,    # right arm pos y
        1.0953,    # right arm pos z
        0.5,       # right arm quat
        0.5,
        -0.5,
        0.5,
        0.0,       # left_hand_index_0_joint
        0.0,       # left_hand_middle_0_joint
        0.0,       # left_hand_thumb_0_joint
        0.0,       # right_hand_index_0_joint
        0.0,       # right_hand_middle_0_joint
        0.0,       # right_hand_thumb_0_joint
        0.0,       # left_hand_index_1_joint
        0.0,       # left_hand_middle_1_joint
        0.0,       # left_hand_thumb_1_joint
        0.0,       # right_hand_index_1_joint
        0.0,       # right_hand_middle_1_joint
        0.0,       # right_hand_thumb_1_joint
        0.0,       # left_hand_thumb_2_joint
        0.0,       # right_hand_thumb_2_joint
    ])

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2  # policy update rate 1/60 * 5 Hz = 12 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 60  # 60 Hz
        self.sim.render_interval = 2  # 30 Hz

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