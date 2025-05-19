# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import torch

from pink.tasks import FrameTask # Make sure pink is installed if this is a direct import

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

from . import mdp # Assuming mdp is a sub-module (observations.py, rewards.py, terminations.py)

# Import G1_29DOF_CFG
from isaaclab_assets.robots.unitree import G1_29DOF_CFG  # isort: skip

# Pre-defined G1 Hand joint positions for open/closed state
# IMPORTANT: Verify these joint values and order based on G1's actual model and desired grip
# These names should match the `hand_joint_names` in ActionsCfg
G1_RIGHT_HAND_JOINT_NAMES_ORDERED = [
    "right_five_joint", "right_three_joint", "right_six_joint",
    "right_four_joint", "right_zero_joint", "right_one_joint",
    "right_two_joint",
]

G1_HAND_JOINTS_OPEN_DICT = {
    "right_zero_joint": 0.0, "right_one_joint": 0.0, "right_two_joint": 0.0,
    "right_three_joint": 0.0, "right_four_joint": 0.0, "right_five_joint": 0.0,
    "right_six_joint": 0.0,
}
G1_HAND_JOINTS_CLOSED_DICT = {
    "right_zero_joint": 1.0, "right_one_joint": 1.0, "right_two_joint": 1.0, # Adjust these values
    "right_three_joint": 1.0, "right_four_joint": 1.0, "right_five_joint": 1.0,
    "right_six_joint": 1.0, # Adjust these values for desired grip
}

# Convert to ordered lists based on hand_joint_names defined later in ActionsCfg
# These can be used by TrajectoryPlayer or other components if needed.
G1_HAND_JOINTS_OPEN_ORDERED = [G1_HAND_JOINTS_OPEN_DICT.get(name, 0.0) for name in G1_RIGHT_HAND_JOINT_NAMES_ORDERED]
G1_HAND_JOINTS_CLOSED_ORDERED = [G1_HAND_JOINTS_CLOSED_DICT.get(name, 0.0) for name in G1_RIGHT_HAND_JOINT_NAMES_ORDERED]


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.55, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, 0.40, 1.0413), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg( # Using Cylinder as an example, replace with Cube or desired object
            radius=0.03, # Example for a small cube-like object
            height=0.06,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1), # Lighter mass
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.85), metallic=0.2), # Blueish
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )
    # Alternatively, for a Cube:
    # object = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, 0.40, 0.8), rot=(1.0, 0.0, 0.0, 0.0)), # Adjusted height for cube
    #     spawn=sim_utils.CubeCfg(
    #         size=(0.05, 0.05, 0.05), # 5cm cube
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15), metallic=0.2), # Reddish
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             friction_combine_mode="max",
    #             restitution_combine_mode="min",
    #             static_friction=0.9,
    #             dynamic_friction=0.9,
    #             restitution=0.0,
    #         ),
    #     ),
    # )


    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=G1_29DOF_CFG.spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74),
            rot=(0.7071, 0.0, 0.0, 0.7071), # Facing +y
            joint_pos={
                # Legs (default standing pose from G1_29DOF_CFG, can be adjusted)
                ".*_hip_pitch_joint": -0.20,
                ".*_knee_joint": 0.42,
                ".*_ankle_pitch_joint": -0.23,
                # Torso
                "torso_joint": 0.0,
                # Left arm (fixed, out of the way or neutral)
                "left_shoulder_pitch_joint": 0.5, # Example: slightly forward
                "left_shoulder_roll_joint": 0.3,  # Example: slightly abducted
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": 1.0,  # Example: bent elbow
                "left_elbow_roll_joint": 0.0,
                # Right arm (initial pose for picking, can be adjusted)
                "right_shoulder_pitch_joint": 0.45, # From G1_29DOF_CFG: 0.35
                "right_shoulder_roll_joint": -0.2, # From G1_29DOF_CFG: -0.16
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": 1.0,  # From G1_29DOF_CFG: 0.87
                "right_elbow_roll_joint": 0.0,
                # Hands (Open by default, using ordered values)
                **{name: pos for name, pos in zip(G1_RIGHT_HAND_JOINT_NAMES_ORDERED, G1_HAND_JOINTS_OPEN_ORDERED)},
                # Left hand joints (can be all zero if fixed open)
                "left_zero_joint": 0.0, "left_one_joint": 0.0, "left_two_joint": 0.0,
                "left_three_joint": 0.0, "left_four_joint": 0.0, "left_five_joint": 0.0,
                "left_six_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators=G1_29DOF_CFG.actuators,
        soft_joint_pos_limit_factor=G1_29DOF_CFG.soft_joint_pos_limit_factor,
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
        pink_controlled_joint_names=[ # G1 right arm joints
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_elbow_roll_joint",
        ],
        ik_urdf_fixed_joint_names=[ # Joints fixed in URDF for IK
            # Legs
            ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
            ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
            # Torso
            "torso_joint",
            # Left Arm (as it's not controlled by this IK setup)
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint", "left_elbow_roll_joint",
            # ALL Hand joints (both left and right)
            "left_five_joint", "left_three_joint", "left_six_joint", "left_four_joint",
            "left_zero_joint", "left_one_joint", "left_two_joint",
            "right_five_joint", "right_three_joint", "right_six_joint", "right_four_joint",
            "right_zero_joint", "right_one_joint", "right_two_joint",
        ],
        # Hand joint names for direct control (G1 right hand) - Order matters!
        hand_joint_names=G1_RIGHT_HAND_JOINT_NAMES_ORDERED,
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link", # Or "pelvis" if that's the robot's root for IK
            num_hand_joints=len(G1_RIGHT_HAND_JOINT_NAMES_ORDERED), # Should be 7
            show_ik_warnings=False,
            variable_input_tasks=[
                FrameTask( # Right palm link control
                    "right_palm_link",
                    position_cost=1.0,
                    orientation_cost=1.0,
                    lm_damping=10.0,
                    gain=0.1, # Lower gain for smoother movements, adjust as needed
                ),
            ],
            fixed_input_tasks=[
                # No fixed tasks for head or other parts for now
            ],
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        # Actions from previous step
        actions = ObsTerm(func=mdp.last_action)
        # Full robot joint state
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}) # Use relative for better generalization
        robot_joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}) # Use relative

        # Object state (relative to world or end-effector)
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")}) # XYZW
        object_lin_vel = ObsTerm(func=base_mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_ang_vel = ObsTerm(func=base_mdp.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("object")})

        # Right end-effector (palm) state
        right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos) # World frame
        right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat) # World frame, XYZW

        # Relative pose between EEF and Object
        eef_to_object_pos_rel = ObsTerm(func=mdp.eef_to_object_pos_relative) # Custom obs: object_pos - eef_pos in eef frame
        eef_to_object_rot_rel = ObsTerm(func=mdp.eef_to_object_rot_relative) # Custom obs: object_rot relative to eef_rot

        # Right hand joint state
        right_hand_joint_pos = ObsTerm(func=mdp.get_right_hand_joint_pos) # Only positions
        # right_hand_joint_vel = ObsTerm(func=mdp.get_right_hand_joint_vel) # Optional: velocities

        # Optional: Full robot link states (can be large)
        # robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)


        def __post_init__(self):
            self.enable_corruption = False # Typically False for teleop/testing
            self.concatenate_terms = True # Concatenate all observations into a single vector for the policy


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropped = DoneTerm(
        func=mdp.object_dropped, params={"object_cfg": SceneEntityCfg("object"), "ground_height_thresh": 0.75} # Adjusted for G1 table height
    )
    # success = DoneTerm(func=mdp.object_reached_target) # Define this in mdp.terminations


@configclass
class EventCfg:
    """Configuration for events."""
    # On reset, randomize object position slightly
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_ids=-1), # refers to the RigidObjectCfg
            "pose_range": {"x": [-0.1, 0.1], "y": [-0.1, 0.1], "z": [0.0, 0.0]}, # Relative to initial pose
            "velocity_range": {}, # No initial velocity
        },
    )
    # Could add robot joint randomization if needed for training
    # reset_robot_joints = EventTerm(func=mdp.reset_joints_by_offset, mode="reset", params={...})


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
    events: EventCfg = EventCfg()

    # Rewards - Add a RewardsCfg if you plan to train an RL agent
    # For teleoperation, rewards are not strictly necessary but can be useful for metrics.
    # rewards: RewardsCfg = RewardsCfg() # Define RewardsCfg and terms in mdp.rewards
    rewards = None
    # Unused managers for basic teleoperation
    commands = None # Not using curriculum commands for teleop
    # rewards = None # Rewards can be defined if metrics are desired
    curriculum = None

    # Position of the XR anchor in the world frame (if using XR, else default)
    xr: XrCfg = XrCfg(anchor_pos=(0.0, 0.0, 0.0), anchor_rot=(0.0, 0.0, 0.0, 1.0)) # w last for quat

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action for G1: [right_palm_pos (3), right_palm_quat_xyzw (4), right_hand_joint_pos (7)]
    # Example: robot looking forward, arm slightly forward, hand open
    # Calculate this based on a desired initial FK pose if possible, or use a safe known pose.
    # Assuming initial robot pose has right_palm_link at roughly (0.3, -0.2, 0.9) relative to robot base
    # and orientation is identity relative to robot base (which is rotated 90 deg yaw).
    # This needs to be in WORLD FRAME or relative to PINK base_link_name if that's how controller is set up.
    # For now, a placeholder based on initial joint config might be more robust to get from FK.
    # Let's use a simple example: slightly in front, hand open.
    # Pos: (0.4, -0.3, 0.9) in world (assuming robot at origin, facing +y)
    # Quat XYZW: (0, 0, 0, 1) for identity if world frame, or (0,0,sin(pi/4),cos(pi/4)) if matching robot base
    # For simplicity, let's use values that would keep the arm somewhat neutral.
    # The values from the prompt were: pos (0.22878, 0.2536, 1.0953)
    # quat_wxyz (0.5, 0.5, -0.5, 0.5) -> quat_xyzw (0.5, -0.5, 0.5, 0.5)
    idle_action_pos = [0.3, 0.3, 0.9] # Example world position in front of robot
    idle_action_quat_xyzw = [0.0, 0.0, 0.7071, 0.7071] # Example: Palm pointing down-forward
    idle_action_hand = G1_HAND_JOINTS_OPEN_ORDERED # Open hand

    idle_action = torch.tensor(idle_action_pos + idle_action_quat_xyzw + idle_action_hand)


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4 # Control at 60/4 = 15 Hz
        self.episode_length_s = 30.0 # Longer for pick-place
        # simulation settings
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation # Render at control rate

        # Convert USD to URDF and change revolute joints to fixed
        # Ensure G1_29DOF_CFG.spawn.usd_path is correct
        if G1_29DOF_CFG.spawn.usd_path is None:
            raise ValueError("G1_29DOF_CFG.spawn.usd_path is not defined. Cannot convert to URDF.")

        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            G1_29DOF_CFG.spawn.usd_path, self.temp_urdf_dir, force_conversion=True # Changed self.scene.robot.spawn.usd_path
        )
        ControllerUtils.change_revolute_to_fixed(
            temp_urdf_output_path, self.actions.pink_ik_cfg.ik_urdf_fixed_joint_names
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
        self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path

        # Print the path to the generated URDF file
        print(f"Generated G1 URDF path for PinkIK: {temp_urdf_output_path}")