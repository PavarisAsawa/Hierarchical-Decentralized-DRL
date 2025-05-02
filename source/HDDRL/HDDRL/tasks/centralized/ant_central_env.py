
from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv , DirectMARLEnvCfg , DirectRLEnvCfg , DirectRLEnv
from isaaclab.markers import VisualizationMarkers

from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclab.terrains import TerrainImporterCfg

from isaaclab.scene import InteractiveSceneCfg

from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.ant import ANT_CFG
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.utils import configclass

from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
import isaacsim.core.utils.torch as torch_utils

@configclass
class AntCentralEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    state_space = 0
    action_scale = 1
    # action spaces each leg : dof_effort

    action_space = 8
    observation_space = 10


    # observation each leg -> dof_pos , dos_vel , dof_torque , contact_force 


    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ANT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # contact_senspr : ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/.*_foot", history_length=3, update_period=0.005, track_air_time=True
    # )

    joint_gears: list = [15, 15, 15, 15, 15, 15, 15, 15]
    # joint_gears: list = [1, 1, 1, 1, 1, 1, 1, 1]

    # Reward Function
    tracking_lin_vel_weight: float = 1
    yaw_weight: float = 1


    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.005
    alive_reward_scale: float = 0.5
    dof_vel_scale: float = 0.2

    death_cost: float = -2.0
    termination_height: float = 0.31

    # angular_velocity_scale: float = 1.0
    # contact_force_scale: float = 0.1

class AntCentralEnv(DirectRLEnv):
    cfg : AntCentralEnvCfg

    def __init__(self, cfg: AntCentralEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # robot properties
        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        # All joint index
        self._joint_dof_idx, _ = self.robot.find_joints(".*")

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.sim.device)
    
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


        # for dof in range(self.cfg.dof_per_leg):
        #     self.fl_indices.append(self.robot.joint_names.index(self.cfg.fl_joint_names[dof]))
        #     self.fr_indices.append(self.robot.joint_names.index(self.cfg.fr_joint_names[dof]))
        #     self.hl_indices.append(self.robot.joint_names.index(self.cfg.hl_joint_names[dof]))
        #     self.hr_indices.append(self.robot.joint_names.index(self.cfg.hr_joint_names[dof]))

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # print(actions)


    def _apply_action(self)-> None:
        forces = self.action_scale * self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)
        # print(self.robot._data.computed_torque)

    def _get_observations(self) -> dict:
        '''
        -- Global -- 
        Linear Vel : Robot XY in base (2)
        Yaw : Robot Yaw in base (1)
        HLC -> Command XY/Yaw (3)
        > 6

        -- Local -- 
        Dof Position (2)
        Dof Vel (2)
        > 4
        >> 10
        '''
        observations =  torch.cat(
                (
                    ## -- Global -- ##
                    self.robot.data.root_lin_vel_b[:,:2] ,
                    self.robot.data.root_ang_vel_b[:, 2:3] ,
                    self._commands,
                    ## -- Local -- ##
                    self.robot.data.joint_pos[: , :] ,
                    self.robot.data.joint_vel[: , :] ,
                ),
                dim=-1                   
            )
        obs = {"policy" : observations}
        return obs

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:        
        self._compute_intermediate_values()
        truncate = self.episode_length_buf >= self.max_episode_length - 1
        terminate = self.torso_position[:, 2] < self.cfg.termination_height
        # print(terminate)
        # print(truncate)
        # print('*'*30)

        return terminate, truncate
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # to_target = self.targets[env_ids] - default_root_state[:, :3]
        # to_target[:, 2] = 0.0
        # self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        self._compute_intermediate_values()

    def _get_rewards(self) -> torch.Tensor:
        rew_global = torch.tensor(0)

        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        yaw_error = torch.square(self._commands[:, 2] - self.robot.data.heading_w)
        yaw_error_mapped = torch.exp(-yaw_error / 0.25)

        rew_global = lin_vel_error_mapped * self.cfg.tracking_lin_vel_weight + yaw_error_mapped * self.cfg.yaw_weight
        return rew_global
    
# @torch.jit.script
# def compute_intermediate_values(
#     targets: torch.Tensor,
#     torso_position: torch.Tensor,
#     torso_rotation: torch.Tensor,
#     velocity: torch.Tensor,
#     ang_velocity: torch.Tensor,
#     dof_pos: torch.Tensor,
#     dof_lower_limits: torch.Tensor,
#     dof_upper_limits: torch.Tensor,
#     inv_start_rot: torch.Tensor,
#     basis_vec0: torch.Tensor,
#     basis_vec1: torch.Tensor,
#     potentials: torch.Tensor,
#     prev_potentials: torch.Tensor,
#     dt: float,
# ):
#     to_target = targets - torso_position
#     to_target[:, 2] = 0.0

#     torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
#         torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
#     )

#     vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
#         torso_quat, velocity, ang_velocity, targets, torso_position
#     )

#     dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

#     to_target = targets - torso_position
#     to_target[:, 2] = 0.0
#     prev_potentials[:] = potentials
#     potentials = -torch.norm(to_target, p=2, dim=-1) / dt

#     return (
#         up_proj,
#         heading_proj,
#         up_vec,
#         heading_vec,
#         vel_loc,
#         angvel_loc,
#         roll,
#         pitch,
#         yaw,
#         angle_to_target,
#         dof_pos_scaled,
#         prev_potentials,
#         potentials,
#     )