
from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv , DirectMARLEnvCfg
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
class AntLegEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    state_space = 0
    action_scale = 1
    possible_agents = ["fl_leg" , "fr_leg" , "hl_leg" , "hr_leg"]
    action_spaces = {"fl_leg" : 2 ,
                     "fr_leg" : 2 ,
                     "hl_leg" : 2 ,
                     "hr_leg" : 2 , 
                    }
    # action spaces each leg : dof_effort


    observation_spaces = {"fl_leg" : 10 ,
                     "fr_leg" : 10 ,
                     "hl_leg" : 10 ,
                     "hr_leg" : 10 , 
                    }


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
    dof_per_leg = 2
    fl_joint_names = [ # inx 0 1
        "front_left_leg" ,
        "front_left_foot" ,
    ]
    fr_joint_names = [ # inx 2 3
        "front_right_leg" ,
        "front_right_foot" ,
    ]
    hl_joint_names = [ # inx 4 5
        "left_back_leg" ,
        "left_back_foot" ,
    ]
    hr_joint_names = [ # inx 6 7
        "right_back_leg" ,
        "right_back_foot" ,
    ]

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

class AntLegEnv(DirectMARLEnv):
    cfg : AntLegEnvCfg

    def __init__(self, cfg: AntLegEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # robot properties
        self.action_scale = self.cfg.action_scale
        self.num_dofs = self.cfg.dof_per_leg
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)

        self.fl_indices = list()
        self.fr_indices = list()
        self.hl_indices = list()
        self.hr_indices = list()

        # legged joint index
        for dof in range(self.num_dofs):
            self.fl_indices.append(self.robot.joint_names.index(self.cfg.fl_joint_names[dof]))
            self.fr_indices.append(self.robot.joint_names.index(self.cfg.fr_joint_names[dof]))
            self.hl_indices.append(self.robot.joint_names.index(self.cfg.hl_joint_names[dof]))
            self.hr_indices.append(self.robot.joint_names.index(self.cfg.hr_joint_names[dof]))

        # All joint index
        self._joint_dof_idx, _ = self.robot.find_joints(".*")

        # buffers for position targets
        self.front_left_target = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.front_right_target = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.hind_left_target = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.front_left_target = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.hind_right_target = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        
        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
    
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

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions
        # print(actions)


    def _apply_action(self):
        fl = self.actions["fl_leg"]
        fr = self.actions["fr_leg"]
        hl = self.actions["hl_leg"]
        hr = self.actions["hr_leg"]

        actions = torch.zeros((self.num_envs,len(self.joint_gears)), device=self.device, dtype=fl.dtype)

        actions[:, self.fl_indices] = fl
        actions[:, self.fr_indices] = fr
        actions[:, self.hl_indices] = hl
        actions[:, self.hr_indices] = hr
        forces = self.action_scale * self.joint_gears * actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)
    
    def _get_observations(self) -> dict[str, torch.Tensor]:
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
        # print(self.robot.data.root_lin_vel_b[:,:2].shape)
        # print(self.robot.data.root_ang_vel_b[:, 2:3].shape)
        # print(self._commands.shape)
        # print(self.robot.data.joint_pos[: , self.fl_indices].shape)
        # print(self.robot.data.joint_vel[: , self.fl_indices].shape)

        observations = {
            "fl_leg" : torch.cat(
                (
                    ## -- Global -- ##
                    self.robot.data.root_lin_vel_b[:,:2] ,
                    self.robot.data.root_ang_vel_b[:, 2:3] ,
                    self._commands,
                    ## -- Local -- ##
                    self.robot.data.joint_pos[: , self.fl_indices] ,
                    self.robot.data.joint_vel[: , self.fl_indices] ,
                ),
                dim=-1                
            ),
            "fr_leg" : torch.cat(
                (
                    ## -- Global -- ##
                    self.robot.data.root_lin_vel_b[:,:2] ,
                    self.robot.data.root_ang_vel_b[:, 2:3] ,
                    self._commands,
                    ## -- Local -- ##
                    self.robot.data.joint_pos[: , self.fr_indices] ,
                    self.robot.data.joint_vel[: , self.fr_indices] ,
                ),
                dim=-1                
            ),
            "hl_leg" : torch.cat(
                (
                    ## -- Global -- ##
                    self.robot.data.root_lin_vel_b[:,:2] ,
                    self.robot.data.root_ang_vel_b[:, 2:3] ,
                    self._commands,
                    ## -- Local -- ##
                    self.robot.data.joint_pos[: , self.hl_indices] ,
                    self.robot.data.joint_vel[: , self.hl_indices] ,
                ),
                dim=-1                
            ),
            "hr_leg" : torch.cat(
                (
                    ## -- Global -- ##
                    self.robot.data.root_lin_vel_b[:,:2] ,
                    self.robot.data.root_ang_vel_b[:, 2:3] ,
                    self._commands,
                    ## -- Local -- ##
                    self.robot.data.joint_pos[: , self.hr_indices] ,
                    self.robot.data.joint_vel[: , self.hr_indices] ,
                ),
                dim=-1                
            ),
        }
        return observations

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:        
        self._compute_intermediate_values()
        truncate = self.episode_length_buf >= self.max_episode_length - 1
        terminate = self.torso_position[:, 2] < self.cfg.termination_height

        terminates = {agent : terminate for agent in self.cfg.possible_agents}
        truncates = {agent : truncate for agent in self.cfg.possible_agents}

        return terminates, truncates
    
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

        # clear out any old actions for those envs:
        for leg_name, act_tensor in self.actions.items():
            # zero only the rows corresponding to env_ids
            self.actions[leg_name] = torch.zeros(self.num_envs , self.num_dofs)

        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0) # Curriculum add here

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rew_global = torch.tensor(0)

        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        yaw_error = torch.square(self._commands[:, 2] - self.robot.data.heading_w)
        yaw_error_mapped = torch.exp(-yaw_error / 0.25)

        rew_global = lin_vel_error_mapped * self.cfg.tracking_lin_vel_weight + yaw_error_mapped * self.cfg.yaw_weight
        return {
            "fl_leg" : rew_global , 
            "fr_leg" : rew_global , 
            "hl_leg" : rew_global , 
            "hr_leg" : rew_global , 
            }
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