
from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv , DirectMARLEnvCfg

from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclab.terrains import TerrainImporterCfg

from isaaclab.scene import InteractiveSceneCfg

from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.ant import ANT_CFG
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns , ContactSensor

from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
import isaacsim.core.utils.torch as torch_utils

from isaaclab.markers import VisualizationMarkersCfg , VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


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
        collision_group=0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Set Marker for dedug
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ANT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot" , 
        debug_vis=True
        )
    # contact_senspr : ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/.*_foot", history_length=3, update_period=0.005, track_air_time=True
    # )

    # joint_gears: list = [15, 15, 15, 15, 15, 15, 15, 15]
    joint_gears: list = [1, 1, 1, 1, 1, 1, 1, 1]
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

    contact_force = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_foot",
        update_period=0.0,
        history_length=6,
        track_air_time = True
    )


    # Reward Function
    tracking_lin_vel_weight: float = 3.0
    yaw_weight: float = 0.0
    torque_weight : float = 0.0 #-1.25e-4
    roll_weight : float = -0.1
    pitch_weight : float = -0.1
    up_weight: float = -0.3
    dof_vel_scale: float = 0.2

    energy_cost_scale: float = 0.00025
    actions_cost_scale: float = 0.0001
    alive_reward_scale: float = 0.5
    dof_at_limit_scale: float =0.01

    death_cost: float = -2.0    
    termination_height: float = 0.36
    termination_height_up : float = 1.0

    angular_velocity_scale: float = 1.0
    contact_force_scale: float = 0.1

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
        
        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 2, device=self.sim.device)
        
        # Set value
        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        
        # set debug
        self.has_debug_vis_implementation = True
        self.set_debug_vis(True)\
        
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "up",
                "alive",
                "action_fl",
                "action_fr",
                "action_hl",
                "action_hr",
                "electricity_fl",
                "electricity_fr",
                "electricity_hl",
                "electricity_hr",
                "dof_at_limit_fl",
                "dof_at_limit_fr",
                "dof_at_limit_hl",
                "dof_at_limit_hr",
                "global_reward",
                "fl_local",
                "fr_local",
                "hl_local",
                "hr_local"
            ]
        }

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # Add contact sensor
        self.contact_sensor = ContactSensor(self.cfg.contact_force)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel
        self.dof_pos_scaled = torch_utils.maths.unscale(self.dof_pos, self.robot.data.soft_joint_pos_limits[0, :, 0],self.robot.data.soft_joint_pos_limits[0, :, 1])

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions
        # print(actions)


    def _apply_action(self)-> None:
        fl = self.actions["fl_leg"]
        fr = self.actions["fr_leg"]
        hl = self.actions["hl_leg"]
        hr = self.actions["hr_leg"]
        actions = torch.zeros((self.num_envs,len(self.joint_gears)), device=self.sim.device, dtype=torch.float32)
        actions[:, self.fl_indices] = fl
        actions[:, self.fr_indices] = fr
        actions[:, self.hl_indices] = hl
        actions[:, self.hr_indices] = hr
        forces = self.action_scale * self.joint_gears * actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

        # print(self.robot._data.computed_torque)

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
        foot_status = self._get_foot_status()  # [: , leg]  leg : fl , fr , hl ,hr

        observations = {
            "fl_leg" : torch.cat(
                (
                    ## -- Global -- ##
                    self.robot.data.root_lin_vel_b[:,:2] ,
                    self.robot.data.projected_gravity_b,
                    self._commands,
                    foot_status,
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
                    self.robot.data.projected_gravity_b,
                    self._commands,
                    foot_status,
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
                    self.robot.data.projected_gravity_b,
                    self._commands,
                    foot_status,
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
                    self.robot.data.projected_gravity_b,
                    self._commands,
                    foot_status,
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
        terminate = torch.logical_or(self.torso_position[:, 2] < self.cfg.termination_height , self.torso_position[:, 2] > self.cfg.termination_height_up)

        terminates = {agent : terminate for agent in self.cfg.possible_agents}
        truncates = {agent : truncate for agent in self.cfg.possible_agents}
        # print(terminate)
        # print(truncate)
        # print('*'*30)

        return terminates, truncates
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # for leg in self.actions:
        #     self.actions[leg][env_ids] = 0.0
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # clear out any old actions for those envs:
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.5, 1.5) # Curriculum add here
        self._compute_intermediate_values()

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # ------------------- Global ------------------- #
        # Lin vel Error 
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25) * self.cfg.tracking_lin_vel_weight

        # Command Error Yaw
        # yaw_error = torch.square(self._commands[:, 2] - self.robot.data.heading_w)
        # yaw_error_mapped = torch.exp(-yaw_error / 0.25) * self.cfg.yaw_weight

        # Orientation Term
        # euler = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        # roll_reward = torch.abs(euler[0]) * self.cfg.roll_weight
        # pitch_reward = torch.abs(euler[1]) * self.cfg.pitch_weight

        # Upreward
        up_reward = torch.zeros(self.num_envs , device=self.sim.device)
        up_reward = torch.where(self.robot.data.projected_gravity_b[: , 2] > -0.9, torch.abs(self.robot.data.projected_gravity_b[: , 2]) * self.cfg.up_weight, up_reward)

        # Alivy
        alive_reward = torch.ones(self.num_envs ,device=self.sim.device) * self.cfg.alive_reward_scale * 0.0


        # rew_global = torch.tensor(0)
        rew_global = lin_vel_error_mapped  + up_reward  + alive_reward
        # ------------------- Local ------------------- #
        fl_joint_torques = torch.sum(torch.square(self.robot.data.applied_torque[:,self.fl_indices]), dim=1) * self.cfg.torque_weight
        fr_joint_torques = torch.sum(torch.square(self.robot.data.applied_torque[:,self.fr_indices]), dim=1) * self.cfg.torque_weight
        hl_joint_torques = torch.sum(torch.square(self.robot.data.applied_torque[:,self.hl_indices]), dim=1) * self.cfg.torque_weight
        hr_joint_torques = torch.sum(torch.square(self.robot.data.applied_torque[:,self.hr_indices]), dim=1) * self.cfg.torque_weight

        fl_action = torch.sum(torch.square(self.actions["fl_leg"]), dim=1) * self.cfg.actions_cost_scale
        fr_action = torch.sum(torch.square(self.actions["fr_leg"]), dim=1) * self.cfg.actions_cost_scale
        hl_action = torch.sum(torch.square(self.actions["hl_leg"]), dim=1) * self.cfg.actions_cost_scale
        hr_action = torch.sum(torch.square(self.actions["hr_leg"]), dim=1) * self.cfg.actions_cost_scale

        # energy penalty for movement
        # print(f"pokpokpokpokpokpokpokpokpok :{self.robot.data.projected_gravity_b[0 , 2]}")
        fl_electricity_cost = torch.sum(torch.abs(self.actions["fl_leg"] * self.dof_vel[: , self.fl_indices] * self.cfg.dof_vel_scale) * self.motor_effort_ratio[:2].unsqueeze(0),dim=-1,) * self.cfg.energy_cost_scale
        fr_electricity_cost = torch.sum(torch.abs(self.actions["fr_leg"] * self.dof_vel[: , self.fr_indices] * self.cfg.dof_vel_scale) * self.motor_effort_ratio[:2].unsqueeze(0),dim=-1,) * self.cfg.energy_cost_scale
        hl_electricity_cost = torch.sum(torch.abs(self.actions["hl_leg"] * self.dof_vel[: , self.hl_indices] * self.cfg.dof_vel_scale) * self.motor_effort_ratio[:2].unsqueeze(0),dim=-1,) * self.cfg.energy_cost_scale
        hr_electricity_cost = torch.sum(torch.abs(self.actions["hr_leg"] * self.dof_vel[: , self.hr_indices] * self.cfg.dof_vel_scale) * self.motor_effort_ratio[:2].unsqueeze(0),dim=-1,) * self.cfg.energy_cost_scale

        fl_action = torch.where(torch.all(self.actions["fl_leg"] > 1.0, dim=1), torch.ones_like(rew_global) * self.cfg.actions_cost_scale, 0)
        fr_action = torch.where(torch.all(self.actions["fr_leg"] > 1.0, dim=1), torch.ones_like(rew_global) * self.cfg.actions_cost_scale, 0)
        hl_action = torch.where(torch.all(self.actions["hl_leg"] > 1.0, dim=1), torch.ones_like(rew_global) * self.cfg.actions_cost_scale, 0)
        hr_action = torch.where(torch.all(self.actions["hr_leg"] > 1.0, dim=1), torch.ones_like(rew_global) * self.cfg.actions_cost_scale, 0)
        # dof at limit cost
        fl_dof_at_limit_cost = torch.sum(self.dof_pos_scaled[: , self.fl_indices] > 0.98, dim=-1) * self.cfg.dof_at_limit_scale
        fr_dof_at_limit_cost = torch.sum(self.dof_pos_scaled[: , self.fr_indices] > 0.98, dim=-1) * self.cfg.dof_at_limit_scale
        hl_dof_at_limit_cost = torch.sum(self.dof_pos_scaled[: , self.hl_indices] > 0.98, dim=-1) * self.cfg.dof_at_limit_scale
        hr_dof_at_limit_cost = torch.sum(self.dof_pos_scaled[: , self.hr_indices] > 0.98, dim=-1) * self.cfg.dof_at_limit_scale

        fl_local = fl_action + fl_electricity_cost + fl_dof_at_limit_cost
        fr_local = fr_action + fr_electricity_cost + fr_dof_at_limit_cost
        hl_local = hl_action + hl_electricity_cost + hl_dof_at_limit_cost
        hr_local = hr_action + hr_electricity_cost + hr_dof_at_limit_cost

        # ------------------- All ------------------- #
        fl_rew = torch.where(self.reset_buf, torch.ones_like(rew_global) * self.cfg.death_cost, rew_global + fl_local)
        fr_rew = torch.where(self.reset_buf, torch.ones_like(rew_global) * self.cfg.death_cost, rew_global + fr_local)
        hl_rew = torch.where(self.reset_buf, torch.ones_like(rew_global) * self.cfg.death_cost, rew_global + hl_local)
        hr_rew = torch.where(self.reset_buf, torch.ones_like(rew_global) * self.cfg.death_cost, rew_global + hr_local)

        rewards = {
            "lin_vel": lin_vel_error_mapped,
            "up": up_reward,
            "alive": alive_reward,
            "action_fl": fl_action,
            "action_fr": fr_action,
            "action_hl": hl_action,
            "action_hr": hr_action,
            "electricity_fl": fl_electricity_cost,
            "electricity_fr": fr_electricity_cost,
            "electricity_hl": hl_electricity_cost,
            "electricity_hr": hr_electricity_cost,
            "dof_at_limit_fl": fl_dof_at_limit_cost,
            "dof_at_limit_fr": fr_dof_at_limit_cost,
            "dof_at_limit_hl": hl_dof_at_limit_cost,
            "dof_at_limit_hr": hr_dof_at_limit_cost,
            "global_reward": rew_global,
            "fl_local": fl_local,
            "fr_local": fr_local,
            "hl_local": hl_local,
            "hr_local": hr_local
        }
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return {
            "fl_leg" : fl_rew, 
            "fr_leg" : fr_rew, 
            "hl_leg" : hl_rew, 
            "hr_leg" : hr_rew, 
            }
    
    def _get_foot_status(self):
        f = self.scene["contact_sensor"].data.net_forces_w  # shape (num_envs, 4, 3)
        # compute per-foot norm
        foot_force_norm = torch.norm(f, dim=-1)            # (num_envs, 4)
        # threshold at 1.0
        foot_status = (foot_force_norm > 1.0).float()      # (num_envs, 4)
        return foot_status


    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
        # Init marker
            self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
            self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            self.goal_pos_visualizer.set_visibility(False)

    # def _debug_vis_callback(self, event):
    #     # update the markers
    #     self.goal_pos_visualizer.visualize(self._desired_pos_w)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        # print("pokpokpokpokpokpokpokpok")

        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        # Green
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        # Blue
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # # display markers
        # self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        # self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
    