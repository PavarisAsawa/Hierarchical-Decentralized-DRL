
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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns , ContactSensor

from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from isaaclab.markers import VisualizationMarkersCfg , VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG , POSITION_GOAL_MARKER_CFG
import isaacsim.core.utils.torch as torch_utils
# from .decentralized.ant_decentral_env import AntLegEnvCfg

# LOW_LEVEL = AntLegEnvCfg()
@configclass
class AntHighLevelCfg(DirectRLEnvCfg):
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
    # Set Marker for dedug
    current_vel_visualizer: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
    goal_pos_visualizer_cfg: VisualizationMarkersCfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/Command/position_goal")


    current_vel_visualizer.markers["arrow"].scale = (0.5, 0.5, 0.5)
    goal_pos_visualizer_cfg.markers["target_near"].scale = (0.5, 0.5, 0.5)

    # robot
    robot: ArticulationCfg = ANT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # contact_senspr : ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/.*_foot", history_length=3, update_period=0.005, track_air_time=True
    # )
    contact_force = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_foot",
        update_period=0.0,
        history_length=6,
        track_air_time = True
    )
    # joint_gears: list = [15, 15, 15, 15, 15, 15, 15, 15]
    joint_gears: list = [1, 1, 1, 1, 1, 1, 1, 1]

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


class AntHighLevelEnv(DirectRLEnv):
    cfg : AntHighLevelCfg

    def __init__(self, cfg: AntHighLevelCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # robot properties
        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self.pos_env = torch.zeros(self.num_envs , 2 , dtype=torch.float32, device=self.sim.device) # 2 for pos
        self.heading = torch.zeros(self.num_envs , dtype=torch.float32, device=self.sim.device)
        # All joint index
        self._joint_dof_idx, _ = self.robot.find_joints(".*")

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 2, device=self.sim.device)

        # set debug
        self.has_debug_vis_implementation = True
        self.set_debug_vis(True)

        # self._episode_sums = {
        #     key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        #     for key in [
                # "target",
        #     ]
        # }


    
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
        # Add contact sensor
        self.contact_sensor = ContactSensor(self.cfg.contact_force)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


        # for dof in range(self.cfg.dof_per_leg):
        #     self.fl_indices.append(self.robot.joint_names.index(self.cfg.fl_joint_names[dof]))
        #     self.fr_indices.append(self.robot.joint_names.index(self.cfg.fr_joint_names[dof]))
        #     self.hl_indices.append(self.robot.joint_names.index(self.cfg.hl_joint_names[dof]))
        #     self.hr_indices.append(self.robot.joint_names.index(self.cfg.hr_joint_names[dof]))


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
        foot_status = self._get_foot_status()  # [: , leg]  leg : fl , fr , hl ,hr

        # get position in env_frame
        pos_env = self.pos_env - self.robot.data.default_root_state
        observations =  torch.cat(
                (
                    # High Level Only :)
                    pos_env ,           # inx 0-1
                    self.heading ,      # inx 2
                    self._commands ,    # 3-5
                    # Low level
                    self.robot.data.root_lin_vel_b[:,:2] ,
                    self.robot.data.projected_gravity_b ,
                    foot_status,
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
        
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-5, 5) # pos env + uniform

        # to_target = self.targets[env_ids] - default_root_state[:, :3]
        # to_target[:, 2] = 0.0
        # self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        self._compute_intermediate_values()

        # Logging
        # extras = dict()
        # for key in self._episode_sums.keys():
        #     episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
        #     extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length
        #     self._episode_sums[key][env_ids] = 0.0
        # self.extras["log"] = dict()
        # self.extras["log"].update(extras)
        # extras = dict()
        # extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        # extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        # self.extras["log"].update(extras)

    def _get_rewards(self) -> torch.Tensor:
        # ------------------- Global ------------------- #
        # Target Error
        target = torch.norm(self._commands - self.pos_env , dim=1)
        target_reward = 1-torch.tanh(target / 1.5)
        rewards = {
            "target": target_reward,
        }
        # Logging
        # for key, value in rewards.items():
        #     self._episode_sums[key] += value
        rew = torch.zeros(self.num_envs)
        return rew
    
    def _get_foot_status(self):
        f = self.scene["contact_sensor"].data.net_forces_w  # shape (num_envs, 4, 3)
        # compute per-foot norm
        foot_force_norm = torch.norm(f, dim=-1)            # (num_envs, 4)
        # threshold at 1.0
        foot_status = (foot_force_norm > 1.0).float()      # (num_envs, 4)
        return foot_status
    
    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel
        self.dof_pos_scaled = torch_utils.maths.unscale(self.dof_pos, self.robot.data.soft_joint_pos_limits[0, :, 0],self.robot.data.soft_joint_pos_limits[0, :, 1])

        self.pos_env = self.robot.data.root_state_w[:,:2] - self.scene.env_origins[:,:2]
        self.heading = self.robot.data.heading_w


    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
        # Init marker
            self.goal_pos_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
            self.out_vel_visualizer = VisualizationMarkers(self.cfg.out_vel_visualizer_cfg)
            self.goal_pos_visualizer.set_visibility(True)
            self.out_vel_visualizer.set_visibility(True)
        else:
            self.out_vel_visualizer.set_visibility(False)
            self.goal_pos_visualizer.set_visibility(False)

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
        target_pos_w = self.scene.env_origins[:,:2] + self._commands
        target_pos_w[:, 2] += 0.5
        # Blue
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        
        # # display markers
        # self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        self.goal_pos_visualizer.visualize(target_pos_w, vel_arrow_quat, 0.5)

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

    