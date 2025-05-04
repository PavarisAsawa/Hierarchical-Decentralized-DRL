# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import copy
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.utils import store_code_state


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # resolve dimensions of observations
        obs, extras = self.env.reset()
        num_obs = {}
        for agent in self.env.agent_names:
            num_obs[agent] = obs[agent].shape[1]

        # resolve type of privileged observations
        # if self.training_type == "rl":
        #     if "critic" in extras["observations"]:
        #         self.privileged_obs_type = "critic"  # actor-critic reinforcement learnig, e.g., PPO
        #     else:
        #         self.privileged_obs_type = None
        # if self.training_type == "distillation":
        #     if "teacher" in extras["observations"]:
        #         self.privileged_obs_type = "teacher"  # policy distillation
        #     else:
        #         self.privileged_obs_type = None

        # # resolve dimensions of privileged observations
        # if self.privileged_obs_type is not None:
        #     num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        # else:
        self.privileged_obs_type = None
        num_privileged_obs = num_obs

        self.alg = {}  # Dict to hold one PPO per agent
        self.policies = {}
        policy_class = eval(self.policy_cfg["class_name"])
        alg_class = eval(self.alg_cfg["class_name"])
        for agent in self.env.agent_names:
            policy_cfg_copy = copy.deepcopy(self.policy_cfg)
            policy_cfg_copy.pop("class_name", None)
            alg_cfg_copy = copy.deepcopy(self.alg_cfg)
            alg_cfg_copy.pop("class_name", None)
            policy: ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = policy_class(
                num_obs[agent], num_privileged_obs[agent], self.env.num_actions[agent], **policy_cfg_copy
            ).to(self.device)
            self.policies[agent] = policy

            # # resolve dimension of rnd gated state
            # if "rnd_cfg" in alg_cfg_copy and alg_cfg_copy["rnd_cfg"] is not None:
            #     # check if rnd gated state is present
            #     rnd_state = extras["observations"].get("rnd_state", {}).get(agent, None)
            #     if rnd_state is None:
            #         raise ValueError(f"RND state missing for agent {agent}.")
            #     alg_cfg_copy["rnd_cfg"]["num_states"] = rnd_state.shape[1]
            #     alg_cfg_copy["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

            # if using symmetry then pass the environment config object
            if "symmetry_cfg" in alg_cfg_copy and alg_cfg_copy["symmetry_cfg"] is not None:
                # this is used by the symmetry function for handling different observation terms
                alg_cfg_copy["symmetry_cfg"]["_env"] = env

            # initialize algorithm
            alg: PPO | Distillation = alg_class(policy, device=self.device, **alg_cfg_copy, multi_gpu_cfg=self.multi_gpu_cfg)
            self.alg[agent] = alg


        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        self.obs_normalizer = {}
        self.privileged_obs_normalizer = {}

        for agent in self.env.agent_names:
            if self.empirical_normalization:
                self.obs_normalizer[agent] = EmpiricalNormalization(shape=[num_obs[agent]], until=1.0e8).to(self.device)
                self.privileged_obs_normalizer[agent] = EmpiricalNormalization(shape=[num_privileged_obs[agent]], until=1.0e8).to(self.device)
            else:
                self.obs_normalizer[agent] = torch.nn.Identity().to(self.device)
                self.privileged_obs_normalizer[agent] = torch.nn.Identity().to(self.device)

        # init storage and model
        for agent in self.env.agent_names:
            self.alg[agent].init_storage(
                self.training_type,
                self.env.num_envs,
                self.num_steps_per_env,
                [num_obs[agent]],
                [num_privileged_obs[agent]],
                [self.env.num_actions[agent]],
            )

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # check if teacher is loaded
        if self.training_type == "distillation":
            for agent, alg in self.alg.items():
                if not alg.policy.loaded_teacher:
                    raise ValueError(f"Teacher model parameters not loaded for agent '{agent}'. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.reset()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        for agent in self.env.agent_names:
            obs[agent] = obs[agent].to(self.device)
            privileged_obs[agent] = privileged_obs[agent].to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # # create buffers for logging extrinsic and intrinsic rewards
        # if any(self.alg[a].rnd for a in self.env.agent_names):
        #     erewbuffer = deque(maxlen=100)
        #     irewbuffer = deque(maxlen=100)
        #     cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        #     cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        # if self.is_distributed:
        #     print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
        #     self.alg.broadcast_parameters()
        #     # TODO: Do we need to synchronize empirical normalizers?
        #     #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = {}
                    for agent in self.env.agent_names:
                        actions[agent] = self.alg[agent].act(obs[agent], privileged_obs[agent]).to(self.env.device)
                    # Step the environment
                    obs, rewards, dones, infos = self.env.step(actions)
                    # Move to device
                    for agent in self.env.agent_names:
                        obs[agent] = obs[agent].to(self.device)
                        rewards[agent] = rewards[agent].to(self.device)
                        dones[agent] = dones[agent].to(self.device)
                        # perform normalization
                        obs[agent] = self.obs_normalizer[agent](obs[agent])
                        if self.privileged_obs_type is not None:
                            privileged_obs[agent] = self.privileged_obs_normalizer[agent](
                                infos["observations"][self.privileged_obs_type][agent].to(self.device)
                            )
                        else:
                            privileged_obs[agent] = obs[agent]

                        # process the step
                        agent_info = {k: v[agent] for k, v in infos.items() if agent in v}
                        self.alg[agent].process_env_step(rewards[agent], dones[agent], agent_info)

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        cur_reward_sum += torch.stack([rewards[a] for a in self.env.agent_names]).sum(dim=0)
                        # if self.alg.rnd:
                        #     cur_ereward_sum += rewards
                        #     cur_ireward_sum += intrinsic_rewards  # type: ignore
                        #     cur_reward_sum += rewards + intrinsic_rewards
                        # else:
                        #     cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (torch.stack([dones[a] for a in self.env.agent_names]).any(dim=0)).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        # -- intrinsic and extrinsic rewards
                        # if self.alg.rnd:
                        #     erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        #     irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        #     cur_ereward_sum[new_ids] = 0
                        #     cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                if self.training_type == "rl":
                    for agent in self.env.agent_names:
                        self.alg[agent].compute_returns(privileged_obs[agent])

            # update policy
            loss_dict = {agent: self.alg[agent].update() for agent in self.env.agent_names}

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    val = ep_info[key]
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor([val], device=self.device)
                    if val.ndim == 0:
                        val = val.unsqueeze(0)
                    infotensor = torch.cat((infotensor, val.to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        for agent in self.env.agent_names:
            policy = self.alg[agent].policy
            mean_std = policy.action_std.mean()
            # -- Losses
            for key, value in locs["loss_dict"][agent].items():
                self.writer.add_scalar(f"{agent}/Loss/{key}", value, locs["it"])
            self.writer.add_scalar(f"{agent}/Loss/learning_rate", self.alg[agent].learning_rate, locs["it"])

            # -- Policy
            self.writer.add_scalar(f"{agent}/Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            # if isinstance(self.alg, dict) and any(self.alg[agent].rnd for agent in self.env.agent_names):
            #     for agent in self.env.agent_names:
            #         if self.alg[agent].rnd:
            #             self.writer.add_scalar(f"{agent}/Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
            #             self.writer.add_scalar(f"{agent}/Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
            #             self.writer.add_scalar(f"{agent}/Rnd/weight", self.alg[agent].rnd.weight, locs["it"])
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        log_string = f"{'#' * width}\n{str.center(width, ' ')}\n\n"
        log_string += f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"

        for agent in self.env.agent_names:
            mean_std = self.alg[agent].policy.action_std.mean()
            log_string += f"{agent + ' noise std:':>{pad}} {mean_std.item():.2f}\n"
            # -- Losses
            for key, value in locs["loss_dict"][agent].items():
                log_string += f"{f'{agent} {key} loss:':>{pad}} {value:.4f}\n"
        if len(locs["rewbuffer"]) > 0:
            # -- Rewards)
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                            locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "iter": self.current_learning_iteration,
            "infos": infos,
            "agent_data": {},
        }
        for agent in self.env.agent_names:
            agent_data = {
                "model_state_dict": self.alg[agent].policy.state_dict(),
                "optimizer_state_dict": self.alg[agent].optimizer.state_dict()
            }
            # # -- Save RND model if used
            # if self.alg[agent].rnd:
            #     agent_data["rnd_state_dict"] = self.alg[agent].rnd.state_dict()
            #     agent_data["rnd_optimizer_state_dict"] = self.alg[agent].rnd_optimizer.state_dict()
            # -- Save observation normalizer if used
            if self.empirical_normalization:
                agent_data["obs_norm_state_dict"] = self.obs_normalizer[agent].state_dict()
                agent_data["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer[agent].state_dict()

            saved_dict["agent_data"][agent] = agent_data
        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        agent_data = loaded_dict["agent_data"]
        # -- Load model
        for agent in self.env.agent_names:
            resumed_training = self.alg[agent].policy.load_state_dict(agent_data[agent]["model_state_dict"])
            # # -- Load RND model if used
            # if self.alg[agent].rnd and "rnd_state_dict" in agent_data[agent]:
            #     self.alg[agent].rnd.load_state_dict(agent_data[agent]["rnd_state_dict"])
            # -- Load observation normalizer if used
            if self.empirical_normalization:
                if resumed_training:
                    self.obs_normalizer[agent].load_state_dict(agent_data[agent]["obs_norm_state_dict"])
                    self.privileged_obs_normalizer[agent].load_state_dict(agent_data[agent]["privileged_obs_norm_state_dict"])
                else:
                    self.privileged_obs_normalizer[agent].load_state_dict(agent_data[agent]["obs_norm_state_dict"])
            # -- load optimizer if used
            if load_optimizer and resumed_training:
                self.alg[agent].optimizer.load_state_dict(agent_data[agent]["optimizer_state_dict"])
                # if self.alg[agent].rnd and "rnd_optimizer_state_dict" in agent_data[agent]:
                #     self.alg[agent].rnd_optimizer.load_state_dict(agent_data[agent]["rnd_optimizer_state_dict"])
            # -- load current learning iteration
            if resumed_training:
                self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos", {})

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        policies = {}
        for agent in self.env.agent_names:
            if device is not None:
                self.alg[agent].policy.to(device)
            policies[agent] = self.alg[agent].policy.act_inference
            if self.cfg["empirical_normalization"]:
                if device is not None:
                    self.obs_normalizer[agent].to(device)
                policies[agent] = lambda x, a=agent: self.alg[a].policy.act_inference(self.obs_normalizer[a](x))  # noqa: E731
        return policies

    def train_mode(self):
        for agent in self.env.agent_names:
            # -- PPO
            self.alg[agent].policy.train()
            # # -- RND
            # if self.alg[agent].rnd:
            #     self.alg[agent].rnd.train()
            # -- Normalization
            if self.empirical_normalization:
                self.obs_normalizer[agent].train()
                self.privileged_obs_normalizer[agent].train()

    def eval_mode(self):
        for agent in self.env.agent_names:
        # -- PPO
            self.alg[agent].policy.eval()
            # # -- RND
            # if self.alg[agent].rnd:
            #     self.alg[agent].rnd.eval()
            # -- Normalization
            if self.empirical_normalization:
                self.obs_normalizer[agent].eval()
                self.privileged_obs_normalizer[agent].eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'.")
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")

        # initialize torch distributed
        torch.distributed.init_process_group(
            backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
        )
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)