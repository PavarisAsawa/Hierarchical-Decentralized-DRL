# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
print("args_cli.task: ", args_cli.task)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
# from PlasticNeuralNet.tasks.slalom import *
from isaaclab.envs import DirectMARLEnv , DirectMARLEnvCfg

import HDDRL.tasks

# PLACEHOLDER: Extension template (do not remove this comment)
print("args_cli.task: ", args_cli.task)


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    # env = gym.make(args_cli.task, cfg=env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            # actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            actions = random_action2()
            print("actions: ", actions)
            env.step(actions)

    # close the simulator
    env.close()

def random_action2():
    dim = args_cli.num_envs
    if args_cli.num_envs is None:
        dim = 4096
    return torch.empty(dim, 8).uniform_(-20, 20)  #20*torch.rand(dim, 8)

def random_action() -> dict[str, torch.Tensor]:
    # shape tuple must be positional, not as size=
    dim = args_cli.num_envs
    if args_cli.num_envs is None:
        dim = 4096
    # action = torch.zeros(dim, 2)
    action = torch.zeros(dim, 2)
    return {
        "fl_leg": 2*torch.rand(dim, 2),
        "fr_leg": 2*torch.rand(dim, 2),
        "hl_leg": 2*torch.rand(dim, 2),
        "hr_leg": 2*torch.rand(dim, 2),
    }

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()