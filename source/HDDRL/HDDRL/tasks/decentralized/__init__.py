import gymnasium as gym

# from . import agents
##
# Register Gym environments.
##
from .ant_decentral_env import AntLegEnvCfg
from . import agents

gym.register(
    id='decentral',
    entry_point=f"{__name__}.ant_decentral_env:AntLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_decentral_env:AntLegEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SlalomPPORunnerCfg",
    },
)

gym.register(
    id='central',
    entry_point=f"{__name__}.ant_central_env:AntCentralEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_central_env:AntCentralEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:testPPORunnerCfg",
    },
)

gym.register(
    id='test',
    entry_point=f"{__name__}.ant_test_env:AntEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_test_env:AntEnvCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SlalomPPORunnerCfg",
    },
)