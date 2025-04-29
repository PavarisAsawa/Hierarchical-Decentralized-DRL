import gymnasium as gym

# from . import agents
##
# Register Gym environments.
##
from .ant_decentral_env import AntLegEnvCfg

gym.register(
    id='decentral',
    entry_point=f"{__name__}.ant_decentral_env:AntLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_decentral_env:AntLegEnvCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SlalomPPORunnerCfg",
    },
)