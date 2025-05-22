import gymnasium as gym

# from . import agents
##
# Register Gym environments.
##
from . import agents

gym.register(
    id='navigate',
    entry_point=f"{__name__}.ant_navigate_env:AntNavigateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_navigate_env:AntNavigateCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NavigationEnvPPORunnerCfg",
    },
)

# gym.register(
#     id='central',
#     entry_point=f"{__name__}.ant_central_env:AntCentralEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.ant_central_env:AntCentralEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:testPPORunnerCfg",
#     },
# )

# gym.register(
#     id='test',
#     entry_point=f"{__name__}.ant_test_env:AntEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.ant_test_env:AntEnvCfg",
#         # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SlalomPPORunnerCfg",
#     },
# )