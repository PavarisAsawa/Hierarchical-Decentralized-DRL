# Hierarchical Decentralized Deep Reinforcement Learning
> this repository is on FRA 503: Deep Reinforcement Learning for Robotics class Project
- IsaacLab extension for Hierarchical Decentralized control using Deep reinforcement learning in Isaaclab environment

### Description of the project
- In nature, control of movement happens in a hierarchical and decentralized fashion is a central characteristic of biological motor control that allows for fast responses relying on local sensory information. This motivates to ask whether decentralization as seen in biological control architectures might also be beneficial for embodied sensory-motor control systems when using DRL 

### Goal
- The goal is to develop a Hierarchical Decentralized Deep Reinforcement Learning (HDDRL) architecture for a four-legged robot, where control is divided into two layers: 
    - A High-Level Controller (HLC): handles strategic navigation decisions based on heuristic functions (e.g., distance between robot and target). 
    -  A Low-Level Controller (LLC): handles local locomotion tasks like standing and walking, using only local joint states and latent information from the HLC. 

### Requirements 
- Develop a hierarchical and decentralized deep reinforcement learning (DRL) control system for a four-legged robot. 
- Implement two-layer control architecture: 
    - High-Level Controller (HLC): Responsible for navigation decisions based on heuristic function (distance between agent and target). 
    - Low-Level Controller (LLC): Responsible for local locomotion control (standing, walking) using only local sensory inputs and HLC signals. 
- Both HLC and LLC must be trained using PPO. 
- Evaluate centralized vs decentralized policies in terms of robustness, adaptability, and performance in navigation tasks. 

### Scope
- Using four-legged Ant robot model in Isaac lab environment. 
- Using PPO for baseline algorithm. 
- Compare LLC Centralize (4 legs in 1 agent) and fully-decentralize (separate all 4 legs) with centralize HLC. 

## Installation

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# Option 1: HTTPS
git clone https://github.com/PavarisAsawa/Hierarchical-Decentralized-DRL.git

```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
# Enter the repository
cd Hierarchical-Decentralized-DRL
python -m pip install -e source/HDDRL
```

- Verify that the extension is correctly installed by running the following command:

```bash
python scripts/rsl_rl/train.py --task=central
```

## System Overview
This project is contain with 2 level with controller
- `High Level` : for navigation task
- `Low Level` : for locomotion task
    - for low level we can model as 2 type of controller is 
    - Decentralized controller : we seperate policy to control each leg
    - Centralize controller : all leg is controlled by only 1 policy

<div style="text-align: center;">
    <img src="images/architecture.png" alt="architecture" width="400"/>
    <p><em>System Architecture (c : low level with centralized controller , d : low level with decentralized contoller)</em></p>
    <p><em>source : https://arxiv.org/pdf/2210.08003</em></p>
</div>


## Environment and Agent
### Environment and Agent : Low Level Decentralized
For Low Level Decentralized we using IPPO algorithms which is each leg seperate agent but using same global or average reward for updating policy

#### Observavtion space:
- Global Observation (every leg have this Observation as same)
    - Linear Velocity: Linear velocity of robot in x, y axis (size = 2)
    - Projected Gravity: Direction of gravity force project to robot frame (size = 3)
    - Velocity Command: Target velocity command in x, y axis in locomotion task (size = 2)
    - Foot Contact: Contact sensor at all foot that tell each leg contact to floor or not (size = 4)
- Local Observation (each leg have just itself leg observation)
    - Joint Position: position of each joint on leg (size = 2)
    - Joint Velocity: velocity of each joint on leg (size = 2) 
- Size of observation space is 15 for each leg.

#### Reward:
- Linear Velocity: Reward given that robot velocity close to target velocity
```
lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25) * self.cfg.tracking_lin_vel_weight
```
- Upright Posture: Reward given that robot robot stand straight
```
up_reward = torch.zeros(self.num_envs , device=self.sim.device)
up_reward = torch.where(self.robot.data.projected_gravity_b[: , 2] > -0.9, torch.abs(self.robot.data.projected_gravity_b[: , 2]) * self.cfg.up_weight, up_reward)
```
- Action: Reward given when robot do an action
```
fl_action = torch.sum(torch.square(self.actions["fl_leg"]), dim=1) * self.cfg.actions_cost_scale
```
- Electricity: Reward given when robot make an electricity
```
 fl_electricity_cost = torch.sum(torch.abs(self.actions["fl_leg"] * self.dof_vel[: , self.fl_indices] * self.cfg.dof_vel_scale) * self.motor_effort_ratio[:2].unsqueeze(0),dim=-1,) * self.cfg.energy_cost_scale
```
- DoF at Limit: Reward given that robot joint position more than minimum
```
fl_dof_at_limit_cost = torch.sum(self.dof_pos_scaled[: , self.fl_indices] > 0.98, dim=-1) * self.cfg.dof_at_limit_scale
```

For reward Action, Electricity and DoF at Limit there are local reward which is each agent(leg) calculate seperate and return seperate reward

For IPPO algorithm so reward will average to make global reward that use for update each leg policy

If the robot is terminated due to an out-of-range base position in z axis, the agent will receive a reward equal to death_cost.

**List of weight**
- tracking_lin_vel_weight: float = 3.0
- up_weight: float = -0.3
- dof_vel_scale: float = 0.2
- energy_cost_scale: float = 0.00025
- actions_cost_scale: float = 0.0001
- dof_at_limit_scale: float =0.01
- death_cost: float = -2.0   

#### Termination:
- Timeout: The episode ends when the robot stays alive for the specified time limit.
- Torso position: The episode ends if the robot's base position (Z) goes too high (more than 1.0) or too low (less than 0.36). This prevents the agent from jumping or collapsing flat on the floor.

#### Action
Action is joint effort in decentralized each agent can control 2 joint in each leg.

#### Agent:

Each leg using seperate agent which is PPO base. Each leg have same hyper paprameter which is
```
experiment_name = "decentral"
num_steps_per_env = 24
max_iterations = 1500
save_interval = 100
empirical_normalization = False

# === Policy Network ===
policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_hidden_dims=[128, 128, 128],
    critic_hidden_dims=[128, 128, 128],
    activation="elu",
)

# === PPO Hyperparameters ===
algorithm = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=3e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)
```
### Environment and Agent : Low Level Centralized
For Low Level Centralized we using normal PPO algorithms

#### Observavtion space:
- Linear Velocity: Linear velocity of robot in x, y axis (size = 2)
- Projected Gravity: Direction of gravity force project to robot frame (size = 3)
- Velocity Command: Target velocity command in x, y axis in locomotion task (size = 2)
- Foot Contact: Contact sensor at all foot that tell each leg contact to floor or not (size = 4)
- Joint Position: position of each joint on leg (size = 8)
- Joint Velocity: velocity of each joint on leg (size = 8) 

Size of observation space is 27.

#### Reward:
- Linear Velocity: Reward given that robot velocity close to target velocity
```
lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25) * self.cfg.tracking_lin_vel_weight
```
- Upright Posture: Reward given that robot robot stand straight
```
up_reward = torch.zeros(self.num_envs , device=self.sim.device)
up_reward = torch.where(self.robot.data.projected_gravity_b[: , 2] > -0.9, torch.abs(self.robot.data.projected_gravity_b[: , 2]) * self.cfg.up_weight, up_reward)
```
- Action: Reward given when robot do an action
```
action_rew = torch.sum(torch.square(self.actions), dim=1) * self.cfg.actions_cost_scale /4.0
```
- Electricity: Reward given when robot make an electricity
```
electricity_cost = torch.sum(torch.abs(self.actions * self.dof_vel * self.cfg.dof_vel_scale) * self.motor_effort_ratio.unsqueeze(0),dim=-1,) * self.cfg.energy_cost_scale / 4.0
```
- DoF at Limit: Reward given that robot joint position more than minimum
```
dof_at_limit_cost = torch.sum(self.dof_pos_scaled > 0.98, dim=-1)  * self.cfg.dof_at_limit_scale /4.0
```

For reward Action, Electricity and DoF at Limit there are local reward for decentralize so in central we sum all 4 leg and devide by 4 to calculate average from all leg.

If the robot is terminated due to an out-of-range base position in z axis, the agent will receive a reward equal to death_cost.

**List of weight**
- tracking_lin_vel_weight: float = 3.0
- up_weight: float = -0.3
- dof_vel_scale: float = 0.2
- energy_cost_scale: float = 0.00025
- actions_cost_scale: float = 0.0001
- dof_at_limit_scale: float =0.01
- death_cost: float = -2.0   

#### Termination:
- Timeout: The episode ends when the robot stays alive for the specified time limit.
- Torso position: The episode ends if the robot's base position (Z) goes too high (more than 1.0) or too low (less than 0.36). This prevents the agent from jumping or collapsing flat on the floor.

#### Action
Action is joint effort in centralized agent can control all of joint (8 joints)

#### Agent:

It's PPO base. And have hyper paprameter which is
```
num_steps_per_env = 24
max_iterations = 1500
save_interval = 100
experiment_name = "central"
empirical_normalization = False

policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
)
algorithm = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=3e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)
```
### Environment and Agent : High Level
In high level it have objective to nevigate robot to target postion.

#### Observavtion space:
- Heading: direction that robot heading to (yaw angle) (size = 1)
- Distance: Distance from robot to targ
et position in x-y axis (size = 2)

Size of observation space is 3.

#### Reward:
- Distance Penalty: Penalty given when robot position far from target position.
```
target = -torch.norm(self._commands - self.pos_env , dim=1)
```

#### Termination:
- Timeout: The episode ends when the robot stays alive for the specified time limit.
- Torso position: The episode ends if the robot's base position (Z) goes too high (more than 1.0) or too low (less than 0.36). This prevents the agent from jumping or collapsing flat on the floor.
- Reach target: The episode ends if distance between robot and target is in threshold.

#### Action

Action is target velocity in x and y direction in robot frame and pass it to low level agent.

#### Agent:

It's PPO base. And have hyper paprameter which is (policy_low and algorithm_low is hyperparameter of low level should config to same as low level model)
```
num_steps_per_env = 8
max_iterations = 1500
save_interval = 50
experiment_name = "Navigate_decen"
empirical_normalization = False
policy = RslRlPpoActorCriticCfg(
    init_noise_std=0.5,
    actor_hidden_dims=[128, 128],
    critic_hidden_dims=[128, 128],
    activation="elu",
)
algorithm = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)

# === Policy Network ===
policy_low = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_hidden_dims=[128, 128, 128],
    critic_hidden_dims=[128, 128, 128],
    activation="elu",
)

# === PPO Hyperparameters ===
algorithm_low = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=3e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)
```

## Training and Playing
### Training and Playing : Low Level Decentralized
**Train**
```
python scripts/rsl_rl/train_IPPO.py --task decentral
```
**Play**
```
python scripts/rsl_rl/play_IPPO.py --task decentral --load_run floder_name(inside logs/rsl_rl/decentral) --checkpoint model_xxx --num_envs 1
```

**Change Agent configuration**
- You can change config at "source/HDDRL/HDDRL/tasks/decentralized/agents/rsl_rl_ppo_cfg.py"
### Training and Playing : Low Level Centralized
**Train**
```
python scripts/rsl_rl/train.py --task central
```
**Play**
```
python scripts/rsl_rl/play.py --task central --load_run floder_name(inside logs/rsl_rl/central) --checkpoint model_xxx --num_envs 1
```

**Change Agent configuration**
- You can change config at "source/HDDRL/HDDRL/tasks/centralized/agents/rsl_rl_ppo_cfg.py"

### Training and Playing : High Level
#### High Level with Decentral
**Setting up**
- Change low level decentral model at file "scripts/rsl_rl/train_high_decen.py" variable **"model_name"**

- Change **policy_low** config and **algorithm_low** config to match config of low level decentral model at "source/HDDRL/HDDRL/tasks/highlevel/agents/rsl_rl_ppo_cfg.py" class **DecenNavigationEnvPPORunnerCfg**

- Change source/HDDRL/HDDRL/tasks/highlevel/\__init__.py to
```
"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DecenNavigationEnvPPORunnerCfg",
# "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CenNavigationEnvPPORunnerCfg",
```

**Train**
```
python scripts/rsl_rl/train_high_decen.py --task highlevel
```

**Play**
```
python scripts/rsl_rl/play_high_decen.py --task highlevel --load_run floder_name(inside logs/rsl_rl/Navigate_decen) --checkpoint model_xxx --num_envs 1
```

**Change Agent configuration**
- You can change config at "source/HDDRL/HDDRL/tasks/highlevel/agents/rsl_rl_ppo_cfg.py" class **DecenNavigationEnvPPORunnerCfg**

#### High Level with Central
**Setting up**
- Change low level central model at file "scripts/rsl_rl/train_high_cen.py" variable **"model_name"**

- Change **policy_low** config and **algorithm_low** config to match config of low level central model at "source/HDDRL/HDDRL/tasks/highlevel/agents/rsl_rl_ppo_cfg.py" class **CenNavigationEnvPPORunnerCfg**

- Change source/HDDRL/HDDRL/tasks/highlevel/\__init__.py to
```
# "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DecenNavigationEnvPPORunnerCfg",
"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CenNavigationEnvPPORunnerCfg",
```

**Train**
```
python scripts/rsl_rl/train_high_cen.py --task highlevel
```

**Play**
```
python scripts/rsl_rl/play_high_cen.py --task highlevel --load_run floder_name(inside logs/rsl_rl/Navigate_cen) --checkpoint model_xxx --num_envs 1
```

**Change Agent configuration**
- You can change config at "source/HDDRL/HDDRL/tasks/highlevel/agents/rsl_rl_ppo_cfg.py" class **CenNavigationEnvPPORunnerCfg**

## Result
### Result : Low Level
Task Performance
- centralized:

![<video controls src="images/cen.mp4" title="Title"></video>](images/cen.gif)

- decentralized:

![<video controls src="images/decen.mp4" title="Title"></video>](images/decen.gif)

From video we can see that centralized can do task more smooth than decentral. It can control velocity better and also have better walking posture.

Compare in RL term:
![alt text](images/low_result.png)
![alt text](images/low_vel_error.png)

From graph
- episode length: Decentral have more episode length espacially at first which mean it learn to not terminate before do a locomotion task
- reward: Decentral have more reward which can assume that it from others reward not from velocity error. And it also have more variance in decentral
- Velocity error: we can see that it have higher variance for the decentralized policy. And more error than central

**Conclude**
Decentralized Learning:
- Each leg is controlled independently, resulting in uneven usage—some legs contribute more to locomotion while others contribute less or remain inactive.
- Because the legs do not share a common state or policy, the same leg may behave different when others leg not same state. It's make similar states need to do morre than 1 across episodes.
- This leads to higher variance in velocity errors, as shown in the graphs.

Centralized Learning:
- All four legs operate under a shared policy and have full state information, enabling more synchronized movement.
- This consistent coordination results in lower velocity errors and less variance, leading to more stable and reliable behavior.

### Result : High Level
Task Performance
- centralized:

![<video controls src="images/Navigate_cen.mp4" title="Title"></video>](images/Navigate_cen.gif)

- decentralized:

![<video controls src="images/Navigate_decen.mp4" title="Title"></video>](images/Navigate_decen.gif)

From video we can see that decentralized can reach target faster and faster adapt to next target velocity. Central still better posture but it use more times to adapt to new veocity commands.
Compare in RL term:
![alt text](images/high_result.png)

From graph
- Central learn navigate faster because robust low lovel task. But decentral reach higher reward because speed of adaptation.

**Conclude**
Decentralized Learning:
- Independent leg control allows faster adaptation to target speed changes because each leg can respond individually without needing synchronization.
- However, the learning process is noisier due to the lack of global coordination.

Centralized Learning:
- Achieves slightly lower final reward but demonstrates more stable and predictable performance.
- When target velocities change, the agent must coordinate all legs simultaneously, which slows adaptation.
- Learns faster in the early stages due to lower low-level velocity error, enabling more consistent feedback to the high-level controller.

## Conclusion
Decentralized learning provides more flexibility and can achieve higher rewards in dynamic tasks, but at the cost of higher variance and inconsistent behavior across legs. This can be beneficial when rapid adaptation is needed. In contrast, centralized learning ensures consistent and synchronized control, resulting in smoother, more stable locomotion and faster convergence in the early stages of training. The choice between these methods depends on the task requirements: decentralized for adaptability and centralized for stability and posture.