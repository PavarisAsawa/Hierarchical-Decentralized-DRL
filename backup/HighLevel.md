# HDDRL : High Level

## Reinforcement Learning : Environment
In IsaacLab 

### Reward Term
- Distance Penalty : get penalty when have a distance between target and 
- Action : encourages the agent to actually move its legs (i.e. to swing rather than remain still)
- Electricity Cost : promotes energy-efficient motions
- DoF limit : discourages holding a joint at its extreme and thus encourages continual leg swinging
- death : if Agent is Terminated got the penalty

all weight value is contain in below list

**List of weight**
- tracking_lin_vel_weight: float = 3.0
- up_weight: float = -0.3
- dof_vel_scale: float = 0.2
- energy_cost_scale: float = 0.00025
- actions_cost_scale: float = 0.0001
- alive_reward_scale: float = 0.5
- dof_at_limit_scale: float =0.01
- death_cost: float = -2.0    
- termination_height: float = 0.36
- termination_height_up : float = 1.0

### Observation Term
- Heading : Heading or yaw of robot relative to world frame
- Distance : Distance between target and agent

### Action
- Action is target between 

### Termination Condition
termination condition is same for both centralized and decentralized is 
- Terminate when base position(Z) < termination_height Or > termination_height_up

## Reinforcement Learning : Agent
