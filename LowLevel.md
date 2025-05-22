# HDDRL : Low Level

## Reinforcement Learning : Environment
In IsaacLab 

### Reward Term
- Linear Velocity Tracking : encourages matching the desired base velocity.
- Upright Posture term : keeps the robot upright and prevents falling
- Alive : constant positive reward each step for remaining “alive” (not terminated) 
- Action : encourages the agent to actually move its legs (i.e. to swing rather than remain still)
- Electricity Cost : promotes energy-efficient motions
- DoF limit : discourages holding a joint at its extreme and thus encourages continual leg swinging
- death : if Agent is Terminated got the penalth

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

#### Reward Term in decentralized
- for decentralize we seperate in to Global and Local reward by
- **Global**
    - Linear Velocity Tracking
    - Upright Posture
    - Alive 
- **Local**
    - Action
    - Electricity Cost
    - DoF limit

### Action
for centralized agent can control all of joint, but decentralized can control only one leg each controller

### Termination Condition
termination condition is same for both centralized and decentralized is 
- Terminate when base position(Z) < termination_height Or > termination_height_up

## Reinforcement Learning : Agent
