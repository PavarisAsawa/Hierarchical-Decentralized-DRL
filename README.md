# Hierarchical Decentralized Deep Reinforcement Learning
> this repository is on FRA 503: Deep Reinforcement Learning for Robotics class Project
- IsaacLab extension for Hierarchical Decentralized control using Deep reinforcement learning in Isaaclab environment

### Description of the project
- In nature, control of movement happens in a hierarchical and decentralized fashion is a central characteristic of biological motor control that allows for fast responses relying on local sensory information. This motivates to ask whether decentralization as seen in biological control architectures might also be beneficial for embodied sensory-motor control systems when using DRL 
- 

### Goal
- The goal is to develop a Hierarchical Decentralized Deep Reinforcement Learning (HDDRL) architecture for a four-legged robot, where control is divided into two layers: 
    - A High-Level Controller (HLC): handles strategic navigation decisions based on heuristic functions (e.g., distance between robot and target). 
    -  A Low-Level Controller (LLC): handles local locomotion tasks like standing and walking, using only local joint states and latent information from the HLC. 

### Requirements 
- Develop a hierarchical and decentralized deep reinforcement learning (DRL) control system for a four-legged robot. 
- Implement two-layer control architecture: 
    - High-Level Controller (HLC): Responsible for navigation decisions based on heuristic function (distance between agent and target). 
    - Low-Level Controller (LLC): Responsible for local locomotion control (standing, walking) using only local sensory inputs and HLC signals. 
- Both HLC and LLC must be trained using model-free RL methods (e.g., PPO, A2C). 
- Evaluate centralized vs decentralized policies in terms of robustness, adaptability, and performance in navigation tasks. 

### Scope
- Using four-legged Ant robot model in Isaac lab environment. 
- Using PPO and A2C for baseline algorithm. 
- Compare LLC Centralize (4 legs in 1 agent) and fully-decentralize (separate all 4 legs) with centralize HLC. 
- State observation is heuristic function (distance between agent and target) for HLC and joint state for LLC. 
- Using Curriculum learning for training robots. 

## Installation

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# Option 1: HTTPS
git clone https://github.com/PavarisAsawa/Hierarchical-Decentralized-DRL.git

```



- Using a python interpreter that has Isaac Lab installed, install the library

```bash
# Enter the repository
cd IsaacLabExtensionTemplate
python -m pip install -e source/HDDRL
```

- Verify that the extension is correctly installed by running the following command:

```bash
python scripts/rsl_rl/train.py --task=Template-Isaac-Velocity-Rough-Anymal-D-v0
```
