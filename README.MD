# Multiplicative Compositional Policies

This repo provides implementations of Multiplicative Compositional Policies (MCP), which is a method for learning reusable motor skills that can be composed to produce a range of complex behaviors. All code is written in Python 3, using PyTorch, NumPy, and [Stable-Baselines3](https://stable-baselines3.readthedocs.io). Experiments are simulated with the [MuJoCo Physics Engine](https://mujoco.org/). The project is built on [DRLoco](https://drloco.readthedocs.io), an implementation of [DeepMimic Framework](https://github.com/xbpeng/DeepMimic) with Stable-Baselines3.

---

>  MCP: Learning Composable Hierarchical Control with Multiplicative Compositional Policies. NeurIPS 2019. <br /> 
>  __[[Paper]](https://arxiv.org/abs/1905.09808)__

---

### Experiments Details
The character we decided to work with is a simple degrees-of-freedom (DoFs) ant. Although the paper employs imitative rewards for other characters, it trains the ant by a common RL approach (no imitation). Consequently, we trained the ant in this manner. Additionally, we devised various training methods:
Model Name | Description
------------ | -------------
MCPPO | The paper jointly trained primitives end-to-end, leading to the specializations. In MCPPO, we trained each primitive separately for an individual task. 
MCP Naive | Like the other characters, we incorporate expert demonstrations in the pre-training phase.
<p align="center">
<img src="archi.png" alt="An illustraion of the method" title="An illustraion of the method" width = "50%"/>
</p>


#### Pre-training Tasks
In our experiments, the pre-training phase consists of four different heading tasks: heading north, south, east, and west. For MCP Naive, we provided a corpus of reference motions. We followed the approach used to pre-train the humanoid in the paper. For the rest, we specified the goals and the reward functions to encourage the agent to navigate the desired direction.

#### Fine-tuning Tasks
To evaluate the agents, we considered four new heading tasks: north-west, north-east, south-west, and south-east. The goal and the reward function are defined in the same way as in pre-training.


#### Expert Data for MCP Naive
Since there is no mocap data for the ant, we needed to develop experts to generate reference data. Accordingly, we trained four different MLP policies with PPO, each of which learned to navigate north/south/east/west. We consider each policy as an expert in a particular direction, and thus, we produce actions that can be regarded as reference trajectories.

The reference trajectories used for MCP Naive that were produced by PPO were observed to be very noisy leading to poor performance. Also, the sensitivity of Ant to the reward scaling factors led to the Ant being unable to learn via imitation of reference trajtory.



## Installation
To install requirements, please refer to [DRLoco installation documentation](https://drloco.readthedocs.io/en/latest/main/install.html).

## How to run

#### Initial experiments

```sh
python mcppo.py

python mcp_naive.py
```

#### Experiments from paper

```sh
cd mcp

python train_mcp.py
python train_mcppo.py
python scratch_ant.py

python transfer.py
```

## To generate the trajectories


#### Initial Experiments
```sh

bash gen_plots.sh
bash make_traj.sh
```

#### Experiments from paper
```sh
cd mcp

bash gen_plots.sh
bash make_traj.sh
```
