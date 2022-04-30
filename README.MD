# Multiplicative Compositional Policies

This repo provides implementations of Multiplicative Compositional Policies (MCP), which is a method for learning reusable motor skills that can be composed to produce a range of complex behaviors. All code is written in Python 3, using PyTorch and NumPy. 

---

>  MCP: Learning Composable Hierarchical Control with Multiplicative Compositional Policies. NeurIPS 2019. <br /> 
>  __[[Paper]](https://arxiv.org/abs/1905.09808)__

---

## Initial experiments

```sh
python mcppo.py

python mcp_naive.py
```

## Experiments from paper

```sh
cd mcp

python train_mcp.py
python train_mcppo.py
python scratch_ant.py

python transfer.py
```

## To generate the trajectories


### Initial Experiments
```sh

bash gen_plots.sh
bash make_traj.sh
```

### Experiments from paper
```sh
cd mcp

bash gen_plots.sh
bash make_traj.sh
```
