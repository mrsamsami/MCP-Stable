# Multiplicative Compositional Policies

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