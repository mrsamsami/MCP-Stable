from typing import List, Tuple

import torch
import torch as th
from torch import nn


def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    return model


class MCPHiddenLayers(nn.Module):
    """
    Custom hidden network architecture for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param state_dim: dimension of the input features
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        num_primitives: int,
        learn_log_std: bool,
        big_model: bool,
    ):
        super(MCPHiddenLayers, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions

        if big_model:
            hidden_layer_sizes = [512, 256]
            value_hidden_layer_sizes = [1024, 512]
        else:
            hidden_layer_sizes = [64, 64]
            value_hidden_layer_sizes = [64, 64]

        self.latent_dim_pi = 512
        self.latent_dim_vf = value_hidden_layer_sizes[1]

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.num_primitives = num_primitives
        self.action_dim = action_dim
        self.learn_log_std = learn_log_std

        # build the Policy network hidden layers

        # Gating Function:
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
            nn.ReLU(),
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
            nn.ReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_layer_sizes[1] * 2, hidden_layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[1], num_primitives),
            nn.Sigmoid(),
        )

        self.primitive_state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
            nn.ReLU(),
        )

        self.primitives = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[1]),
                    nn.ReLU(),
                    nn.Linear(hidden_layer_sizes[1], self.action_dim * 2),
                )
                for _ in range(num_primitives)
            ]
        )

        # build the Value network hidden layers
        self.value_net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, value_hidden_layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(value_hidden_layer_sizes[0], self.latent_dim_vf),
            nn.ReLU(),
        )

    def freeze_primitives(self):
        self.primitives = nn.ModuleList([freeze_model(mod) for mod in self.primitives])
        self.primitives = freeze_model(self.primitives)

    def forward_weights(self, features: th.Tensor) -> th.Tensor:
        state, goal = torch.split(features, [self.state_dim, self.goal_dim], -1)
        state_embed = self.state_encoder(state)
        goal_embed = self.goal_encoder(goal)
        embed = th.cat((state_embed, goal_embed), -1)
        weights = self.gate(embed)
        return weights

    def forward_primitive(self, i: int, prim_embed: th.Tensor) -> List[th.Tensor]:
        out = self.primitives[i](prim_embed)
        mu, log_std = th.split(out, self.action_dim, -1)
        sigma = th.ones_like(mu) * log_std.exp()
        return mu, sigma

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        bs = features.shape[0]

        state, _ = torch.split(features, [self.state_dim, self.goal_dim], -1)
        weights = self.forward_weights(features)

        prim_embed = self.primitive_state_encoder(state)

        outs = [
            self.forward_primitive(i, prim_embed) for i in range(self.num_primitives)
        ]
        mus, sigmas = zip(*outs)

        mus = torch.stack(mus, 1)
        sigmas = torch.stack(sigmas, 1)
        weights = weights[..., None]

        assert (
            mus.shape[0] == bs
            and mus.shape[1] == self.num_primitives
            and mus.shape[2] == self.action_dim
        )
        assert (
            sigmas.shape[0] == bs
            and sigmas.shape[1] == self.num_primitives
            and sigmas.shape[2] == self.action_dim
        )

        denom = (weights / sigmas).sum(-2)
        unnorm_mu = (weights / sigmas * mus).sum(-2)

        mean = unnorm_mu / denom
        if not self.learn_log_std:
            scale_tril = 1 / denom
            return mean, scale_tril
        else:
            return mean

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        value = self.value_net(features)
        return value

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)


class MCPPOHiddenLayers(nn.Module):
    """
    Custom hidden network architecture for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param state_dim: dimension of the input features
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        models: List[nn.Module],
        learn_log_std: bool,
        big_model: bool,
    ):
        super(MCPPOHiddenLayers, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions

        if big_model:
            hidden_layer_sizes = [512, 256]
            value_hidden_layer_sizes = [1024, 512]
        else:
            hidden_layer_sizes = [64, 64]
            value_hidden_layer_sizes = [64, 64]

        self.latent_dim_pi = 512
        self.latent_dim_vf = value_hidden_layer_sizes[1]

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.num_primitives = len(models)
        self.action_dim = action_dim
        self.learn_log_std = learn_log_std

        # build the Policy network hidden layers

        # Gating Function:
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
            nn.ReLU(),
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
            nn.ReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_layer_sizes[1] * 2, hidden_layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[1], self.num_primitives),
            nn.Sigmoid(),
        )

        self.primitive_state_encoder = nn.Identity()

        self.primitives = [freeze_model(mod) for mod in models]

        # build the Value network hidden layers
        self.value_net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, value_hidden_layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(value_hidden_layer_sizes[0], self.latent_dim_vf),
            nn.ReLU(),
        )

    def forward_weights(self, features: th.Tensor) -> th.Tensor:
        state, goal = torch.split(features, [self.state_dim, self.goal_dim], -1)
        state_embed = self.state_encoder(state)
        goal_embed = self.goal_encoder(goal)
        embed = th.cat((state_embed, goal_embed), -1)
        weights = self.gate(embed)
        return weights

    def forward_primitive(self, i: int, state: th.Tensor) -> List[th.Tensor]:
        model = self.primitives[i]
        get_action = lambda embed: model.action_net(
            model.mlp_extractor.forward_actor(embed)
        )
        prim_embed = self.primitive_state_encoder(state)
        mu = get_action(prim_embed)
        log_std = self.primitives[i].log_std.clone()
        sigma = th.ones_like(mu) * log_std.exp()
        return mu, sigma

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        bs = features.shape[0]

        state, _ = torch.split(features, [self.state_dim, self.goal_dim], -1)
        weights = self.forward_weights(features)

        outs = [self.forward_primitive(i, state) for i in range(self.num_primitives)]
        mus, sigmas = zip(*outs)

        mus = torch.stack(mus, 1)
        sigmas = torch.stack(sigmas, 1)
        weights = weights[..., None]

        assert (
            mus.shape[0] == bs
            and mus.shape[1] == self.num_primitives
            and mus.shape[2] == self.action_dim
        )
        assert (
            sigmas.shape[0] == bs
            and sigmas.shape[1] == self.num_primitives
            and sigmas.shape[2] == self.action_dim
        )

        denom = (weights / sigmas).sum(-2)
        unnorm_mu = (weights / sigmas * mus).sum(-2)

        mean = unnorm_mu / denom
        assert mean.shape == (bs, self.action_dim)
        if not self.learn_log_std:
            scale_tril = 1 / denom
            return mean, scale_tril
        else:
            return mean

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        value = self.value_net(features)
        return value

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)
