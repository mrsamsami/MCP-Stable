'''Code adopted from: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#advanced-example'''
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch
import torch as th
from torch import nn, distributions

from drloco.config import hypers
from drloco.common.utils import log
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomHiddenLayers(nn.Module):
    """
    Custom hidden network architecture for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the input features
    """
    def __init__(
        self,
        feature_dim: int
    ):
        super(CustomHiddenLayers, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = hypers.hid_layer_sizes[-1]
        self.latent_dim_vf = hypers.hid_layer_sizes[-1]

        # Build the hidden layers of a fully connected neural network
        # currently we're using the same architecture for pi and vf
        layers = []
        for size, activation_fn in zip(hypers.hid_layer_sizes, hypers.activation_fns):
            layers += [nn.Linear(feature_dim, size), activation_fn()]
            feature_dim = size

        # build the Policy network hidden layers
        self.policy_net = nn.Sequential(*layers)
        # build the Value network hidden layers
        self.value_net = nn.Sequential(*layers)

        # log('Hidden Layer Network Architecture:\n' + str(self.policy_net))


    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

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
        num_primitives: int
    ):
        super(MCPHiddenLayers, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.latent_dim_pi = hypers.hid_layer_sizes[-1]
        self.latent_dim_vf = hypers.hid_layer_sizes[-1]
        self.num_primitives = num_primitives
        self.param_size = hypers.mcp_hid_layer_sizes[-1] // 2

        # build the Policy network hidden layers

        # Gating Function:
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_primitives),
            nn.Sigmoid()
        )

        self.primitive_state_encoder = nn.Sequential(
            nn.Sequential(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.primitives = nn.ModuleList([nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, self.param_size * 2)) for _ in range(num_primitives)])

        # self.mcp_shared_net = self.get_net(hypers.mcp_shared_hid_layer_sizes, hypers.mcp_shared_activation_fns)
        #
        # self.primitives, self.goal_encoders, self.state_encoders, self.gates = [], [], [], []
        # for i in range(num_primitives):
        #     self.primitives.append(self.get_net(hypers.mcp_hid_layer_sizes, hypers.mcp_activation_fns))
        #     self.goal_encoders.append(self.get_net(hypers.gate_hid_layer_sizes, hypers.gate_activation_fns))
        #     self.state_encoders.append(self.get_net(hypers.gate_hid_layer_sizes, hypers.gate_activation_fns))
        #     self.gates.append(self.get_net(hypers.gate_hid_layer_sizes, hypers.gate_activation_fns))

        # build the Value network hidden layers
        self.value_net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        # self.value_net = self.get_net(hypers.hid_layer_sizes, hypers.activation_fns)

        # log('Hidden Layer Network Architecture:\n' + str(self.policy_net))

    def get_net(self, hid_layer_sizes, activation_fns):
        layers = []
        feature_dim = self.state_dim
        for size, activation_fn in zip(hid_layer_sizes, activation_fns):
            layers += [nn.Linear(feature_dim, size), activation_fn()]
            feature_dim = size
        return nn.Sequential(*layers)

    def forward_actor(self, state: th.Tensor, goal: th.Tensor) -> th.Tensor:
        mus, sigmas, weights = [], [], []

        state_embed = self.state_encoder(state)
        goal_embed = self.goal_encoder(goal)
        embed = th.cat((state_embed, goal_embed), -1)
        weights = self.gate(embed)

        prim_embed = self.primitive_state_encoder(state)
        for i in range(self.num_primitives):
            out = self.primitives[i](prim_embed)
            mu, sigma = th.split(out, 2, -1)
            mus.append(mu)
            sigmas.append(sigma)

        denom, unnorm_mu = 0, 0
        for i in range(self.num_primitives):
            denom += weights[i] / sigmas[i]
            unnorm_mu += weights[i] / sigmas[i] * mus[i]

        mean = unnorm_mu / denom
        scale_tril = th.diag_embed(1 / denom)
        return distributions.MultivariateNormal(mean, scale_tril=scale_tril).sample()



        # h = self.mcp_shared_net(features)
        # for i in range(self.num_primitives):
        #     g = self.goal_encoders[i](goal)
        #     s = self.state_encoders[i](features)
        #     weights.append(self.gate[i](th.cat((s, g), 1)))
        #     output = self.primitives[i](h)
        #     mus.append(output[:self.param_size])
        #     sigmas.append(output[self.param_size:])
        #
        # denom, unnorm_mu = 0, 0
        #
        # for i in range(self.num_primitives):
        #     denom += weights[i] / sigmas[i]
        #     unnorm_mu += weights[i] / sigmas[i] * mus[i]
        #
        # mean = unnorm_mu / denom
        # scale_tril = th.diag_embed(1 / denom)
        # return distributions.MultivariateNormal(mean, scale_tril=scale_tril).sample()

    def forward_critic(self, state: th.Tensor, goal: th.Tensor) -> th.Tensor:
        value_input = th.cat((state, goal), -1)
        value = self.value_net(value_input)
        return value


    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        state = features['state']
        goal = features['goal']
        return self.forward_actor(state, goal), self.forward_critic(state, goal)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        log('Using our CustomActorCriticPolicy!')

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomHiddenLayers(self.features_dim)
