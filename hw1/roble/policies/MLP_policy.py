import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from hw1.roble.infrastructure import pytorch_util as ptu
from hw1.roble.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        if self._discrete:
            self._logits_na = ptu.build_mlp(input_size=self._ob_dim,
                                           output_size=self._ac_dim,
                                           params=self._network)
            self._logits_na.to(ptu.device)
            self._mean_net = None
            self._logstd = None
            self._optimizer = optim.Adam(self._logits_na.parameters(),
                                        self._learning_rate)
        else:
            self._logits_na = None
            self._mean_net = ptu.build_mlp(input_size=self._ob_dim,
                                      output_size=self._ac_dim,
                                      params=self._network)
            self._mean_net.to(ptu.device)

            if self._deterministic:
                self._optimizer = optim.Adam(
                    itertools.chain(self._mean_net.parameters()),
                    self._learning_rate
                )
            else:
                self._logstd = nn.Parameter(
                    torch.zeros(self._ac_dim, dtype=torch.float32, device=ptu.device)
                )
                self._logstd.to(ptu.device)
                self._optimizer = optim.Adam(
                    itertools.chain([self._logstd], self._mean_net.parameters()),
                    self._learning_rate
                )

        if self._nn_baseline:
            self._baseline = ptu.build_mlp(
                input_size=self._ob_dim,
                output_size=1,
                params=self._network
            )
            self._baseline.to(ptu.device)
            self._baseline_optimizer = optim.Adam(
                self._baseline.parameters(),
                self._learning_rate,
            )
        else:
            self._baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO:
        # # Provide the logic to produce an action from the policy
        obs = torch.FloatTensor(obs).unsqueeze(0).to(ptu.device)
        # action = self(obs).rsample()
        action = self(obs).mean #.rsample()
        # action = self(obs).rsample()
        action = ptu.to_numpy(action)
        return action


    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self._discrete:
            logits = self._logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            if self._deterministic:
                ##  TODO output for a deterministic policy
                action_distribution = self._mean_net(observation)
            else:
                
                ##  TODO output for a stochastic policy
                mean = self._mean_net(observation)
                std = torch.exp(self._logstd)
                dist = distributions.Normal(mean, std)
                action_distribution = dist #.rsample()
                # action_distribution = self._mean_net(observation)
        return action_distribution
    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        # pass
        raise NotImplementedError

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loss = nn.MSELoss()

    def update(
        self, observations, actions,
        adv_n=None, acs_labels_na=None, qvals=None
        ):
        
        # TODO: update the policy and return the loss
        observations = torch.FloatTensor(observations).to(ptu.device)
        actions = torch.FloatTensor(actions).to(ptu.device)
        dist = self(observations)
        # print('observations', observations.shape)
        # print('actions', actions.shape)

        # pred_actions = dist.rsample()
        # loss = self._loss(pred_actions, actions)
        loss = -dist.log_prob(actions).mean() #(pred_actions - actions/).pow(2).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_idm(
        self, observations, actions, next_observations,
        adv_n=None, acs_labels_na=None, qvals=None
        ):

        observations = torch.FloatTensor(observations).to(ptu.device)
        actions = torch.FloatTensor(actions).to(ptu.device)
        next_observations = torch.FloatTensor(next_observations).to(ptu.device)

        # TODO: Create the full input to the IDM model (hint: it's not the same as the actor as it takes both obs and next_obs)
        full_input = torch.cat((observations, next_observations), dim=1)
        pred_actions = self(full_input)
        loss = self._loss(pred_actions, actions)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return {
            'Training Loss IDM': ptu.to_numpy(loss),
        }