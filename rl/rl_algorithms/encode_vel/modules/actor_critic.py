# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from termcolor import cprint

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class HistEncoder(nn.Module):
    def __init__(self, num_encoder_obs, encoder_hidden_dims):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_encoder_obs, encoder_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[0], encoder_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[1], encoder_hidden_dims[2]),
        )

    def forward(self, obs_his):
        """
        Encodes proprioceptive observation history
        Input:
            obs_his: history of observation
        """
        return self.encoder(obs_his)


class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        fixed_std=False,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        # ---- Privileged information ----
        self.num_actor_input = num_actor_obs + 3
        self.num_critic_input = num_critic_obs

        self.history_len = kwargs['hist_encoder']['include_history_steps']
        self.num_encoder_input = num_actor_obs * self.history_len
        self.encoder_mlp_unit = kwargs['hist_encoder']['priv_mlp_units']

        # # History Encoder
        # self.hist_encoder = HistEncoder(self.num_encoder_input, self.encoder_mlp_unit)
        # cprint(f"Encoder MLP: {self.hist_encoder}", 'green', attrs=['bold'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_actor_input, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_critic_input, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        cprint(f"Actor MLP: {self.actor}", 'cyan', attrs=['bold'])
        cprint(f"Critic MLP: {self.critic}", 'cyan', attrs=['bold'])

        # History Encoder
        self.hist_encoder = HistEncoder(self.num_encoder_input, self.encoder_mlp_unit)
        cprint(f"Encoder MLP: {self.hist_encoder}", 'green', attrs=['bold'])

        # Action noise
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean*0. + std)

    # def act(self, observations, **kwargs):
    #     self.update_distribution(observations)
    #     return self.distribution.sample()

    def act(self, obs_dict, **kwargs):
        mean, std, _, _ = self._actor_critic(obs_dict)
        self.distribution = Normal(mean, mean*0. + std)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # def act_inference(self, observations):
    #     actions_mean = self.actor(observations)
    #     return actions_mean

    def act_inference(self, obs_dict):
        actions_mean, _, _, _ = self._actor_critic(obs_dict)
        return actions_mean

    # def evaluate(self, critic_observations, **kwargs):
    #     value = self.critic(critic_observations)
    #     return value

    def evaluate(self, obs_dict, **kwargs):
        _, _, value, _ = self._actor_critic(obs_dict)
        return value

    def predict_entrinsic(self, obs_dict, **kwargs):
        _, _, _, extrin = self._actor_critic(obs_dict)
        return extrin

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        obs_privileged = obs_dict['privileged_obs']
        obs_his = obs_dict['proprio_hist']

        # actor_obs = obs
        critic_obs = obs_privileged
        extrin_en = self.hist_encoder(obs_his) # extrinsic of encoder
        actor_obs = torch.cat([extrin_en, obs], dim=-1)  ## 45 + 3

        mu = self.actor(actor_obs)
        value = self.critic(critic_obs)
        sigma = self.std.to(mu.device)

        return mu, mu * 0 + sigma, value, extrin_en

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None