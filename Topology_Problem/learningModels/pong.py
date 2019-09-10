# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import gym

from .atari_wrappers_deprecated import wrap_dqn

def _get_our_paddle(obs_t):
    the_slice = obs_t[:,74]
    pad_color = np.full(the_slice.shape, 147)
    diff_to_paddle = np.absolute(the_slice - pad_color)
    return np.argmin(diff_to_paddle) + 3

def _get_enemy_paddle(obs_t):
    the_slice = obs_t[:,9]
    pad_color = np.full(the_slice.shape, 148)
    diff_to_paddle = np.absolute(the_slice - pad_color)
    return np.argmin(diff_to_paddle) + 3

def _get_ball(obs_t):
    if (np.max(obs_t) < 200):
        return np.array([0, 0])
    idx = np.argmax(obs_t)
    return np.unravel_index(idx, obs_t.shape)

def get_pong_env():
    return wrap_dqn(gym.make('PongNoFrameskip-v4'))

def get_pong_symbolic(obs):
    return obs
