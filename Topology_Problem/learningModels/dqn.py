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

import tensorflow as tf
import tensorflow.contrib.layers as layers
import os
# os.environ['TF_CPP_MIN_VLOG_LEVEL']='4'
import multiprocessing
import numpy as np
import threading
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=164)
np.set_printoptions(formatter={'float_kind': lambda x: "%8.3f" % x})
import multiprocessing as mp
import os
import random
import scipy.signal
import tensorflow as tf
import tensorflow.contrib.slim as slim
import timeit

from datetime import datetime
# from memory_profiler import profile
from random import choice
from time import time
from time import sleep

from tensorflow.python.client import timeline
from tensorflow.python.profiler import model_analyzer


from ..config import *

if TOPOLOGY == 'VL2':
    from ..config import Da, Di
elif TOPOLOGY == 'FATTREE':
    from ..config import K
elif TOPOLOGY == 'JELLYFISH':
    pass

# I/O values
tf.app.flags.DEFINE_string("model_path", "nr/data-train/",
                           "Directory to write the model")

tf.app.flags.DEFINE_string("log_dir", "nr/data-logs/", "Directory to logs.")

tf.app.flags.DEFINE_string("sim_out_dir", "eval",
                           "Directory to save the simulation")

tf.app.flags.DEFINE_boolean("load_model", LOAD_MODEL, "Restore the model")

tf.app.flags.DEFINE_boolean("clipping", CLIPPING, "Enable clipping or not?")

tf.app.flags.DEFINE_float("clipping_value", CLIPPING_VALUE, "Clipping Value")

# Flags governing reward
tf.app.flags.DEFINE_float("gamma", GAMMA, "Gamma parameter.")

tf.app.flags.DEFINE_float("max_reward", MAX_REWARD,
                          "Maximum reward for a link.")
# Iteration parameters
tf.app.flags.DEFINE_float("period", float(TIME_STEP),
                          "Time between state changes.")

tf.app.flags.DEFINE_integer("num_steps", NUM_VALID_STEPS,
                            "Number of episode steps to be considered valid.")

tf.app.flags.DEFINE_integer("num_workers", NUM_WORKERS,
                            "Number of concurrent workers.")

tf.app.flags.DEFINE_integer("max_episode_length",
                            int(SIMULATION_TIME / TIME_STEP),
                            "Max number of steps in episode.")

# Flags governing state/policy
tf.app.flags.DEFINE_integer("nt_size", NT_SIZE, "Total size of the nt state.")

tf.app.flags.DEFINE_integer("tm_size", TM_SIZE, "Total size of the tm state.")

tf.app.flags.DEFINE_integer("a_size", A_SIZE,
                            "Total size of the action space.")

tf.app.flags.DEFINE_integer("k_acts", K_ACTIONS,
                            "Number of actions allowed in given step.")

# Flags governing learning rate
tf.app.flags.DEFINE_float("learning_rate", LEARNING_RATE,
                          "Initial learning rate.")

tf.app.flags.DEFINE_float("learning_rate_decay_factor",
                          LEARNING_RATE_DECAY_FACTOR,
                          "Learning rate decay factor.")

tf.app.flags.DEFINE_integer("num_epochs_per_decay", NUM_EPOCHS_PER_DECAY,
                            "Number of epochs er decay.")

FLAGS = tf.app.flags.FLAGS

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


# As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
class DQNPolicy:
    def __init__(self, model_path, tm_size, a_size, scope):
        # Setup
        self.model_path = model_path
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            with tf.variable_scope(scope):
                self.inputs_tm = tf.placeholder(shape=[None, (tm_size**2) * 2],
                                dtype=tf.float32,
                                name='in_tm')
                self.tmIn = tf.reshape(self.inputs_tm,
                                       shape=[-1, tm_size, tm_size, 2],
                                       name='reshape_tm')  # [?, 16, 16, 2]
                self.trafficMatrixLayers = []
                for layerNum in range(0, len(RL_MODEL)):
                    if layerNum == 0:
                        inputs = self.tmIn
                    else:
                        inputs = self.trafficMatrixLayers[-1]

                    layer = RL_MODEL[layerNum]

                    currentLayer = slim.conv2d(
                        activation_fn=tf.nn.elu,
                        inputs=inputs,
                        num_outputs=layer[0],
                        weights_initializer=tf.contrib.layers.
                        xavier_initializer(),
                        kernel_size=[layer[1], layer[2]],
                        stride=[layer[3], layer[4]],
                        padding='VALID',
                        scope=('conv' + str(layerNum)))

                    self.trafficMatrixLayers.append(currentLayer)

                hidden = slim.fully_connected(slim.flatten(
                    self.trafficMatrixLayers[-1]),
                                              256,
                                              activation_fn=tf.nn.elu,
                                              scope='fc0')  # [?, 256]
                    

                self.policy = tf.contrib.layers.fully_connected(
                    hidden,
                    a_size,
                    activation_fn=tf.nn.softmax,
                    weights_initializer=normalized_columns_initializer(0.01),
                    biases_initializer=None)  # [?, a_size] -> [?, 28]

                self.value = tf.contrib.layers.fully_connected(
                    hidden,
                    FLAGS.k_acts,
                    activation_fn=None,
                    weights_initializer=normalized_columns_initializer(1.0),
                    biases_initializer=None)  # [?, k_acts] -> [?, 6]

                if FLAGS.clipping:
                    self.policy = tf.clip_by_value(self.policy, 1e-10,
                                                tf.reduce_max(self.policy) + 1)
                    self.value = tf.clip_by_value(self.value, 1e-10,
                                                tf.reduce_max(self.value) + 1)
                
            # Load
            tf.train.Saver().restore(self.sess, self.model_path)

    def predict_q(self, inputs_tm):
        with self.graph.as_default():
            # Inputs
            feed_dict = {}
            feed_dict[self.inputs_tm] = inputs_tm

            # Q values
            qs = self.sess.run(self.policy, feed_dict=feed_dict)

            return qs
    
    def predict(self, inputs_tm):
        with self.graph.as_default():
            # Inputs
            feed_dict = {}
            feed_dict[self.inputs_tm] = inputs_tm

            # Q values
            a_dist = self.sess.run(self.policy, feed_dict=feed_dict)

            #action
            ret = []
            for a in a_dist:
                temp_selects = [_ for _ in range(len(a))]
                temp_selects = np.random.choice(temp_selects,
                                                size=FLAGS.k_acts,
                                                replace=False,
                                                p=a)
                temp_selects = temp_selects.tolist()
                ret.append(temp_selects)
            return ret


# class DQNPolicy:
#     def __init__(self, env, model_path):
#         # Setup
#         self.env = env
#         self.model_path = model_path
#         self.num_actions = env.action_space.n #6
#         self.input_shape = env.observation_space.shape
#         self.graph = tf.Graph()
#         self.sess = tf.Session(graph=self.graph)

#         with self.graph.as_default():
#             with tf.variable_scope('deepq'):
#                 # Observations
#                 self.imgs = tf.placeholder(tf.uint8, [None] + list(self.input_shape), name='observation')

#                 # Randomness
#                 self.stochastic_ph = tf.placeholder(tf.bool, (), name='stochastic')
#                 self.update_eps_ph = tf.placeholder(tf.float32, (), name='update_eps')
#                 eps = tf.get_variable('eps', (), initializer=tf.constant_initializer(0))

#                 # Q-function
#                 with tf.variable_scope('q_func'):
#                     # Normalization
#                     out = tf.cast(self.imgs, tf.float32) / 255.0

#                     # Convolutions
#                     with tf.variable_scope('convnet'):
#                         out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
#                         out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
#                         out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)

#                     # Flatten
#                     conv_out = layers.flatten(out)

#                     # Fully connected
#                     with tf.variable_scope('action_value'):
#                         value_out = layers.fully_connected(conv_out, num_outputs=512, activation_fn=None)
#                         value_out = tf.nn.relu(value_out)
#                         value_out = layers.fully_connected(value_out, num_outputs=self.num_actions, activation_fn=None)

#                 # Q values
#                 self.qs = value_out

#                 # Deterministic actions
#                 deterministic_actions = tf.argmax(self.qs, axis=1)

#                 # Stochastic actions
#                 batch_size = tf.shape(self.imgs)[0]
#                 random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions, dtype=tf.int64)
#                 chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
#                 stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

#                 # Output actions
#                 self.output_actions = tf.cond(self.stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
#                 self.update_eps_expr = eps.assign(tf.cond(self.update_eps_ph >= 0, lambda: self.update_eps_ph, lambda: eps))

#             # Load
#             tf.train.Saver().restore(self.sess, self.model_path)

#     def predict_q(self, imgs):
#         with self.graph.as_default():
#             # Inputs
#             feed_dict = {}
#             feed_dict[self.imgs] = imgs
#             feed_dict[self.update_eps_ph] = -1.0
#             feed_dict[self.stochastic_ph] = False

#             # Q values
#             qs = self.sess.run(self.qs, feed_dict=feed_dict)

#             return qs
    
#     def predict(self, imgs):
#         with self.graph.as_default():
#             # Inputs
#             feed_dict = {}
#             feed_dict[self.imgs] = imgs
#             feed_dict[self.update_eps_ph] = -1.0
#             feed_dict[self.stochastic_ph] = False

#             # Action
#             acts = self.sess.run(self.output_actions, feed_dict=feed_dict)

#             # Updates
#             self.sess.run(self.update_eps_expr, feed_dict=feed_dict)

#             return acts