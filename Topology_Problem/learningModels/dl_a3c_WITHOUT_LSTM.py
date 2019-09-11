"""
    Code from: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2#.ckt96znsk
    @author Arthur Juliani

    Modified by:
    @author Chris Streiffer
"""
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
from memory_profiler import profile
from random import choice
from time import time
from time import sleep

from tensorflow.python.client import timeline
from tensorflow.python.profiler import model_analyzer

from config import A_SIZE
from config import CHECKPOINT_ITERATIONS
from config import CLIPPING
from config import CLIPPING_VALUE
from config import GAMMA
from config import K_ACTIONS
from config import LEARNING_RATE
from config import LEARNING_RATE_DECAY_FACTOR
from config import LOAD_MODEL
from config import MAX_REWARD
from config import MEMORY_STATISTICS
from config import NODE2VEC
from config import NT_SIZE
from config import NUM_DIMENSIONS
from config import NUM_EPOCHS_PER_DECAY
from config import NUM_VALID_STEPS
from config import NUM_WORKERS
from config import RESULTS_FILE
from config import RL_MODEL
from config import SIMULATION_TIME
from config import SPARSE_INPUT
from config import TESTING
from config import TF_STATS_FOLDER
from config import TIME_STEP
from config import TIME_STEP
from config import TEST_TIMING_INFO
from config import TEST_TIMING_FILE
from config import TM_SIZE
from config import TOPOLOGY
from config import TOPOLOGY_INCLUDED
from config import DIRECTION

if TOPOLOGY == 'VL2':
    from config import Da, Di
elif TOPOLOGY == 'FATTREE':
    from config import K
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


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount_loc(x, gamma, k, num_steps):
    xarr = np.array(x)
    vals = scipy.signal.lfilter([1], [1, -gamma],
                                np.fliplr(xarr.transpose()),
                                axis=1)
    return np.fliplr(vals).transpose()


# Discounting function used to calculate discounted returns.
def discount_tot(x, gamma, k, num_steps):
    return np.repeat(
        scipy.signal.lfilter([1], [1, -gamma], x[::-k], axis=0)[::-1], k)


#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def calc_dist(dist, temp_actions):
    for i in range(len(temp_actions)):
        dist[i] = 0.0 if temp_actions[i] == 1 else dist[i]
    return dist / np.sum(dist, dtype=np.float32)


class AC_Network():
    def __init__(self, nt_size, tm_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            Input and visual encoding layers
            if NODE2VEC:
                self.inputs_tm = tf.placeholder(
                    shape=[None, (tm_size * NUM_DIMENSIONS) * 2],
                    dtype=tf.float32,
                    name='in_tm')
                self.tmIn = tf.reshape(self.inputs_tm,
                                       shape=[-1, tm_size, NUM_DIMENSIONS, 2],
                                       name='reshape_tm')
            elif SPARSE_INPUT:
                ## HACK for TF Bug.
                self.tensor_shapes = tf.placeholder(shape=[3], dtype=np.int64)
                self.flow_sizes = tf.sparse.placeholder(
                    shape=self.tensor_shapes, dtype=np.float32)

                flow_sizes_ = tf.sparse.reshape(self.flow_sizes,
                                                shape=[-1, (tm_size**2)],
                                                name='flow_sizes_reshaped')

                self.flow_times = tf.sparse.placeholder(
                    shape=self.tensor_shapes, dtype=np.float32)

                flow_times_ = tf.sparse.reshape(self.flow_times,
                                                shape=[-1, (tm_size**2)],
                                                name='flow_times_reshaped')
            else:
                self.inputs_tm = tf.placeholder(shape=[None, (tm_size**2) * 2],
                                                dtype=tf.float32,
                                                name='in_tm')
                self.tmIn = tf.reshape(self.inputs_tm,
                                       shape=[-1, tm_size, tm_size, 2],
                                       name='reshape_tm')  # [?, 16, 16, 2]

            self.trafficMatrixLayers = []

            if SPARSE_INPUT:
                # pass
                HIDDEN_LAYER_0 = tf.Variable(initial_value=tf.random_normal(
                    shape=[tm_size**2, tm_size**2 / 2], stddev=.05),
                                             name='hidden_layer_0',
                                             dtype=np.float32)

                flow_sizes = tf.sparse.matmul(flow_sizes_, HIDDEN_LAYER_0)
                flow_times = tf.sparse.matmul(flow_times_, HIDDEN_LAYER_0)

                self.combined_input = tf.concat([flow_times, flow_sizes], 1)
                self.trafficMatrixLayers.append(self.combined_input)
            else:
                for layerNum in xrange(0, len(RL_MODEL)):
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

            if TOPOLOGY_INCLUDED:
                self.inputs_nt = tf.placeholder(shape=[None, nt_size**2],
                                                dtype=tf.float32,
                                                name='in_nt')

                self.ntIn = tf.reshape(self.inputs_nt,
                                       shape=[-1, nt_size, nt_size, 1],
                                       name='reshape_nt')  # [?, 8, 8, 1]

                self.networkInputLayers = []

                for layerNum in xrange(0, len(RL_MODEL)):
                    if layerNum == 0:
                        inputs = self.ntIn
                    else:
                        inputs = self.networkInputLayers[-1]

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
                        scope=('conv' +
                               str(layerNum + len(self.trafficMatrixLayers))))

                    self.networkInputLayers.append(currentLayer)

                combined_input = tf.concat([
                    slim.flatten(self.trafficMatrixLayers[-1]),
                    slim.flatten(self.networkInputLayers[-1])
                ],
                                           1,
                                           name='concat')
                # [?, 832]

                hidden = slim.fully_connected(combined_input,
                                              256,
                                              activation_fn=tf.nn.elu,
                                              scope='fc0')  # [?, 256]
            else:
                hidden = slim.fully_connected(slim.flatten(
                    self.trafficMatrixLayers[-1]),
                                              256,
                                              activation_fn=tf.nn.elu,
                                              scope='fc0')  # [?, 256]

            #Output layers for policy and value estimations
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

            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(
                    shape=[None, FLAGS.k_acts],
                    dtype=tf.int32)  # [?, k_acts] -> [?, 6]
                self.actions_onehot = tf.one_hot(
                    self.actions, a_size,
                    dtype=tf.float32)  # [?, k_acts, a_size] -> [?, 6, 28]
                self.target_v = tf.placeholder(
                    shape=[None, FLAGS.k_acts],
                    dtype=tf.float32)  # [?, 6] (Discounted Rewards)

                # Positive reward
                self.advantages = tf.placeholder(
                    shape=[None, FLAGS.k_acts], dtype=tf.float32
                )  # [?, 6] (Generalized Advantage Estimation)
                self.policy_expand = tf.expand_dims(self.policy,
                                                    1)  # [?, 1, 28]

                self.policy_mul = tf.tile(self.policy_expand,
                                          [1, FLAGS.k_acts, 1])  # [?, 6, 28]
                self.responsible_outputs = tf.reduce_sum(
                    self.policy_mul * self.actions_onehot, [2])  # [?, 6]

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(
                    tf.square(
                        tf.reshape(self.target_v, [-1]) -
                        tf.reshape(self.value, [-1])))
                self.entropy = -tf.reduce_sum(
                    self.policy * tf.log(self.policy))

                # Clipping so that NANs don't appear
                if FLAGS.clipping:
                    self.responsible_outputs = tf.clip_by_value(
                        self.responsible_outputs,
                        tf.reduce_min(self.responsible_outputs) +
                        FLAGS.clipping_value,
                        tf.reduce_max(self.responsible_outputs) +
                        FLAGS.clipping_value)

                self.policy_value = tf.log(
                    self.responsible_outputs) * self.advantages
                self.policy_loss = -tf.reduce_sum(self.policy_value)

                # =================
                # self.entropy becomes NaN.
                # =================
                # self.loss = 0.5 * self.value_loss + self.policy_loss - 0.01*self.entropy

                self.loss = 0.5 * self.value_loss + self.policy_loss

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(
                    self.gradients, 40.0)

                #Apply local gradients to global network
                global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(
                    zip(grads, global_vars))
            





class Worker():
    def __init__(self, simulator, name, nt_size, tm_size, a_size, trainer,
                 model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)

        self.episode_rewards = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(TF_STATS_FOLDER +
                                                    "train_" +
                                                    str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(nt_size, tm_size, a_size, self.name,
                                   trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        #End Doom set-up
        self.env = simulator

        saver = tf.train.Saver(max_to_keep=100)

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True


        with tf.Session(config=config) as sess:
            coord = tf.train.Coordinator()
            if FLAGS.load_model:
                ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading Model: %s' % ckpt.model_checkpoint_path)

        # Setup logging

        # self.state_file = open(os.path.join(FLAGS.log_dir, 'state-%02d-out.txt' % self.number), 'w')
        # self.reward_file = open(os.path.join(FLAGS.log_dir, 'reward-%02d-out.txt' % self.number), 'w')
        # self.train_file_loc = open(os.path.join(FLAGS.log_dir, 'train-loc-%02d-out.txt' % self.number), 'w')
        # self.train_file_glo = open(os.path.join(FLAGS.log_dir, 'train-glo-%02d-out.txt' % self.number), 'w')

    def predict_q(self, imgs):
        with self.graph.as_default():
            # Inputs
            feed_dict = {}
            feed_dict[self.imgs] = imgs
            feed_dict[self.update_eps_ph] = -1.0
            feed_dict[self.stochastic_ph] = False

            # Q values
            qs = self.sess.run(self.qs, feed_dict=feed_dict)

            return qs
    
    def predict(self, imgs):
        with self.graph.as_default():
            # Inputs
            feed_dict = {}
            feed_dict[self.imgs] = imgs
            feed_dict[self.update_eps_ph] = -1.0
            feed_dict[self.stochastic_ph] = False

            # Action
            acts = self.sess.run(self.output_actions, feed_dict=feed_dict)

            # Updates
            self.sess.run(self.update_eps_expr, feed_dict=feed_dict)

            return acts


    def parse_sparse_matrix(self, sparse_matrix_list):
        flow_times = [epoch[0] for epoch in sparse_matrix_list]
        flow_sizes = [epoch[1] for epoch in sparse_matrix_list]

        parsed_flow_times = [
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32)
        ]
        parsed_flow_sizes = [
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32)
        ]

        for i in xrange(len(sparse_matrix_list)):
            # FLOW-TIMES
            temp_flow_times = flow_times[i].nonzero()
            index = np.zeros(len(temp_flow_times[0]), dtype=np.int64)
            index[:] = i

            temp_flow_times = np.dstack(
                (index, temp_flow_times[0], temp_flow_times[1]))[0]

            if temp_flow_times.size != 0:
                if parsed_flow_times[0].size == 0:
                    parsed_flow_times[0] = temp_flow_times
                else:
                    parsed_flow_times[0] = np.vstack(
                        (parsed_flow_times[0], temp_flow_times))

                parsed_flow_times[1] = np.append(parsed_flow_times[1],
                                                 flow_times[i].values())

            # FLOW-SIZES
            temp_flow_sizes = flow_sizes[i].nonzero()
            index = np.zeros(len(temp_flow_sizes[0]), dtype=np.int64)
            index[:] = i

            temp_flow_sizes = np.dstack(
                (index, temp_flow_sizes[0], temp_flow_sizes[1]))[0]

            if temp_flow_sizes.size != 0:
                if parsed_flow_sizes[0].size == 0:
                    parsed_flow_sizes[0] = temp_flow_sizes
                else:
                    parsed_flow_sizes[0] = np.vstack(
                        (parsed_flow_sizes[0], temp_flow_sizes))

                parsed_flow_sizes[1] = np.append(parsed_flow_sizes[1],
                                                 flow_sizes[i].values())

        if parsed_flow_times[0].size == 0:
            parsed_flow_times = [
                np.array([[0, 0, 0]], dtype=np.int64),
                np.array([0], dtype=np.float32)
            ]
        if parsed_flow_sizes[0].size == 0:
            parsed_flow_sizes = [
                np.array([[0, 0, 0]], dtype=np.int64),
                np.array([0], dtype=np.float32)
            ]

        return (parsed_flow_times, parsed_flow_sizes)

    def train(self, rollout, sess, gamma, bootstrap_value, step, disount_fn):
        rollout = np.array(rollout)

        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        self.rewards_plus = np.concatenate(
            (np.vstack(rewards), bootstrap_value), axis=0)

        discounted_rewards = disount_fn(self.rewards_plus, gamma, FLAGS.k_acts,
                                        FLAGS.num_steps + 1)[:-1]
        self.value_plus = np.concatenate((np.vstack(values), bootstrap_value),
                                         axis=0)

        advantages = np.vstack(rewards) + gamma * (self.value_plus[1:, :] -
                                                   self.value_plus[:-1, :])
        advantages = disount_fn(advantages, gamma, FLAGS.k_acts,
                                FLAGS.num_steps)
        advantages = np.clip(advantages, 0.0, FLAGS.max_reward)

        # advantages = rewards

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # rnn_state = self.local_AC.state_init

        if TOPOLOGY_INCLUDED:
            feed_dict = {
                self.local_AC.target_v:
                np.vstack(discounted_rewards),
                self.local_AC.inputs_nt:
                np.vstack([obs[0] for obs in observations]),
                self.local_AC.inputs_tm:
                np.vstack([obs[1] for obs in observations]),
                self.local_AC.actions:
                np.vstack(actions),
                self.local_AC.advantages:
                np.vstack(advantages)
            }
        else:
            if SPARSE_INPUT:
                flow_times, flow_sizes = self.parse_sparse_matrix(observations)
                flow_times_indices, flow_times_values = flow_times
                flow_sizes_indices, flow_sizes_values = flow_sizes

                feed_dict = {
                    self.local_AC.target_v:
                    np.vstack(discounted_rewards),
                    self.local_AC.tensor_shapes:
                    [len(values), FLAGS.tm_size, FLAGS.tm_size],
                    self.local_AC.flow_times:
                    (flow_times_indices, flow_times_values),
                    self.local_AC.flow_sizes:
                    (flow_sizes_indices, flow_times_values),
                    self.local_AC.actions:
                    np.vstack(actions),
                    self.local_AC.advantages:
                    np.vstack(advantages)
                }
            else:
                feed_dict = {
                    self.local_AC.target_v: np.vstack(discounted_rewards),
                    self.local_AC.inputs_tm: np.vstack(observations),
                    self.local_AC.actions: np.vstack(actions),
                    self.local_AC.advantages: np.vstack(advantages)
                }

        val, pol, t_l, v_l, p_l, e_l, g_n, v_n, _, resp, adv, p, a = sess.run(
            [
                self.local_AC.value, self.local_AC.policy, self.local_AC.loss,
                self.local_AC.value_loss, self.local_AC.policy_loss,
                self.local_AC.entropy, self.local_AC.grad_norms,
                self.local_AC.var_norms, self.local_AC.apply_grads,
                self.local_AC.responsible_outputs, self.local_AC.advantages,
                self.local_AC.policy_value, self.local_AC.actions_onehot
            ],
            feed_dict=feed_dict)

        return v_l / len(rollout), p_l / len(rollout), e_l / len(
            rollout), g_n, v_n, t_l

    def work(self, max_episode_length, gamma, sess, coord, saver):
        # def pickTop_K(k, probs, collision_lst):
        #     def search_collision_lst(lst_of_lst, item):
        #         ret = []
        #         for lst in lst_of_lst:
        #             if item in lst:
        #                 ret += lst

        #         return list(set(ret))

        #     ret = []

        #     while len(ret) < k:
        #         index = np.argmax(probs)
        #         ret.append(index)

        #         collisions = search_collision_lst(collision_lst, index)
        #         probs[collisions] = 0

        #         if np.max(probs) == 0:
        #             break

        #     return ret

        # def collision():
        #     collision_dict_src = {}
        #     collision_dict_dst = {}
        #     index = 0

        #     for i in xrange(0, FLAGS.tm_size):
        #         for j in xrange(0, i):
        #             if i in collision_dict_src:
        #                 collision_dict_src[i].append(index)
        #             else:
        #                 collision_dict_src[i] = [index]

        #             if j in collision_dict_dst:
        #                 collision_dict_dst[j].append(index)
        #             else:
        #                 collision_dict_dst[j] = [index]

        #             index = index + 1

        #     if DIRECTION == 'UNI':
        #         for j in xrange(0, FLAGS.tm_size):
        #             for i in xrange(0, j):
        #                 if i in collision_dict_src:
        #                     collision_dict_src[i].append(index)
        #                 else:
        #                     collision_dict_src[i] = [index]

        #                 if j in collision_dict_dst:
        #                     collision_dict_dst[j].append(index)
        #                 else:
        #                     collision_dict_dst[j] = [index]

        #                 index = index + 1

        #     return (collision_dict_src.values() + collision_dict_dst.values())

        episode_count = sess.run(self.global_episodes)
        max_steps = FLAGS.max_episode_length
        # collision_lst = collision()
        # print collision_lst

        print "STARTED WORKING -- NAME: %s, COUNT: %d" % (self.name,
                                                          episode_count)

        # write_line(self.reward_file, 'INIT: time:%s' % (datetime.now()))

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                # self.action_file = open(os.path.join(FLAGS.log_dir, 'actions-%02d-out.txt' % self.number), 'w')
                sess.run(self.update_local_ops)
                loss_buffer = []
                episode_buffer_a = []
                episode_step_count = 0
                d = False
                d_inc = False

                self.env.send(('RESET', None))
                s = self.env.recv()

                # saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.ckpt')
                # print ("Saved Model")
                # return

                if s is None:
                    self.env.send(('END', None))
                    return

                self.env.send(('IS_EPISODE_FINISHED', None))
                while (self.env.recv() == False):
                    if TESTING and TEST_TIMING_INFO:
                        start_time = timeit.default_timer()
                    #Take an action using probabilities from policy network output.
                    temp_actions = np.zeros((1, FLAGS.a_size))
                    temp_selects = []

                    if TOPOLOGY_INCLUDED:
                        a_dist, v = sess.run(
                            [self.local_AC.policy, self.local_AC.value],
                            feed_dict={
                                self.local_AC.inputs_nt: [s[0]],
                                self.local_AC.inputs_tm: [s[1]]
                            })
                    else:
                        if SPARSE_INPUT:
                            flow_times, flow_sizes = self.parse_sparse_matrix(
                                [s])
                            flow_times_indices, flow_times_values = flow_times
                            flow_sizes_indices, flow_sizes_values = flow_sizes

                            a_dist, v = sess.run(
                                [self.local_AC.policy, self.local_AC.value],
                                feed_dict={
                                    self.local_AC.tensor_shapes:
                                    [1, FLAGS.tm_size, FLAGS.tm_size],
                                    self.local_AC.flow_times:
                                    (flow_times_indices, flow_times_values),
                                    self.local_AC.flow_sizes:
                                    (flow_sizes_indices, flow_sizes_values)
                                })
                        else:
                            a_dist, v = sess.run(
                                [self.local_AC.policy, self.local_AC.value],
                                feed_dict={self.local_AC.inputs_tm: [s]})
                    # print "Initial Distribution: " + str(a_dist)
                    a_dist = a_dist[0]

                    # if TESTING:
                    #     temp_selects = pickTop_K(FLAGS.k_acts,
                    #                              np.array(a_dist),
                    #                              collision_lst)

                    # print temp_selects
                    # print len(temp_selects)

                    # if not TESTING:
                    temp_selects = [_ for _ in xrange(len(a_dist))]
                    temp_selects = np.random.choice(temp_selects,
                                                    size=FLAGS.k_acts,
                                                    replace=False,
                                                    p=a_dist)
                    temp_selects = temp_selects.tolist()

                    for a_sel in temp_selects:
                        temp_actions[0][a_sel] = 1

                    if TOPOLOGY_INCLUDED:
                        self.env.send(('TOPO_UPDATE', temp_actions))
                        self.env.recv()

                    if TESTING and TEST_TIMING_INFO:
                        end_time = timeit.default_timer()

                        f = open(TEST_TIMING_FILE, 'a')
                        f.write(str(end_time - start_time))
                        f.write('\n')
                        f.close()

                    self.env.send(('OPTICAL_LINKS', temp_selects))
                    s1, r, d = self.env.recv()

                    self.episode_rewards.append(np.sum(r[0]))
                    self.episode_mean_values.append(np.mean(v[0]))

                    episode_buffer_a.append([s, temp_selects, r[0], s1, d, v])

                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.

                    if TESTING == False:
                        if len(episode_buffer_a
                               ) == FLAGS.num_steps or d == True:
                            if TOPOLOGY_INCLUDED:
                                v = sess.run(self.local_AC.value,
                                             feed_dict={
                                                 self.local_AC.inputs_nt:
                                                 [s1[0]],
                                                 self.local_AC.inputs_tm:
                                                 [s1[1]]
                                             })
                            else:
                                if SPARSE_INPUT:
                                    flow_times, flow_sizes = self.parse_sparse_matrix(
                                        [s])
                                    flow_times_indices, flow_time_values = flow_times
                                    flow_sizes_indices, flow_sizes_values = flow_sizes

                                    v = sess.run(
                                        self.local_AC.value,
                                        feed_dict={
                                            self.local_AC.tensor_shapes:
                                            [1, FLAGS.tm_size, FLAGS.tm_size],
                                            self.local_AC.flow_times:
                                            (flow_times_indices,
                                             flow_times_values),
                                            self.local_AC.flow_sizes:
                                            (flow_sizes_indices,
                                             flow_sizes_values)
                                        })
                                else:
                                    v = sess.run(self.local_AC.value,
                                                 feed_dict={
                                                     self.local_AC.inputs_tm:
                                                     [s1]
                                                 })
                            # Since we don't know what the true final return is, we "bootstrap" from our current
                            # value estimation.
                            v_l, p_l, e_l, g_n, v_n, totalLoss = self.train(
                                episode_buffer_a, sess, gamma, v,
                                episode_step_count, discount_loc)

                            if self.name == 'worker_0':
                                loss_buffer.append(totalLoss)

                            episode_buffer_a = []
                            sess.run(self.update_local_ops)

                    # Update the state
                    s = s1

                    # Check if we're done here
                    if d == True or episode_step_count == max_steps:
                        break

                    self.env.send(('IS_EPISODE_FINISHED', None))

                if self.name == 'worker_0':
                    sess.run(self.increment)
                    f = (lambda x, y: x + y)

                    if TESTING == False:
                        fileName = ("CNN_" + str(FLAGS.max_episode_length) +
                                    "_" + str(FLAGS.period) + "_")
                        results = open(RESULTS_FILE, "a")
                        results.write("Episode Number: " + str(episode_count) +
                                      " ")
                        results.write("Time: " + str(datetime.now().time()) +
                                      " ")
                        results.write(
                            "Episode Loss: " +
                            str(reduce(f, loss_buffer) / len(loss_buffer)) +
                            '\n')
                        results.close()

                    # if (episode_count % 10 == 0):
                    #     os.system('spd-say "done done"')

                episode_count += 1
                if (not TESTING and \
                   episode_count % CHECKPOINT_ITERATIONS == 0 and \
                   self.name == 'worker_0'):
                    saver.save(
                        sess, self.model_path + '/worker-' + str(self.name) +
                        '-model-' + str(episode_count) + '.ckpt')

                    if not TESTING:
                        total_reward = np.sum(self.episode_rewards)
                        mean_value = np.mean(self.episode_mean_values)

                        # print self.episode_rewards
                        self.episode_rewards = []
                        self.episode_mean_values = []

                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Reward',
                                          simple_value=float(total_reward))
                        summary.value.add(tag='Perf/Length',
                                          simple_value=episode_step_count)
                        summary.value.add(tag='Perf/Value',
                                          simple_value=float(mean_value))
                        summary.value.add(tag='Losses/Value Loss',
                                          simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss',
                                          simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy',
                                          simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm',
                                          simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm',
                                          simple_value=float(v_n))
                        self.summary_writer.add_summary(summary, episode_count)
                        self.summary_writer.flush()

class DecisionTreeSimulator(object):
    def __init__(self, simulator, network_simulator):
        self.simulator = simulator
        self.network_simulator = network_simulator

    def load(self):
        # LOAD DECISION TREE FROM DISK

        return decision_tree

    def run(self):    
        monitor = threading.Event()
        sim = self.simulator(monitor,
                             'sim-%s-%02d' % (FLAGS.sim_out_dir, i),
                             worker_ID)

        # Initialize the network simulator
        out_dir = "sim-%s-%02d/outputs" % (FLAGS.sim_out_dir, i)
        nm = self.network_simulator(sim, monitor)
        nm.set_period(FLAGS.period)
        nm.set_actions(FLAGS.k_acts, FLAGS.a_size)
        nm.set_max_reward(FLAGS.max_reward)
        nm.init()

        state = nm.reset()
        decision_tree = self.load()

        while nm.is_episode_finished() == False:
            actions = self.decision_tree.predict(state)
            nm.step(actions)



class RLModelSimulator(object):
    def __init__(self, simulator, network_simulator):
        self.simulator = simulator
        self.network_simulator = network_simulator
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)

    # # @profile
    def generate_envs(self):
        envs = []

        def generate_enviornment(worker_ID, pipe):
            monitor = threading.Event()
            sim = self.simulator(monitor,
                                 'sim-%s-%02d' % (FLAGS.sim_out_dir, i),
                                 worker_ID)

            # Initialize the network simulator
            out_dir = "sim-%s-%02d/outputs" % (FLAGS.sim_out_dir, i)
            nm = self.network_simulator(sim, monitor)
            nm.set_period(FLAGS.period)
            nm.set_actions(FLAGS.k_acts, FLAGS.a_size)
            nm.set_max_reward(FLAGS.max_reward)
            nm.init()

            while True:
                command, data = pipe.recv()

                if command == 'RESET':
                    pipe.send(nm.reset())
                elif command == 'IS_EPISODE_FINISHED':
                    pipe.send(nm.is_episode_finished())
                elif command == 'OPTICAL_LINKS':
                    pipe.send(nm.step(data))
                elif command == 'TOPO_UPDATE':
                    pipe.send(nm.network_topology_update_step(data))
                elif command == 'END':
                    # clear_checkpoints(FLAGS.model_path)
                    print "Ending Worker!"
                    break

        for i in range(FLAGS.num_workers):
            p1, p2 = mp.Pipe()
            p = mp.Process(target=generate_enviornment, args=(i, p2))
            p.start()
            envs.append(p1)

        return envs

    # @memory_profiler
    def run(self):

        tf.reset_default_graph()

        if not os.path.exists(FLAGS.model_path):
            os.makedirs(FLAGS.model_path)

        global_episodes = tf.get_variable(
            'global_episodes', [],
            initializer=tf.constant_initializer(0.0),
            dtype=tf.float32,
            trainable=False)
        num_batches_per_epoch = (FLAGS.max_episode_length / FLAGS.num_steps)
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
        self._lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                              global_episodes,
                                              decay_steps,
                                              FLAGS.learning_rate_decay_factor,
                                              staircase=True)

        trainer = tf.train.AdamOptimizer(learning_rate=self._lr)
        master_network = AC_Network(FLAGS.nt_size, FLAGS.tm_size, FLAGS.a_size,
                                    'global', None)  # Generate global network
        workers = []

        res = self.generate_envs()

        # Create worker classes
        for i in range(FLAGS.num_workers):
            print (i)
            # Create the worker containing the simulator
            workers.append(
                Worker(res[i], i, FLAGS.nt_size, FLAGS.tm_size, FLAGS.a_size,
                       trainer, FLAGS.model_path, global_episodes))

        saver = tf.train.Saver(max_to_keep=100)

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True

        run_metadata = tf.RunMetadata()

        # r1, r2, r3 = model_analyzer.lib.BuildSplitableModel()

        with tf.Session(config=config) as sess:
            coord = tf.train.Coordinator()
            if FLAGS.load_model:
                ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading Model: %s' % ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer(),
                         options=tf.RunOptions(
                             trace_level=tf.RunOptions.FULL_TRACE),
                         run_metadata=run_metadata)
                # tf.profiler.ProfileOptionBuilder.add_step(0, run_metadata)

            # This is where the asynchronous magic happens.
            # Start the "work" process for each worker in a separate threat.
            worker_threads = []
            for worker in workers:
                worker_work = lambda: worker.work(
                    FLAGS.max_episode_length, FLAGS.gamma, sess, coord, saver)
                t = threading.Thread(target=(worker_work))
                t.start()
                sleep(0.05)
                worker_threads.append(t)
            coord.join(worker_threads)

        if not MEMORY_STATISTICS:
            return

        print "\n\n *** PARAMETERS AND SHAPES *** \n\n"
        # Print trainable variable parameter statistics to stdout.
        ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder

        param_stats = tf.profiler.profile(
            tf.get_default_graph(),
            options=ProfileOptionBuilder.trainable_variables_parameter())

        # Use code view to associate statistics with Python codes.
        opts = ProfileOptionBuilder(
            ProfileOptionBuilder.trainable_variables_parameter()
        ).with_node_names(
            show_name_regexes=['.*my_code1.py.*', '.*my_code2.py.*']).build()
        param_stats = tf.profiler.profile(tf.get_default_graph(),
                                          cmd='code',
                                          options=opts)

        print('total_params: %d\n' % param_stats.total_parameters)

        print "\n\n *** FLOATING POINT OPERATIONS *** \n\n"
        tf.profiler.profile(
            tf.get_default_graph(),
            options=tf.profiler.ProfileOptionBuilder.float_operation())

        print "\n\n *** MEMORY + TIMING INFORMATION *** \n\n"
        # Print to stdout an analysis of the memory usage and the timing information
        # broken down by python codes.
        ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
        opts = ProfileOptionBuilder(
            ProfileOptionBuilder.time_and_memory()).with_node_names(
                show_name_regexes=['.*my_code.py.*']).build()

        tf.profiler.profile(tf.get_default_graph(),
                            run_meta=run_metadata,
                            cmd='code',
                            options=opts)

        # Print to stdout an analysis of the memory usage and the timing information
        # broken down by operation types.
        tf.profiler.profile(
            tf.get_default_graph(),
            run_meta=run_metadata,
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.time_and_memory())

        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_01.json', 'w') as f:
            f.write(chrome_trace)