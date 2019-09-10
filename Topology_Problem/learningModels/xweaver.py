import copy
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import datetime
from random import random, sample, uniform, choice

import numpy as np
import json
import cPickle as pickle

from config import ALGORITHM
from config import A_SIZE
from config import K_ACTIONS
from config import LEARNING_RATE
from config import RESULTS_FILE
from config import RL_MODEL
from config import SIMULATION_TIME
from config import TM_SIZE
from config import TRACE_NAME
from config import LOAD_MODEL
from config import CHECKPOINT_ITERATIONS

from network_simulator_interface import *

if ALGORITHM == 'xWeaver':
    from config import MAX_DEPTH, MAX_WIDTH, TRAINING_ITEMS, ETA, NUM_EPOCHS_SCORE, NUM_EPOCHS_MAIN

if ALGORITHM == 'xWeaver' or ALGORITHM == 'xWeaver_DG':
    from config import TM_FILE, TOPOLOGY_FILE, NUM_SAMPLES, MODEL_FOLDER

def parse_results_file(file_path):
    file_data = open(file_path, 'r').read().split('\n')
    file_data = [line.split(' ') for line in file_data]
    file_data = [line[-1] for line in file_data if len(line) > 1]
    file_data = [float(line) for line in file_data]
        
    return file_data

def parse_tm_file(file_path):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.20f}".format(x)})
    file_data = open(file_path, 'r').read().split('\n\n')
    file_data = [sample.split('\n') for sample in file_data if sample != '']
    file_data = [[line.split() for line in sample] for sample in file_data]
    file_data = [np.array([map(float, line) for line in epoch]) for epoch in file_data]

    return file_data

def parse_topology_file(file_path):
    file_data = open(file_path, 'r').read().split('\n')
    file_data = [epoch[epoch.find(': ') + 2:].split(' ') for epoch in file_data]
    file_data = [[int(epoch[_][:-2]) for _ in xrange(2, len(epoch), 3)] for epoch in file_data]
    file_data = [np.array([lnk for lnk in epoch if lnk >= 1]) for epoch in file_data]
    file_data = [[lnk-1 for lnk in epoch] for epoch in file_data]

    return file_data

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class xWeaver(NetworkSimulator):
    def __init__(self, simulator, num_iterations, step_size, k_actions):
        self._monitor = threading.Event()
        self._monitor.clear()
        self._simulator = simulator(self._monitor, 'xWeaver')
        self._num_iterations = num_iterations
        self._step_size = step_size
        self._k_actions = k_actions

    def topo_generation(self, demand_matrix, previous_topo, session):
        """
        Implemented this algorithm from the pseudo code given in the paper.
        Pseudo Code: https://www.cse.ust.hk/~kaichen/papers/sigmetrics18-xweaver.pdf (Page 9)
        """


        current_topo = copy.deepcopy(previous_topo)
        B = []


        # Search depth represents the amount of edge changes 
        # in a neighbouring topology.

        for search_depth in xrange(1, MAX_DEPTH):
            # Local search with random rolls

            iter_topo = copy.deepcopy(current_topo)
            progress = False

            for search_width in xrange(1, MAX_WIDTH):

                # Pick a neighbouring topology.
                non_zero_indices = np.nonzero(iter_topo)[1]
                zero_indices = [idx for idx in xrange(A_SIZE) if idx not in non_zero_indices]

                inner_iter_topo = copy.deepcopy(iter_topo)

                non_zero_idx = sample(non_zero_indices, search_depth)
                zero_idx = sample(zero_indices, search_depth)

                inner_iter_topo[0, non_zero_idx] = 0
                inner_iter_topo[0, zero_idx] = 1

                score_innner_iter_topo = session.run(self.score, 
                                            feed_dict={self.inputs_tm: demand_matrix,
                                                       self.topo: inner_iter_topo})[0][0]
                score_current_topo = session.run(self.score, 
                                            feed_dict={self.inputs_tm: demand_matrix,
                                                       self.topo: current_topo})[0][0]

                ## In our design less score is better hence comparison operator
                ## is reversed.
                if score_innner_iter_topo <= score_current_topo:
                    progress = True
                    current_topo = copy.deepcopy(inner_iter_topo)
                    B.append((current_topo, score_innner_iter_topo))
            
            B = sorted(B, key=lambda x: x[1])

            best_topo = None

            if len(B) != 0:
                best_topo = copy.deepcopy(B[0][0])

            # Jumping out of the local optimal points

            if progress is False:
                if len(B) != 0:
                    if uniform(0,1) <= ETA:
                        current_topo = copy.deepcopy(B[0][0]) # Fix This
                    else:
                        current_topo = copy.deepcopy(choice(B)[0])

        return best_topo


    def main_network(self, scope):
        with tf.variable_scope(scope):
            self.mn_inputs_tm = tf.placeholder(shape=[None,(TM_SIZE**2)],dtype=tf.float32, name='mn_in_tm')
            self.mn_tmIn = tf.reshape(self.mn_inputs_tm,shape=[-1,TM_SIZE,TM_SIZE,1], name='mn_reshape_tm') # [?, 16, 16, 2]

            self.mn_trafficMatrixLayers = []

            for layerNum in xrange(0,len(RL_MODEL)):
                if layerNum == 0:
                    inputs = self.mn_tmIn
                else:
                    inputs = self.mn_trafficMatrixLayers[-1]

                layer = RL_MODEL[layerNum]

                currentLayer = slim.conv2d(activation_fn=tf.nn.elu, 
                    inputs=inputs, num_outputs=layer[0],
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_size=[layer[1], layer[2]], stride = [layer[3], layer[4]],
                    padding='VALID')

                self.mn_trafficMatrixLayers.append(currentLayer)

                # if scope == 'global':
                #     tf.add_to_collection('conv_weights', currentLayer)

            # Connect the input and output
            hidden_1 = slim.fully_connected(slim.flatten(self.mn_trafficMatrixLayers[-1]),256,activation_fn=tf.nn.elu, scope='mn_fc0') # [?, 256]
            hidden_2 = slim.fully_connected(hidden_1,128,activation_fn=tf.nn.elu, scope='mn_fc1') 

            self.mn_policy = tf.contrib.layers.fully_connected(hidden_2,A_SIZE,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None) # [?, a_size] -> [?, 28]

            self.mn_actions = tf.placeholder(shape=[None, K_ACTIONS],dtype=tf.int32) # [?, k_acts] -> [?, 6]
            self.mn_actions_onehot = tf.one_hot(self.mn_actions,A_SIZE,dtype=tf.float32) # [?, k_acts, a_size] -> [?, 6, 28]

            self.mn_policy_expand = tf.expand_dims(self.mn_policy, 1) # [?, 1, 28]
            self.mn_policy_mul = tf.tile(self.mn_policy_expand, [1, K_ACTIONS, 1]) # [?, 6, 28]

            self.mn_responsible_outputs = tf.reduce_sum(self.mn_policy_mul * self.mn_actions_onehot, [2]) # [?, 6]

            self.mn_policy_value = tf.log(self.mn_responsible_outputs)
            self.mn_loss = -tf.reduce_sum(self.mn_policy_value)

            self.mn_optimizer = tf.train.AdamOptimizer(1e-6)
            self.mn_train_op = self.optimizer.minimize(self.mn_loss)

    def scoring(self, scope):
        with tf.variable_scope(scope):
            self.inputs_tm = tf.placeholder(shape=[None,(TM_SIZE**2)],dtype=tf.float32, name='in_tm')
            self.tmIn = tf.reshape(self.inputs_tm,shape=[-1,TM_SIZE,TM_SIZE, 1], name='reshape_tm')

            self.correct_value = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='final_val')

            TM_conv_0 = slim.conv2d(activation_fn=tf.nn.relu, 
                    inputs=self.tmIn, num_outputs=16,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_size=[4, 4], stride =[1, 1],
                    padding='VALID',
                    scope='conv-0')

            TM_conv_1 = slim.conv2d(activation_fn=tf.nn.relu, 
                    inputs=TM_conv_0, num_outputs=32,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_size=[4, 4], stride =[1, 1],
                    padding='VALID',
                    scope='conv-1')

            self.topo = tf.placeholder(shape=[None,A_SIZE],dtype=tf.float32, name='in_nt')
            self.ntIn = tf.reshape(self.topo,shape=[-1,A_SIZE,1], name='reshape_nt') # [?, 8, 8, 1]

            TOPO_conv_0 = slim.conv1d(activation_fn=tf.nn.relu, 
                    inputs=self.ntIn, num_outputs=16,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_size=[4], stride =[1],
                    padding='VALID',
                    scope='conv-2')

            TOPO_conv_1 = slim.conv1d(activation_fn=tf.nn.relu, 
                    inputs=TOPO_conv_0, num_outputs=32,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_size=[4], stride =[1],
                    padding='VALID',
                    scope='conv-3')

            combined_input = tf.concat([slim.flatten(TOPO_conv_1), slim.flatten(TM_conv_1)], 1, name='concat'); # [?, 832]

            # Connect the input and output
            hidden_1 = slim.fully_connected(combined_input,256,activation_fn=tf.nn.elu, scope='fc0')
            hidden_2 = slim.fully_connected(hidden_1,128,activation_fn=tf.nn.elu, scope='fc1')

            self.score = tf.contrib.layers.fully_connected(hidden_2,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)

            self.loss = (self.correct_value[0][0] - self.score[0][0])**2

            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            self.train_op = self.optimizer.minimize(self.loss)

    def train_score_cnn(self, tm, scores, topo, session, saver):
        for epoch in range(NUM_EPOCHS_SCORE):
            loss = []

            for i in range(TRAINING_ITEMS):
                temp_topo = np.zeros((1, A_SIZE), dtype=np.float32)
                temp_topo[0, topo[i]] = 1

                _, l = session.run([self.train_op, self.loss], feed_dict={self.inputs_tm: [tm[i].flatten()],
                                                   self.correct_value: [[scores[i]]],
                                                   self.topo: temp_topo})
                loss.append(l)

            if epoch % CHECKPOINT_ITERATIONS == 0:
                checkpoint_name = MODEL_FOLDER + 'score-' + str(epoch) + '.ckpt'
                saver.save(session, checkpoint_name)
                print ("Saved Model")

            print "Epoch Number: %d, Loss: %.10f" % (epoch, np.average(loss) * 1e9)

    def train_main_cnn(self, tm, scores, topo, session, saver):
        for epoch in range(NUM_EPOCHS_MAIN):
            loss = []

            for i in range(TRAINING_ITEMS):
                temp_topo = np.zeros((1, A_SIZE), dtype=np.float32)
                temp_topo[0, topo[i]] = 1
                updated_topo = self.topo_generation([tm[i].flatten()], temp_topo, session)

                if updated_topo is None:
                    continue

                # print updated_topo
                temp = np.nonzero(updated_topo)[1]

                _, l = session.run([self.mn_train_op, self.mn_loss], 
                                   feed_dict={self.mn_inputs_tm: [tm[i].flatten()],
                                              self.mn_actions: [temp]})
                loss.append(l)

            if epoch % CHECKPOINT_ITERATIONS == 0:
                checkpoint_name = MODEL_FOLDER + 'main-' + str(epoch) + '.ckpt'
                saver.save(session, checkpoint_name)
                print ("Saved Model")

            print "Loss: %.3f" % np.average(loss)


    def run(self):
        ## Training of xWeaver System

        print "Training of xWeaver System"

        self.scoring("SCORING")
        self.main_network("MAIN_NETWORK")

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        print "Parsing Files"

        tm = parse_tm_file(TM_FILE)
        scores = parse_results_file(RESULTS_FILE)
        topo = parse_topology_file(TOPOLOGY_FILE)

        training_samples = min(len(tm), len(scores), len(topo))

        tm = tm[:training_samples]
        scores = scores[:training_samples]
        topo = topo[:training_samples]

        saver = tf.train.Saver(max_to_keep=100)

        if LOAD_MODEL:
            ckpt = tf.train.get_checkpoint_state(MODEL_FOLDER)
            saver.restore(sess,ckpt.model_checkpoint_path)
            print ('Loading Model: %s' % ckpt.model_checkpoint_path)

        # nr/data-train/

        print "Training Scoring CNN"
        self.train_score_cnn(tm, scores, topo, sess, saver)

        # saver_main = tf.train.Saver(max_to_keep=100)

        print "Training Main CNN"
        self.train_main_cnn(tm, scores, topo, sess, saver)

        ## Testing of xWeaver System
        self._simulator.reset()

        while self._monitor.isSet():
            pass

        self.set_max_reward(100.0)
        self.set_period(self._step_size)

        self.read_node_list()

        counter = 0

        print "Testing of xWeaver"

        while (1):
            if TM_TYPE == 'host':
                self._read_traffic_matrix_HOST()
            elif TM_TYPE == 'tor':
                self._read_traffic_matrix_TOR()

            topo = sess.run([self.mn_policy], 
                        feed_dict={self.mn_inputs_tm: [self._tm[:,:,1].flatten()]})[0][0]

            temp_selects = [_ for _ in xrange(len(topo))]
            temp_selects = np.random.choice(temp_selects, size=K_ACTIONS, replace=False, p=topo)
            temp_selects = temp_selects.tolist()
            
            nextActions = self.parse_actions(temp_selects)

            if counter == 0:
                prevActions = []
            else:
                prevActions = [item for item in prevActions if item not in nextActions]

            self._simulator.step(nextActions, prevActions)

            while (self._monitor.isSet()):
                pass

            if self._simulator.is_done():
                break

            prevActions = nextActions
            counter = (counter + 1)


class xWeaver_Data_Generation(NetworkSimulator):
    def __init__(self, simulator, num_iterations, step_size, k_actions):
        self._monitor = threading.Event()
        self._monitor.clear()
        self._simulator = simulator(self._monitor, 'xWeaver')
        self._num_iterations = num_iterations
        self._step_size = step_size
        self._k_actions = k_actions

    def run(self):
        np.set_printoptions(formatter={'float': lambda x: "{0:0.20f}".format(x)})
        for _ in xrange(NUM_SAMPLES):
            if TRACE_NAME == 'Parsed':
                self._simulator.update_worker_id(_)
                
            self._simulator.reset()

            while self._monitor.isSet():
                pass

            self.set_max_reward(100.0)
            self.set_period(self._step_size)

            self.read_node_list()

            links_to_add = self.parse_actions(sample(xrange(0, A_SIZE), K_ACTIONS))

            with open(TOPOLOGY_FILE, "a") as f:
                f.write(str(datetime.datetime.now()) + ": " + str(links_to_add) + "\n")

            links_to_remove = []
            counter = 0

            while self._simulator.is_done() is False:
                self._simulator.step(links_to_add, links_to_remove)

                while (self._monitor.isSet()):
                    pass

                if TM_TYPE == 'host':
                    t = self._read_traffic_matrix_HOST()
                elif TM_TYPE == 'tor':
                    t = self._read_traffic_matrix_TOR()

                if counter == 0:
                    with open(TM_FILE, 'a') as f:
                        np.savetxt(f, t[:,:,1], fmt='%0.20f')
                        f.write('\n')
                counter += 1