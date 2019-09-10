"""
    Code from: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2#.ckt96znsk
    @author Arthur Juliani

    Modified by:
    @author Chris Streiffer
"""

import numpy as np
import os
import scipy.signal
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading

from datetime import datetime
from skimage.transform import resize
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from time import sleep

from config import A_SIZE
from config import CLIPPING
from config import CLIPPING_VALUE
from config import DEBUG_LOGGING
from config import GAMMA
from config import INTERPRETABLE_ML
from config import K_ACTIONS
from config import LEARNING_RATE
from config import LEARNING_RATE_DECAY_FACTOR
from config import LOAD_MODEL
from config import MAX_REWARD
from config import NT_SIZE
from config import NUM_EPOCHS_PER_DECAY
from config import NUM_VALID_STEPS
from config import NUM_WORKERS
from config import RESULTS_FILE
from config import RL_MODEL
from config import SIMULATION_TIME
from config import TESTING
from config import TIME_STEP
from config import TM_SIZE
from config import TOPOLOGY

import matplotlib.pyplot as plt

import multiprocessing as mp

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=164)
np.set_printoptions(formatter={'float_kind': lambda x: "%8.3f" % x})

if TOPOLOGY == 'VL2':
    from config import Da, Di
elif TOPOLOGY == 'FATTREE':
    from config import K
elif TOPOLOGY == 'JELLYFISH':
    pass

# I/O values
tf.app.flags.DEFINE_string("model_path", "nr/data-train/",
                           "Directory to write the model")

tf.app.flags.DEFINE_string("log_dir", "nr/data-logs/",
                           "Directory to logs.")

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

tf.app.flags.DEFINE_integer("max_episode_length", int(SIMULATION_TIME / TIME_STEP),
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
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discount_loc(x, gamma, k, num_steps):
    xarr = np.array(x)
    vals = scipy.signal.lfilter([1], [1, -gamma], np.fliplr(xarr.transpose()), axis=1)
    return np.fliplr(vals).transpose()

# Discounting function used to calculate discounted returns.
def discount_tot(x, gamma, k, num_steps):
    return np.repeat(scipy.signal.lfilter([1], [1, -gamma], x[::-k], axis=0)[::-1], k)

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def write_line(file, str):
  # file.write('%s\n' % str)
  # file.flush()
  pass

def calc_dist(dist, temp_actions):
    for i in range(len(temp_actions)):
        dist[i] = 0.0 if temp_actions[i] == 1 else dist[i]
    return dist/np.sum(dist, dtype=np.float32)

class AC_Network():
    def __init__(self,nt_size,tm_size,a_size,scope,trainer,global_episodes):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs_tm = tf.placeholder(shape=[None,(tm_size**2)*2],dtype=tf.float32, name='in_tm')
            self.tmIn = tf.reshape(self.inputs_tm,shape=[-1,tm_size,tm_size,2], name='reshape_tm') # [?, 16, 16, 2]

            self.trafficMatrixLayers = []

            for layerNum in xrange(0,len(RL_MODEL)):
                if layerNum == 0:
                    inputs = self.tmIn
                else:
                    inputs = self.trafficMatrixLayers[-1]

                layer = RL_MODEL[layerNum]

                currentLayer = slim.conv2d(activation_fn=tf.nn.elu, 
                    inputs=inputs, num_outputs=layer[0],
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_size=[layer[1], layer[2]], stride =[layer[3], layer[4]],
                    padding='VALID')

                self.trafficMatrixLayers.append(currentLayer)

                # if scope == 'global':
                #     tf.add_to_collection('conv_weights', currentLayer)

            # Connect the input and output
            hidden = slim.fully_connected(slim.flatten(self.trafficMatrixLayers[-1]),256,activation_fn=tf.nn.elu, scope='fc0') # [?, 256]

            # Stuff Added
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.tmIn)[:1] # Check This
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])
            # Stuff Added

            # print "Shape: %s" % np.shape(rnn_out)
            
            #Output layers for policy and value estimations
            self.policy = tf.contrib.layers.fully_connected(rnn_out,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None) # [?, a_size] -> [?, 28]

            self.value = tf.contrib.layers.fully_connected (rnn_out,FLAGS.k_acts,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None) # [?, k_acts] -> [?, 6]

            if FLAGS.clipping:
                self.policy = tf.clip_by_value(self.policy, 1e-10, 
                    tf.reduce_max(self.policy) + 1)
                self.value = tf.clip_by_value(self.value, 1e-10, 
                    tf.reduce_max(self.value) + 1)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None, FLAGS.k_acts],dtype=tf.int32) # [?, k_acts] -> [?, 6]
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32) # [?, k_acts, a_size] -> [?, 6, 28]
                self.target_v = tf.placeholder(shape=[None, FLAGS.k_acts],dtype=tf.float32) # [?, 6] (Discounted Rewards)

                # Positive reward
                self.advantages = tf.placeholder(shape=[None, FLAGS.k_acts],dtype=tf.float32) # [?, 6] (Generalized Advantage Estimation)
                self.policy_expand = tf.expand_dims(self.policy, 1) # [?, 1, 28]

                self.policy_mul = tf.tile(self.policy_expand, [1, FLAGS.k_acts, 1]) # [?, 6, 28]
                self.responsible_outputs = tf.reduce_sum(self.policy_mul * self.actions_onehot, [2]) # [?, 6]

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(tf.reshape(self.target_v, [-1]) - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))


                # Clipping so that NANs don't appear
                if FLAGS.clipping:
                    self.responsible_outputs = tf.clip_by_value(self.responsible_outputs, 
                        tf.reduce_min(self.responsible_outputs) + FLAGS.clipping_value, 
                        tf.reduce_max(self.responsible_outputs) + FLAGS.clipping_value)


                self.policy_value = tf.log(self.responsible_outputs)*self.advantages
                self.policy_loss = -tf.reduce_sum(self.policy_value)

                # =================
                # self.entropy becomes NaN.
                # =================
                # self.loss = 0.5 * self.value_loss + self.policy_loss - 0.01*self.entropy

                self.loss = 0.5 * self.value_loss + self.policy_loss - 0.01 * self.entropy


                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars), global_step=global_episodes)

class Worker():
    def __init__(self,simulator,name,nt_size,tm_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards_a = []
        # self.summary_writer = tf.train.SummaryWriter("train_"+str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(nt_size,tm_size,a_size,self.name,trainer,global_episodes)
        self.update_local_ops = update_target_graph('global',self.name)        

        #End Doom set-up
        self.env = simulator

        # Setup logging
        # self.state_file = open(os.path.join(FLAGS.log_dir, 'state-%02d-out.txt' % self.number), 'w')
        # self.reward_file = open(os.path.join(FLAGS.log_dir, 'reward-%02d-out.txt' % self.number), 'w')
        # self.train_file_loc = open(os.path.join(FLAGS.log_dir, 'train-loc-%02d-out.txt' % self.number), 'w')
        # self.train_file_glo = open(os.path.join(FLAGS.log_dir, 'train-glo-%02d-out.txt' % self.number), 'w')
        
    def train(self,rollout,sess,gamma,bootstrap_value,step,disount_fn):
        rollout = np.array(rollout)

        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]

        # print np.shape(bootstrap_value)
        # print np.shape(np.vstack(rewards))

        # print(actions)
        # raw_input("Please wait")
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"

        # print("Rewards: ", np.shape(np.vstack(rewards)))
        # print("Boostrap Value: ", np.shape(bootstrap_value))

        # print rewards
        # print bootstrap_value

        self.rewards_plus = np.concatenate((np.vstack(rewards), bootstrap_value), axis=0)


        discounted_rewards = disount_fn(self.rewards_plus, gamma, FLAGS.k_acts, FLAGS.num_steps+1)[:-1]
        self.value_plus = np.concatenate((np.vstack(values), bootstrap_value), axis=0)

        advantages = np.vstack(rewards) + gamma * (self.value_plus[1:, :] - self.value_plus[:-1, :])
        advantages = disount_fn(advantages, gamma, FLAGS.k_acts, FLAGS.num_steps)
        advantages = np.clip(advantages, 0.0, FLAGS.max_reward)

        # advantages = rewards

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # rnn_state = self.local_AC.state_init
        # self.batch_rnn_state = rnn_state
        feed_dict = {self.local_AC.target_v:np.vstack(discounted_rewards),
            self.local_AC.inputs_tm:np.vstack([obs for obs in observations]),
            self.local_AC.actions:np.vstack(actions),
            self.local_AC.advantages:np.vstack(advantages),
            self.local_AC.state_in[0]:self.batch_rnn_state[0],
            self.local_AC.state_in[1]:self.batch_rnn_state[1]}

        # feed_dict = {self.local_AC.target_v:discounted_rewards,
        #     self.local_AC.inputs_nt:np.vstack([obs[0] for obs in observations]),
        #     self.local_AC.inputs_tm:np.vstack([obs[1] for obs in observations]),
        #     self.local_AC.actions:actions,
        #     self.local_AC.advantages:advantages,
        #     self.local_AC.state_in[0]:rnn_state[0],
        #     self.local_AC.state_in[1]:rnn_state[1]}

        # check_op = tf.add_check_numerics_ops()

        val, pol, t_l, v_l,p_l,e_l,g_n,v_n,_,resp,adv, p,a, self.batch_rnn_state = sess.run([
            # check_op,
            self.local_AC.value,
            self.local_AC.policy,
            self.local_AC.loss,
            self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads,
            self.local_AC.responsible_outputs,
            self.local_AC.advantages,
            self.local_AC.policy_value,
            self.local_AC.actions_onehot,
            self.local_AC.state_out],
            feed_dict=feed_dict)

        # print "=================================================="
        # print "Value: " + str(val)
        # print "Policy: " + str(pol)
        # print "Responsible Outputs: " + str(resp)
        # print "Advantages: " + str(adv)
        # print "Unresponsible Outputs: " + str(unresp)
        # print "Negative Advantages: " + str(n_adv)

        # print "Policy Value: " + str(p)
        # print "Total Loss: " + str(t_l)
        # print "Value Loss: " + str(v_l)
        # print "Policy Loss: " + str(p_l)
        # print "Entropy Loss: " + str(e_l)
        # print "==================================================="


        return v_l / len(rollout), p_l / len(rollout),e_l / len(rollout), g_n, v_n, t_l
        
    def work(self,max_episode_length,gamma,sess,coord,saver):
        episode_count = sess.run(self.global_episodes)
        # total_steps = 0
        max_steps = FLAGS.max_episode_length

        # write_line(self.reward_file, 'INIT: time:%s' % (datetime.now()))
        with sess.as_default(), sess.graph.as_default():   
            while not coord.should_stop():
                # self.action_file = open(os.path.join(FLAGS.log_dir, 'actions-%02d-out.txt' % self.number), 'w')
                sess.run(self.update_local_ops)
                loss_buffer = []
                episode_buffer_a = []
                # episode_buffer_b = []
                # episode_values = [[] for _ in range(7)]
                # episode_reward_loc = np.zeros((1, FLAGS.k_acts))
                # episode_reward_tot = 0
                episode_step_count = 0
                d = False
                d_inc = False
                
                self.env.send(('RESET', None))
                s = self.env.recv()
 
                if s is None:
                    self.env.send(('END', None))
                    return

                # Stuff Added
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                # Stuff Added

                self.env.send(('IS_EPISODE_FINISHED', None))
                while (self.env.recv() == False): 
                    #Take an action using probabilities from policy network output.
                    temp_actions = np.zeros((1, FLAGS.a_size))
                    temp_selects = []
                    a_dist,v,rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out], 
                        feed_dict={
                            self.local_AC.inputs_tm:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]
                    })

                    # print "Initial Distribution: " + str(a_dist)
                    a_dist = a_dist[0]

                    # print a_dist

                    temp_selects = [_ for _ in xrange(len(a_dist))]

                    # print temp_selects
                    # print a_dist

                    temp_selects = np.random.choice(temp_selects, size=FLAGS.k_acts, replace=False, p=a_dist)
                    temp_selects = temp_selects.tolist()

                    for a_sel in temp_selects:
                        temp_actions[0][a_sel] = 1

                    # self.env.network_topology_update_step(temp_actions)
                    # s1,r,d = self.env.step(temp_selects)

                    self.env.send(('OPTICAL_LINKS', temp_selects))
                    s1,r,d = self.env.recv()
   
                    episode_buffer_a.append([s,temp_selects,r[0],s1,d,v])
             
                    episode_step_count += 1
                    
                    if TESTING == False:
                        if len(episode_buffer_a) == FLAGS.num_steps or d == True:
                            v = sess.run(self.local_AC.value, 
                                feed_dict={
                                    self.local_AC.inputs_tm:[s1],
                                    self.local_AC.state_in[0]:rnn_state[0],
                                    self.local_AC.state_in[1]:rnn_state[1]
                            })
                            # Since we don't know what the true final return is, we "bootstrap" from our current
                            # value estimation.
                            v_l,p_l,e_l,g_n,v_n, totalLoss = self.train(episode_buffer_a,sess,gamma,v,episode_step_count,discount_loc)

                            if self.name == 'worker_0':
                                loss_buffer.append(totalLoss)

                            episode_buffer_a = []
                            sess.run(self.update_local_ops)

                    # Update the state
                    s = s1    

                    # Check if we're done here
                    if d == True or episode_step_count == max_steps:
                            break

                    self.env.send(('IS_EPISODE_FINISHED',None))
                
                if self.name == 'worker_0':
                    sess.run(self.increment)
                    f = (lambda x,y: x+y)

                    if TESTING == False:
                        fileName = "CNN_" + str(FLAGS.max_episode_length) + "_" + str(FLAGS.period) + "_"
                        results = open(RESULTS_FILE, "a")
                        results.write("Episode Number: " + str(episode_count) + " ")
                        results.write("Time: " + str(datetime.now().time()) + " ")
                        results.write("Episode Loss: " + str(reduce(f, loss_buffer) / len(loss_buffer)) + '\n')
                        results.close()


                # print tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global')

                episode_count += 1
                if episode_count % 15 == 0 and self.name == 'worker_0':
                    saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.ckpt')
                    print ("Saved Model")

class RLModelSimulator(object):
    def __init__(self, simulator, network_simulator):
        self.simulator = simulator
        self.network_simulator = network_simulator
        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)

    def generate_envs(self):
        envs = []

        def generate_enviornment(worker_ID, pipe):
            monitor = threading.Event()
            sim = self.simulator(monitor, 'sim-%s-%02d' % (FLAGS.sim_out_dir, i))

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
                elif command == 'END':
                    break

        for i in range(FLAGS.num_workers):
            p1, p2 = mp.Pipe()
            p = mp.Process(target=generate_enviornment, args=(i,p2))
            p.start()
            envs.append(p1)

        return envs

    def run(self):
        tf.reset_default_graph()

        if not os.path.exists(FLAGS.model_path):
            os.makedirs(FLAGS.model_path)

        with tf.device("/gpu:0"): 
            global_episodes = tf.get_variable('global_episodes', [], initializer=tf.constant_initializer(0), trainable=False)
            num_batches_per_epoch = (FLAGS.max_episode_length/FLAGS.num_steps)
            decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
            self._lr = tf.train.exponential_decay(FLAGS.learning_rate, global_episodes, decay_steps, FLAGS.learning_rate_decay_factor, staircase=True)
            
            trainer = tf.train.AdamOptimizer(learning_rate=self._lr)
            master_network = AC_Network(FLAGS.nt_size,FLAGS.tm_size,FLAGS.a_size,'global',None,global_episodes) # Generate global network
            workers = []

            res = self.generate_envs()

            # Create worker classes
            for i in range(FLAGS.num_workers):

                # Create the worker containing the simulator
                workers.append(Worker(res[i],i,FLAGS.nt_size,FLAGS.tm_size,FLAGS.a_size,trainer,FLAGS.model_path,global_episodes))

            saver = tf.train.Saver(max_to_keep=5)
        
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            coord = tf.train.Coordinator()
            if FLAGS.load_model:
                ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
                saver.restore(sess,ckpt.model_checkpoint_path)
                print ('Loading Model: %s' % ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())


            if DEBUG_LOGGING:
                print("A3C_Without_LSTM >> Creating Worker Threads")
            
            # This is where the asynchronous magic happens.
            # Start the "work" process for each worker in a separate threat.
            worker_threads = []
            for worker in workers:
                worker_work = lambda: worker.work(FLAGS.max_episode_length,FLAGS.gamma,sess,coord,saver)
                t = threading.Thread(target=(worker_work))
                t.start()
                sleep(0.2)
                worker_threads.append(t)
            coord.join(worker_threads)