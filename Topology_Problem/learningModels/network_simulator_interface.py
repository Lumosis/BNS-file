import os
import networkx as nx
from .node2vec import *
import numpy as np
import threading

from gensim.models import Word2Vec
from scipy.sparse import *

from ..config import *

# from numba import jit
# import cupy as cp
"""
Code to interface with the simulator

Simulator could be embedded directly within code;
would make communication really easy.

"""


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


B_TO_GB = 8 * 1000 * 1000 * 1000


class NetworkSimulator():

    # @timing_function
    def __init__(self, simulator, monitor):
        # self.monitor = monitor
        self.simulator = simulator
        self.monitor = monitor

    # @timing_function
    def set_period(self, period):  # Used
        self.period = period

    # @timing_function
    def set_actions(self, k_acts, a_size):
        self.k_acts = k_acts
        self.a_size = a_size

    # @timing_function
    def set_max_reward(self, reward):
        self.max_reward = reward

    # @timing_function
    def read_node_list(self):
        self.nt_translation = [{}, []]
        topology = merge_dicts(self.simulator.sw_dict,
                               self.simulator.host_dict)

        counter = 0
        for node in topology:
            self.nt_translation[0][node] = counter
            self.nt_translation[1].append(node)
            counter += 1

        self.num_sws = len(self.simulator.tor_sw_list)
        self.num_hosts = len(self.simulator.host_dict)
        self.num_nodes = len(self.simulator.sw_dict) + \
                          self.num_hosts

        if FINISHED_FLOW_TIMES:
            if TM_TYPE == 'host':
                self.finished_flows = np.zeros(
                    (self.num_hosts, self.num_hosts, 2))
            elif TM_TYPE == 'tor':
                self.finished_flows_tor = np.zeros(
                    (self.num_sws, self.num_sws, 2))

    # @timing_function
    def network_topology_update_step(self, actions):
        index = 0
        self.nt_e = np.zeros((self.num_sws, self.num_sws), \
                              dtype=np.float32)

        for i in range(0, self.num_sws):
            for j in range(0, i):
                self.nt_e[i, j] = actions[0][index]
                self.nt_e[j, i] = actions[0][index]
                index += 1
        return self.nt_e.flatten()

    def _node2vec_conversion(self):
        def read_graph(edge_list, directed=True, weighted=True):
            '''
            Reads the input network in networkx.
            '''
            if weighted:
                G = nx.DiGraph()
                G.add_weighted_edges_from(edge_list)
            else:
                G = nx.DiGraph()
                G.add_edges_from(edge_list)

            if not directed:
                G = G.to_undirected()

            return G

        def learn_embeddings(num_nodes,
                             walks,
                             dimensions=128,
                             window_size=10,
                             workers=8,
                             iterations=1):
            '''
            Learn embeddings by optimizing the Skipgram objective using SGD.
            '''
            walks = [map(str, walk) for walk in walks]
            model = Word2Vec(walks,
                             size=dimensions,
                             window=window_size,
                             min_count=0,
                             sg=1,
                             workers=workers,
                             iter=iterations)

            ret = []
            for key in range(num_nodes):
                if str(key) in model.wv:
                    ret.append(np.array(model.wv[str(key)]))
                else:
                    ret.append(np.zeros(dimensions))
            ret = np.array(ret)

            return ret

        indices = np.nonzero(self.tm[:, :, 0])
        indices = zip(indices[0], indices[1])

        flow_times = [(idx[0], idx[1], self.tm[idx[0], idx[1], 0])
                      for idx in indices]
        flow_sizes = [(idx[0], idx[1], self.tm[idx[0], idx[1], 1])
                      for idx in indices]

        graph_flow_times = read_graph(flow_times)
        graph_flow_sizes = read_graph(flow_sizes)

        G_flow_times = node2vec.Graph(
            graph_flow_times, True, p=NODE2VEC_P,
            q=NODE2VEC_Q)  # p and q are hyperparameters.
        G_flow_sizes = node2vec.Graph(graph_flow_sizes,
                                      True,
                                      p=NODE2VEC_P,
                                      q=NODE2VEC_Q)

        G_flow_times.preprocess_transition_probs()
        G_flow_sizes.preprocess_transition_probs()

        walks_flow_times = G_flow_times.simulate_walks(num_walks=NUM_WALKS,
                                                       walk_length=WALK_LENGTH)
        walks_flow_sizes = G_flow_sizes.simulate_walks(num_walks=NUM_WALKS,
                                                       walk_length=WALK_LENGTH)

        if walks_flow_times != []:
            num_nodes = self.num_sws
            flow_times = learn_embeddings(num_nodes, walks_flow_times,
                                          NUM_DIMENSIONS)
            flow_sizes = learn_embeddings(num_nodes, walks_flow_sizes,
                                          NUM_DIMENSIONS)
            ret = np.stack((flow_times, flow_sizes), axis=2)
        else:
            ret = np.zeros((self.num_sws, NUM_DIMENSIONS, 2))

        return ret

    # @timing_function
    def read_traffic_matrix_TOR(self):
        self.flow_start_bytes = {}
        if SPARSE_INPUT:
            self.flow_sizes = dok_matrix((self.num_sws, self.num_sws),
                                         dtype=np.float32)
            self.flow_times = dok_matrix((self.num_sws, self.num_sws),
                                         dtype=np.float32)
        else:
            self.tm = np.zeros((self.num_sws, self.num_sws, 2),
                               dtype=np.float32)

        if self.simulator.scheduler == None:
            curr_time = 0.0
        else:
            curr_time = float(self.simulator.scheduler.sim_clock)

        runningFlows = self.simulator.flow
        # print "Current Time: %s" % str(curr_time)

        for flowID in runningFlows.keys():
            flow = runningFlows[flowID]

            if flow.started == False:
                if ALGORITHM != 'xWeaver' and ALGORITHM != 'xWeaver_DG' and ALGORITHM != 'xWeaver_TRACE':
                    self.flow_start_bytes[flowID] = flow.remaining_tx_bytes
                    continue

            if flow.start_time < curr_time + self.period:
                srcHost = int(flow.src_name[4:])
                dstHost = int(flow.dst_name[4:])

                if srcHost == dstHost:
                    # del runningFlows[flowID]
                    del flow
                    continue

                if len(flow.path) >= 3:
                    if 'tor' in flow.path[1] and 'tor' in flow.path[-2]:
                        src_TOR = int(flow.path[1][3:])
                        dst_TOR = int(flow.path[-2][3:])

                    if flow.finished:
                        if FINISHED_FLOW_TIMES:
                            self.finished_flows_tor[
                                src_TOR, dst_TOR, 0] = np.add(
                                    self.
                                    finished_flows_tor[src_TOR, dst_TOR, 0],
                                    flow.start_time)
                            self.finished_flows_tor[
                                src_TOR, dst_TOR, 1] = np.add(
                                    self.
                                    finished_flows_tor[src_TOR, dst_TOR, 1], 1)
                        # del runningFlows[flowID]
                        del flow
                        continue

                    if src_TOR != dst_TOR:
                        if LIVE_FLOW_TIMES:
                            ft_duration = 0.0 if flow.start_time >= curr_time \
                                           else ((curr_time - flow.start_time)
                                           / self.period)

                            if SPARSE_INPUT:
                                self.flow_times[
                                    src_TOR, dst_TOR] = self.flow_times[
                                        src_TOR, dst_TOR] + ft_duration
                            else:
                                self.tm[src_TOR, dst_TOR, 0] = \
                                    np.add(self.tm[src_TOR, dst_TOR, 0],
                                    ft_duration)

                        if FLOW_SIZES:
                            self.flow_start_bytes[flowID] = \
                                flow.remaining_tx_bytes

                            if ALGORITHM == 'xWeaver' or ALGORITHM == 'xWeaver_DG' or ALGORITHM == 'xWeaver_TRACE':
                                self.tm[src_TOR, dst_TOR, 1] = \
                                    np.add(self.tm[src_TOR, dst_TOR, 1],
                                    flow.remaining_tx_bytes / B_TO_GB)
                            else:
                                if SPARSE_INPUT:
                                    # print self.flow_sizes[src_TOR, dst_TOR]
                                    self.flow_sizes[
                                        src_TOR, dst_TOR] = self.flow_sizes[
                                            src_TOR, dst_TOR] + (
                                                flow.transferred_bytes /
                                                B_TO_GB)
                                else:
                                    self.tm[src_TOR, dst_TOR, 1] = \
                                        np.add(self.tm[src_TOR, dst_TOR, 1],
                                        flow.transferred_bytes / B_TO_GB)

                                flow.transferred_bytes = 0

        if FINISHED_FLOW_TIMES:
            for i in range(self.tm.shape[0]):
                for j in range(self.tm.shape[1]):
                    self.tm[i, j, 0] = np.add(
                        self.tm[i, j, 0],
                        ((curr_time * self.finished_flows_tor[i, j, 1]) -
                         self.finished_flows_tor[i, j, 0]) / self.period)

        if SPARSE_INPUT:
            return self.flow_sizes, self.flow_times
        else:
            return self.tm

    # @timing_function
    def read_traffic_matrix_HOST(self):
        # print "Traffic Matrix"
        self.flow_start_bytes = {}
        self.tm = np.zeros((self.num_hosts, self.num_hosts, 2),
                           dtype=np.float32)

        if self.simulator.scheduler == None:
            curr_time = 0.0
        else:
            curr_time = float(self.simulator.scheduler.sim_clock)
        runningFlows = self.simulator.flow

        del_FlowIDs = []

        for flowID in runningFlows.keys():
            flow = runningFlows[flowID]

            if flow.started == False:
                if ALGORITHM != 'xWeaver' and ALGORITHM != 'xWeaver_DG' or ALGORITHM == 'xWeaver_TRACE':
                    self.flow_start_bytes[flowID] = flow.remaining_tx_bytes
                    continue

            if flow.start_time < curr_time + self.period:
                srcHost = int(flow.src_name[4:])
                dstHost = int(flow.dst_name[4:])

                if srcHost == dstHost:
                    del_FlowIDs.append(flowID)
                    del flow
                    continue

                if flow.finished:
                    if FINISHED_FLOW_TIMES:
                        self.finished_flows[srcHost, dstHost, 0] = np.add(
                            self.finished_flows[srcHost, dstHost, 0],
                            flow.start_time)
                        self.finished_flows[srcHost, dstHost, 1] = np.add(
                            self.finished_flows[srcHost, dstHost, 1], 1)
                    del_FlowIDs.append(flowID)
                    del flow
                    continue

                if srcHost != dstHost:
                    if LIVE_FLOW_TIMES:
                        ft_duration = 0.0 if flow.start_time >= curr_time else (
                            curr_time - flow.start_time) / self.period
                        self.tm[srcHost, dstHost, 0] = np.add(
                            self.tm[srcHost, dstHost, 0], ft_duration)

                    if FLOW_SIZES:
                        if ALGORITHM == 'xWeaver' or ALGORITHM == 'xWeaver_DG' or ALGORITHM == 'xWeaver_TRACE':
                            self.tm[srcHost, dstHost, 1] = np.add(
                                self.tm[srcHost, dstHost, 1],
                                flow.remaining_tx_bytes / B_TO_GB)
                        else:
                            self.tm[srcHost, dstHost, 1] = np.add(
                                self.tm[srcHost, dstHost, 1],
                                flow.transferred_bytes / B_TO_GB)
                            flow.transferred_bytes = 0

                    self.flow_start_bytes[flowID] = flow.remaining_tx_bytes

        for flowID in del_FlowIDs:
            del runningFlows[flowID]

        if FINISHED_FLOW_TIMES:
            for i in range(self.tm.shape[0]):
                for j in range(self.tm.shape[1]):
                    self.tm[i, j, 0] = np.add(
                        self.tm[i, j, 0],
                        ((curr_time * self.finished_flows[i, j, 1]) -
                         self.finished_flows[i, j, 0]) / self.period)

        return self.tm

    # @timing_function
    def read_reward(self, actions):
        total_reward = np.zeros((self.num_sws, self.num_sws), \
                                   dtype=np.float32)
        curr_time = float(self.simulator.scheduler.sim_clock)
        runningFlows = self.simulator.flow

        for flowID in self.flow_start_bytes.keys():
            flow = runningFlows[flowID]

            path = flow.path
            flow_time = (curr_time - flow.start_time)

            if len(path) >= 3:
                if 'tor' in path[1] and 'tor' in path[-2]:
                    src_sw = int(path[1][3:])
                    dst_sw = int(path[-2][3:])

                    total_reward[src_sw, dst_sw] += \
                      (self.flow_start_bytes[flowID] - \
                        float(flow.remaining_tx_bytes)) * (curr_time - flow.job.start_time)

        # print total_reward
        # Determine total reward for individual link


        selected_reward = np.zeros((1, self.k_acts))
        indexb = 0
        # print(actions)
        # print(type(actions))
        # input('read_rew')
        for i in range(0, self.num_sws):
            for j in range(0, i):
                if indexb in actions:
                    # ind = np.where(actions ==　indexb)
                    ind = actions.index(indexb)
                    selected_reward[0, ind] += \
                        total_reward[i, j] / B_TO_GB

                    if DIRECTION == 'BI':
                        selected_reward[0, ind] += \
                            total_reward[j, i] / B_TO_GB
                indexb += 1

        if DIRECTION == 'UNI':
            for j in range(0, self.num_sws):
                for i in range(0, j):
                    if indexb in actions:
                        ind = actions.index(indexb)
                        selected_reward[0, ind] += \
                            total_reward[i, j] / B_TO_GB
                    indexb += 1

        # Clip the reward

        # if np.sum(selected_reward) == 0.0:
        #     selected_reward.fill(-3.0)
        # print selected_reward
        self.selected_reward = np.clip(selected_reward, 0.0, \
                                    self.max_reward).tolist()
        if LOG_REWARDS:
            rewards = open(REWARDS_FILE + '_' + str(os.getpid()), "a")
            rewards.write(str(self.selected_reward) + '\n')
            rewards.close()

    # @timing_function
    def parse_actions(self, actions):
        res = []
        index = 0

        for i in range(0, self.num_sws):
            for j in range(0, i):
                if index in actions:
                    src_sw = 'tor%d' % i
                    dst_sw = 'tor%d' % j

                    res.append([src_sw, dst_sw, index + 1])
                    if DIRECTION == 'BI':
                        res.append([dst_sw, src_sw, -1 * index])
                index = index + 1

        if DIRECTION == 'UNI':
            for j in range(0, self.num_sws):
                for i in range(0, j):
                    if index in actions:
                        src_sw = 'tor%d' % i
                        dst_sw = 'tor%d' % j

                        res.append([src_sw, dst_sw, index + 1])
                    index = index + 1

        return res


    def step(self, actions):
        if SPARSE_INPUT:
            del self.flow_sizes
            del self.flow_times
        else:
            del self.tm  # To Free Memory

        self.prev_actions = [
            item for item in self.prev_actions if item not in actions
        ]

        self.simulator.step(self.parse_actions(actions),
                            self.parse_actions(self.prev_actions))

        while (self.monitor.isSet()):
            pass

        # Read the new state
        self.read_reward(actions)

        if TM_TYPE == 'host':
            self.read_traffic_matrix_HOST()
        elif TM_TYPE == 'tor':
            self.read_traffic_matrix_TOR()

        np.set_printoptions(
            formatter={'float': lambda x: "{0:0.10f}".format(x)})
        with open(TRAFFIC_MATRICES, 'a') as f:
            # f.write(str(self.tm[:,:,0]))
            # f.write('\n')
            f.write(str(self.tm[:, :, 1]))
            f.write('\n')
            f.write('\n')

        # Check if we are done
        self.done = self.simulator.is_done()

        # Store the actions
        self.prev_actions = actions

        if self.done == True:
            if LOG_REWARDS:
                rewards = open(REWARDS_FILE + '_' + str(os.getpid()), "a")
                rewards.write('\n\n')
                rewards.close()
            self.prev_actions = []

            if not NODE2VEC:
                if INPUT_NORMALIZATION:
                    if SPARSE_INPUT:
                        pass
                    else:
                        if np.std(self.tm[:, :, 0]) != 0:
                            self.tm[:, :, 0] = (self.tm[:, :, 0] - np.average(
                                self.tm[:, :, 0])) / np.std(self.tm[:, :, 0])

                        if np.std(self.tm[:, :, 1]) != 0:
                            self.tm[:, :, 1] = (self.tm[:, :, 1] - np.average(
                                self.tm[:, :, 1])) / np.std(self.tm[:, :, 1])

        if TOPOLOGY_INCLUDED:
            if NODE2VEC:
                self.tm = self.node2vec_conversion()

            return [self.nt_e.flatten(),
                    self.tm.flatten()], self.selected_reward, self.done
        else:
            if NODE2VEC:
                self.tm = self.node2vec_conversion()
            if SPARSE_INPUT:
                return [self.flow_times,
                        self.flow_sizes], self.selected_reward, self.done
            else:
                return self.tm.flatten(), self.selected_reward, self.done

    # @timing_function    # Called during training to advance the simulation
    def updated_step(self, actions):
        reward_set = []

        for idx in range(STEPS):
            if self.done == True:
                break

            if SPARSE_INPUT:
                del self.flow_sizes
                del self.flow_times
            else:
                del self.tm  # To Free Memory

            self.prev_actions = [
                item for item in self.prev_actions if item not in actions[idx]
            ]

            self.simulator.step(self.parse_actions(actions[idx]),
                                self.parse_actions(self.prev_actions))

            while (self.monitor.isSet()):
                pass

            # Read the new state
            self.read_reward(actions[idx])

            if TM_TYPE == 'host':
                self.read_traffic_matrix_HOST()
            elif TM_TYPE == 'tor':
                self.read_traffic_matrix_TOR()

            np.set_printoptions(
                formatter={'float': lambda x: "{0:0.10f}".format(x)})
            with open(TRAFFIC_MATRICES, 'a') as f:
                # f.write(str(self.tm[:,:,0]))
                # f.write('\n')
                f.write(str(self.tm[:, :, 1]))
                f.write('\n')
                f.write('\n')

            # Check if we are done
            self.done = self.simulator.is_done()

            # Store the actions
            self.prev_actions = actions[idx]

            if self.done == True:
                if LOG_REWARDS:
                    rewards = open(REWARDS_FILE + '_' + str(os.getpid()), "a")
                    rewards.write('\n\n')
                    rewards.close()
                self.prev_actions = []

            if not NODE2VEC:
                if INPUT_NORMALIZATION:
                    if SPARSE_INPUT:
                        pass
                    else:
                        if np.std(self.tm[:, :, 0]) != 0:
                            self.tm[:, :, 0] = (self.tm[:, :, 0] - np.average(
                                self.tm[:, :, 0])) / np.std(self.tm[:, :, 0])

                        if np.std(self.tm[:, :, 1]) != 0:
                            self.tm[:, :, 1] = (self.tm[:, :, 1] - np.average(
                                self.tm[:, :, 1])) / np.std(self.tm[:, :, 1])

            reward_set.append(np.array(self.selected_reward[0]))

        while len(reward_set) < STEPS:
            reward_set.append(np.zeros((self.k_acts)))

        reward_set = np.array(reward_set)

        if TOPOLOGY_INCLUDED:
            if NODE2VEC:
                self.tm = self.node2vec_conversion()

            return [self.nt_e.flatten(),
                    self.tm.flatten()], reward_set, self.done
        else:
            if NODE2VEC:
                self.tm = self.node2vec_conversion()
            if SPARSE_INPUT:
                return [self.flow_times,
                        self.flow_sizes], reward_set, self.done
            else:
                return self.tm.flatten(), reward_set, self.done

    # @timing_function
    def init(self):
        self.done = False
        self.read_node_list()

        if TM_TYPE == 'host':
            self.read_traffic_matrix_HOST()
        elif TM_TYPE == 'tor':
            self.read_traffic_matrix_TOR()
        # self.read_network_topology()

        np.set_printoptions(
            formatter={'float': lambda x: "{0:0.10f}".format(x)})

        # with open(TRAFFIC_MATRICES, 'a') as f:
        # f.write(str(self.tm[:,:,0]))
        # f.write('\n')
        # f.write(str(self.tm[:,:,1]))
        # f.write('\n')

        self.prev_actions = []

        if TOPOLOGY_INCLUDED:
            self.network_topology_update_step(np.array([np.zeros(self.a_size)
                                                        ]))

    # @timing_function
    def reset(self):
        # Reset the simulator
        ret = self.simulator.reset()

        if ret == None:
            return None

        while self.monitor.isSet():
            pass

        self.done = False
        self.read_node_list()

        if TM_TYPE == 'host':
            self.read_traffic_matrix_HOST()
        elif TM_TYPE == 'tor':
            self.read_traffic_matrix_TOR()

        self.prev_actions = []

        if TOPOLOGY_INCLUDED:
            self.network_topology_update_step([[0] * self.a_size])

        # Return the state to the network
        if TOPOLOGY_INCLUDED:
            if NODE2VEC:
                self.tm = self.node2vec_conversion()
            return [self.nt_e.flatten(), self.tm.flatten()]
        else:
            if NODE2VEC:
                self.tm = self.node2vec_conversion()
            if SPARSE_INPUT:
                return [self.flow_times, self.flow_sizes]
            else:
                return self.tm.flatten()

    # @timing_function
    def is_episode_finished(self):
        return self.done