from config import RESULTS_FILE
from network_simulator_interface import *

import random

class OptimalSolution(NetworkSimulator):
    def __init__(self, simulator, num_iterations, step_size, k_actions):
        self._monitor = threading.Event()
        self._monitor.clear()
        self._simulator = simulator(self._monitor, 'OptimalSolution')
        self._num_iterations = num_iterations
        self._step_size = step_size
        self._optimalLinks = [[]]
        self._k_actions = k_actions

    def run(self):
        for i in xrange(1, self._num_iterations + 1):
            print "Iteration: {}".format(i)

            self._simulator.reset()
        
            while self._monitor.isSet():
                pass

            prevActions = []
            self.set_max_reward(1000000.0)
            self.set_period(self._step_size)
            self.read_node_list()

            for j in xrange(i):
                nextActions = self._optimalLinks[j]

                prevActions = [item for item in prevActions if item not in nextActions]

                if j == 0:
                    prevActions = []

                self._simulator.step(self.parse_actions(nextActions), self.parse_actions(prevActions))

                while (self._monitor.isSet()): ## Monitor is True, Waiting Here
                    pass

                lnkSet = self.read_reward()
                self._optimalLinks[j] = lnkSet

                if self._simulator.is_done():
                    break

                prevActions = nextActions

            if self._optimalLinks[-1] == []:
                results = open(RESULTS_FILE, "a")
                results.write(str(self._optimalLinks) + "\n\n")
                results.close()
                break

            self._optimalLinks.append([])

            if not self._simulator.is_done():
                self._simulator.end()

                while self._monitor.isSet():
                    pass


        self._simulator.reset()

        while self._monitor.isSet():
            pass

        self.set_max_reward(100000.0)
        self.set_period(self._step_size)
        self.read_node_list()

        counter = 0
        prevActions = []

        while (1):
            nextActions = self.parse_actions(self._optimalLinks[counter])
            prevActions = [item for item in prevActions if item not in nextActions]

            if counter == 0:
                prevActions = []

            self._simulator.step(nextActions, self.parse_actions(prevActions))

            while (self._monitor.isSet()):
                pass

            if self._simulator.is_done():
                break

            prevActions = nextActions
            counter = (counter + 1)  % self._num_iterations
            nextActions = self._optimalLinks[counter]

    def read_reward(self):
        total_reward = np.zeros((self._num_sws, self._num_sws), dtype=np.float)
        curr_time = float(self._simulator.scheduler.sim_clock)
        runningFlows = self._simulator.flow

        # print "Read Reward Running"

        for flowID in runningFlows.keys():
            flow = runningFlows[flowID]
            # print flow.path
            path = flow.path

            if flow.started == False:
                continue

            if len(path) >= 3:
                if 'tor' in path[1] and 'tor' in path[-2]:
                    src_sw = int(path[1][3:])
                    dst_sw = int(path[-2][3:])

                    if src_sw == dst_sw:
                        continue

                    total_reward[src_sw, dst_sw] += flow.transferred_bytes
                    flow.transferred_bytes = 0

            if flow.finished == True:
                del flow
                del runningFlows[flowID]

        # Determine total reward for individual link
        self._total_space = self._num_sws * (self._num_sws - 1) / 2
        selected_reward = np.zeros((1, self._total_space))
        ind = 0
        for i in xrange(0, self._num_sws):
            for j in xrange(0, i):
                selected_reward[0, ind] += total_reward[i, j]/B_TO_GB
                selected_reward[0, ind] += total_reward[j, i]/B_TO_GB
                ind += 1

        self.selected_reward = selected_reward.tolist()
        return sorted(range(len(self.selected_reward[0])), key=lambda i: self.selected_reward[0][i])[-self._k_actions:]