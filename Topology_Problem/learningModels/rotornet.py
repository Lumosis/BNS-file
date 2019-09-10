from network_simulator_interface import *
import numpy as np

class RotorNet(NetworkSimulator):
    def __init__(self, simulator, num_iterations, step_size, k_actions,
                 a_size):
        self._monitor = threading.Event()
        self._monitor.clear()
        self._simulator = simulator(self._monitor, 'RoterNetSolution')
        self._num_iterations = num_iterations
        self._step_size = step_size
        self._k_actions = k_actions
        self._a_size = a_size

    def run(self):
        self._simulator.reset()

        while self._monitor.isSet():
            pass

        self.set_max_reward(100.0)
        self.set_period(self._step_size)

        self.read_node_list()

        matchings = self.find_matchings()

        counter = 0
        while (1):
            nextActions = self.parse_actions(matchings[counter % len(matchings)])

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

    def find_matchings(self):
        ret = []
        optical_lnk_index = 0

        for counter in xrange(1, self._num_sws):
            matching = []
            for i in xrange(self._num_sws):
                src_switch = 'tor' + str(i)
                dst_switch = 'tor' + str((i + counter) % (self._num_sws))
                matching.append([src_switch, dst_switch, optical_lnk_index])
                optical_lnk_index += 1
            ret.append(matching)

        return ret