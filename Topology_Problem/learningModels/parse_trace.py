from network_simulator_interface import *
import cPickle as pickle

from config import PARSED_TRACE

class Parse_Trace(NetworkSimulator):
    def __init__(self, simulator, num_iterations, step_size, k_actions):
        self._monitor = threading.Event()
        self._monitor.clear()
        self._simulator = simulator(self._monitor, 'xWeaver')
        self._num_iterations = num_iterations
        self._step_size = step_size
        self._k_actions = k_actions

        self.flows = {}

    def run(self):
        np.set_printoptions(formatter={'float': lambda x: "{0:0.20f}".format(x)})

        self._simulator.reset()

        while self._monitor.isSet():
            pass

        self.set_max_reward(100.0)
        self.set_period(self._step_size)

        self.read_node_list()

        counter = 0
        while self._simulator.is_done() is False:
            self._simulator.step([], [])

            while (self._monitor.isSet()):
                pass

            t = self._read_traffic_matrix_HOST()

            if np.sum(t[:,:,1]) != 0:
                if counter not in self.flows:
                    self.flows[counter] = []

                indices = np.nonzero(t[:,:,1])
                indices = zip(indices[0], indices[1])

                for idx in indices:
                    self.flows[counter].append((idx[0], idx[1], t[:,:,1][idx]))
                counter += 1

        with open(PARSED_TRACE, "w") as job_file:
            job_file.write(pickle.dumps(self.flows))