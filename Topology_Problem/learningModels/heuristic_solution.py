import timeit
from network_simulator_interface import *
from config import TEST_TIMING_INFO
from config import TEST_TIMING_FILE
from config import TESTING


class HeuristicSolution(NetworkSimulator):
    def __init__(self, simulator, step_size, k_actions, algo, direction):
        self.monitor = threading.Event()
        self.monitor.clear()
        self.simulator = simulator(self.monitor, 'HeuristicSolution')
        self.step_size = step_size
        self.k_actions = k_actions
        self.algo = algo
        self.direction = direction

    def run(self):
        self.simulator.reset()

        while self.monitor.isSet():
            pass

        self.set_max_reward(100.0)
        self.set_period(self.step_size)

        self.read_node_list()

        counter = 0
        while (1):
            if TESTING and TEST_TIMING_INFO:
                start_time = timeit.default_timer()

            nextActions = self.parse_actions(self.read_demand())

            if TESTING and TEST_TIMING_INFO:
                end_time = timeit.default_timer()

                f = open(TEST_TIMING_FILE, 'a')
                f.write(str(end_time - start_time))
                f.write('\n')
                f.close()

            if counter == 0:
                prevActions = []
            else:
                prevActions = [item for item in prevActions if item not in nextActions]

            with open(TRAFFIC_MATRICES, 'a') as f:
                f.write(str(nextActions))
                f.write('\n\n\n')

            self.simulator.step(nextActions, prevActions)

            while (self.monitor.isSet()):
                pass

            if self.simulator.is_done():
                break

            prevActions = nextActions
            counter = (counter + 1)

    def read_demand(self):
        total_reward = np.zeros((self.num_sws, self.num_sws),
                                   dtype=np.float)
        curr_time = float(self.simulator.scheduler.sim_clock)
        runningFlows = self.simulator.flow

        # print "Read Reward Running"

        for flowID in runningFlows.keys():
            flow = runningFlows[flowID]

            if self.algo != 'heuristic_0':
                if flow.started == False:
                    continue

            path = flow.path

            if len(path) >= 3:
                if 'tor' in path[1] and 'tor' in path[-2]:
                    src_sw = int(path[1][3:])
                    dst_sw = int(path[-2][3:])

                    if src_sw == dst_sw:
                        continue

                    if self.algo == 'heuristic_1' or self.algo == 'heuristic_0':
                        total_reward[src_sw, dst_sw] += \
                                                flow.remaining_tx_bytes
                    elif self.algo == 'heuristic_2':
                        total_reward[src_sw, dst_sw] += \
                                            flow.transferred_bytes
                        flow.transferred_bytes = 0

            if flow.finished:
                del runningFlows[flowID]
                del flow

        np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})

        with open(TRAFFIC_MATRICES, 'a') as f:
            f.write(str(total_reward))
            f.write('\n')

        # Determine total reward for individual link
        if self.direction == 'BI':      
            self.total_space = self.num_sws * (self.num_sws - 1) / 2
        elif self.direction == 'UNI':
            self.total_space = self.num_sws * (self.num_sws - 1)

        selected_reward = np.zeros((1, self.total_space))
        ind = 0
        for i in xrange(0, self.num_sws):
            for j in xrange(0, i):
                selected_reward[0, ind] += total_reward[i, j] / B_TO_GB

                if self.direction == 'BI':
                    selected_reward[0, ind] += total_reward[j, i] / B_TO_GB
                ind += 1

        if self.direction == 'UNI':
            for j in xrange(0, self.num_sws):
                for i in xrange(0, j):
                    selected_reward[0, ind] += total_reward[i, j] / B_TO_GB
                    ind += 1

        # Clip the reward
        self.selected_reward = np.clip(selected_reward, 0.0, self.max_reward)
        self.selected_reward = self.selected_reward.tolist()

        return sorted(range(len(self.selected_reward[0])), 
                  key=lambda i: self.selected_reward[0][i])[-self.k_actions:]