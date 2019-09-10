"""Implementation of the Edmonds Algorithm."""

from network_simulator_interface import *
from helper import maxWeightMatching


class EdmondsAlgorithm(NetworkSimulator):


    def __init__(self, simulator, step_size, k_actions, algo, direction):
        self.monitor = threading.Event()
        self.monitor.clear()
        self.simulator = simulator(self.monitor, 'EdmondsAlgorithm')
        self.step_size = step_size
        self.k_actions = k_actions
        self.algo = algo
        self.direction = direction

    # @profile
    def run(self):
        self.simulator.reset()

        while self.monitor.isSet():
            pass

        self.set_max_reward(100.0)
        self.set_period(self.step_size)

        self.read_node_list()

        counter = 0
        while (1):
            nextActions = self.parse_actions(self.read_demand())

            if counter == 0:
                prevActions = []
            else:
                prevActions = [item for item in prevActions if \
                               item not in nextActions]

            self.simulator.step(nextActions, prevActions)

            while (self.monitor.isSet()):
                pass

            if self.simulator.is_done():
                break

            prevActions = nextActions
            counter = (counter + 1)

    # @profile
    def read_demand(self):
        total_reward = np.zeros((self.num_sws, self.num_sws), 
                                   dtype=np.float)
        curr_time = float(self.simulator.scheduler.sim_clock)
        runningFlows = self.simulator.flow


        for flowID in runningFlows.keys():
            flow = runningFlows[flowID]

            if flow.finished:
                del runningFlows[flowID]
                del flow
                continue

            srcHost = int(flow.src_name[4:])
            dstHost = int(flow.dst_name[4:])

            if srcHost == dstHost:
                del runningFlows[flowID]
                del flow
                continue

            if self.algo == 'edmonds_0':
                if flow.started == False:
                    continue

            path = flow.path

            if len(path) >= 3:
                if 'tor' in path[1] and 'tor' in path[-2]:
                    src_sw = int(path[1][3:])
                    dst_sw = int(path[-2][3:])

                    if src_sw == dst_sw:
                        del runningFlows[flowID]
                        del flow
                        continue

                    if self.algo == 'edmonds_0':
                        total_reward[src_sw, dst_sw] += \
                            flow.transferred_bytes
                        flow.transferred_bytes = 0
                    elif self.algo == 'edmonds_1':
                        total_reward[src_sw, dst_sw] += \
                            flow.remaining_tx_bytes


        # Determine total reward for individual link
        self.total_space = self.num_sws * (self.num_sws - 1) / 2
        opticalLinks = maxWeightMatching(total_reward)

        ind = 0
        opticalLinkIDS = {}
        for i in xrange(0, self.num_sws):
            for j in xrange(0, i):
                opticalLinkIDS[(i, j)] = ind
                ind += 1

        if self.direction == 'UNI':
            for j in xrange(0, self.num_sws):
                for i in xrange(0, j):
                    opticalLinkIDS[(i, j)] = ind
                    ind += 1

        res = []

        for lnk in opticalLinks:
            if tuple(lnk) in opticalLinkIDS:
                res.append(opticalLinkIDS[tuple(lnk)])
            elif self.direction == 'BI' and tuple((lnk[1], lnk[0])) in opticalLinkIDS:
                res.append(opticalLinkIDS[tuple((lnk[1], lnk[0]))])
            else:
                print "Error: This shouldn't be possible!"

        del opticalLinkIDS
        del opticalLinks
        del total_reward

        return res
