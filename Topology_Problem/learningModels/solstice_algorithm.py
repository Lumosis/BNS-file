import gc

from network_simulator_interface import *

class Solstice(object):
    def QuickStuff(self, origDemandMatrix):
        # Input: Demand Matrix D [Square]
        # Output: k-bistochastic matrix D'

        # TODO: Add assertion for non - square matrix.

        demandMatrix = np.copy(origDemandMatrix)
        ROW = 0
        COLUMN = 1

        rowSums = np.sum(demandMatrix, COLUMN)
        columnSums = np.sum(demandMatrix, ROW)
        maxValue = max(np.max(rowSums), np.max(columnSums))


        size = np.size(demandMatrix, ROW)

        nonZeroIndices = np.nonzero(demandMatrix)
        numNonZeroIndices = np.size(nonZeroIndices[ROW])

        for i in range(numNonZeroIndices):
            currentRow = nonZeroIndices[ROW][i]
            currentColumn = nonZeroIndices[COLUMN][i]

            difference = maxValue - max(rowSums[currentRow], 
                                        columnSums[currentColumn])

            demandMatrix[currentRow, currentColumn] += difference
            rowSums[currentRow] += difference
            columnSums[currentColumn] += difference


        zeroIndices = np.where(demandMatrix == 0)
        numZeroIndices = np.size(zeroIndices[ROW])

        for i in range(numZeroIndices):
            currentRow = zeroIndices[ROW][i]
            currentColumn = zeroIndices[COLUMN][i]

            difference = maxValue - max(rowSums[currentRow], 
                                        columnSums[currentColumn])

            demandMatrix[currentRow, currentColumn] += difference
            rowSums[currentRow] += difference
            columnSums[currentColumn] += difference

        return demandMatrix


    def BigSlice(self, origMatrix, thresh):
        # Input: k-bistochastic matrix D', threshold r
        # Output: A permutation matrix P s.t.

        def FindBipartitePerfectMatching(matrix):
            ''' Leveraged Hopcroft-Karp to find the 
                maximal matching in a bipartite graph. '''

            # Input: Bipartite Matrix
            # Output: Perfect Matching of Bipartite Matrix

            from hopcroftkarp import HopcroftKarp
            import string

            bipartiteGraph = {}

            for rowNum in xrange(0, len(matrix)):
                key = string.lowercase
                bipartiteGraph[key[rowNum]] = set(np.where(matrix[rowNum] == 1)[0])

            perfectMatching = HopcroftKarp(bipartiteGraph).maximum_matching()

            matrix[:] = 0

            for key in perfectMatching:
                val = perfectMatching[key]
                if type(val) == type('a'):
                    matrix[string.lowercase.index(val), key] = 1

            return matrix


        matrix = np.copy(origMatrix)

        belowThresholdIndices = matrix < thresh
        aboveThresholdIndices = matrix >= thresh

        rowSums = np.sum(aboveThresholdIndices, 1)

        matrix[belowThresholdIndices] = 0
        matrix[aboveThresholdIndices] = 1

        if np.size(np.where(rowSums < 1)[0]) == 0:
            return FindBipartitePerfectMatching(matrix)
        else:
            return None

    def Solstice(self, demandMatrix, delta, rateCircuitSwitch, ratePacketSwitch):
        # Input: Demand Matrix D,
        #        Reconfiguratation Delay: delta,
        #        Link Rate of Circuit Switch: rateCircuitSwitch,
        #        Link Rate of Packet Switch: ratePacketSwitch

        # Output: m circuit configurations and durations: {P[i]}, {t[i]},
        #         demand sent to packet switch E
        E = np.copy(demandMatrix)
        demandMatrix = self.QuickStuff(demandMatrix)
        circuitConfiguration = []

        totalTime = 0
        r = 2 ** (int(np.log2(np.max(demandMatrix))))
        iterationNumber = 1

        ROW = 0
        COLUMN = 1

        rowSums = np.sum(demandMatrix, COLUMN)
        columnSums = np.sum(demandMatrix, ROW)

        temp = np.inf

        while (temp > 0):
            P = self.BigSlice(demandMatrix, r)
            # break

            if P is None:
                r = int(r / 2)
            else:
                t = np.min(demandMatrix[tuple(np.where(P == 1))])
                t = float(t) / rateCircuitSwitch
                demandMatrix = demandMatrix - rateCircuitSwitch * t * P
                E = E - rateCircuitSwitch * t * P
                E[E < 0] = 0
                totalTime = totalTime + t + delta
                iterationNumber += 1
                circuitConfiguration.append((P, t))

            rowSums = np.sum(demandMatrix, COLUMN)
            columnSums = np.sum(demandMatrix, ROW)

            temp =  np.size(np.where(rowSums >= ratePacketSwitch * totalTime)[0])
            temp += np.size(np.where(columnSums >= ratePacketSwitch * totalTime)[0])

        return circuitConfiguration, E

class SolsticeAlgorithm(NetworkSimulator):
    def __init__(self, simulator, step_size, k_actions, rP, rC):
        self._monitor = threading.Event()
        self._monitor.clear()
        self._simulator = simulator(self._monitor, 'SolsticeAlgorithm')
        self._step_size = step_size
        self._k_actions = k_actions
        self._rate_packet_switch = rP
        self._rate_circuit_switch = rC

    # @profile
    def run(self):
        self._simulator.reset()

        while self._monitor.isSet():
            pass

        self.set_max_reward(100.0)
        self.set_period(self._step_size)

        self.read_node_list()

        counter = 0
        while (1):
            nextActions = self._read_demand()

            if counter == 0:
                prevActions = []
            else:
                prevActions = [item for item in prevActions if \
                               item not in nextActions]

            self._simulator.step(nextActions, prevActions)

            while (self._monitor.isSet()):
                pass

            if self._simulator.is_done():
                break

            prevActions = nextActions
            counter = (counter + 1)

    # @profile
    def _read_demand(self):
        total_demand = np.zeros((self._num_sws, self._num_sws), 
                                   dtype=np.float)
        curr_time = float(self._simulator.scheduler.sim_clock)
        runningFlows = self._simulator.flow


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

            path = flow.path

            if len(path) >= 3:
                if 'tor' in path[1] and 'tor' in path[-2]:
                    src_sw = int(path[1][3:])
                    dst_sw = int(path[-2][3:])

                    if src_sw == dst_sw:
                        del runningFlows[flowID]
                        del flow
                        continue

                    total_demand[src_sw, dst_sw] += \
                        flow.remaining_tx_bytes

        solstice = Solstice()

        if np.max(total_demand) > 0:
            reconfig_time = 0
            res = solstice.Solstice(total_demand, reconfig_time, 
                                    self._rate_packet_switch,
                                    self._rate_circuit_switch)

            return self.parse_actions(res[0][0][0])
        else:
            return []