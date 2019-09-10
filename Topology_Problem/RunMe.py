"""
Contains the main method.

This file starts the whole
DeepConfig program.
"""

import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd + '/simulator')
sys.path.insert(0, cwd + '/learningModels')

from config import ALGORITHM
from config import K_ACTIONS
from config import LSTM_INCLUDED
from config import SIMULATION_TIME
from config import TIME_STEP
from config import A_SIZE
from config import TRACE_NAME
from config import LINK_BANDWIDTH_AGGR_TOR
from config import OPTICAL_BANDWIDTH
from config import DIRECTION
from config import GENERATE_TRACE
from config import MAX_REWARD

from edmonds_algorithm import EdmondsAlgorithm
from solstice_algorithm import SolsticeAlgorithm
from rotornet import RotorNet
from heuristic_solution import HeuristicSolution
from network_simulator_interface import NetworkSimulator
from optimal_solution import OptimalSolution
from random_algorithm import randomAlgo
from simulator import Simulator
from xweaver import xWeaver
from xweaver import xWeaver_Data_Generation
from parse_trace import Parse_Trace

import threading

if LSTM_INCLUDED is True:
    from dl_a3c_WITH_LSTM import RLModelSimulator
else:
    from dl_a3c_WITHOUT_LSTM import RLModelSimulator


def main():
    """
    Main Function.

    This is the main function for the Topoplogy Problem.
    Starts the simulator and the reinforcment learning module
    alongwith.
    """
    # sys.stdout = None

    print "STARTING SYSTEM ..."

    if GENERATE_TRACE is True:
        monitor = threading.Event()
        sim = Simulator(monitor, 'gen-trace')

        nm = NetworkSimulator(sim, monitor)
        nm.set_period(float(TIME_STEP))
        nm.set_actions(K_ACTIONS, A_SIZE)
        nm.set_max_reward(MAX_REWARD)
        nm.init()

        nm.reset()

        while nm.is_episode_finished() is False:
            nm.step([])

    else:
        if ALGORITHM == 'optimal':
            rl_model = OptimalSolution(Simulator,
                                       int(SIMULATION_TIME / TIME_STEP),
                                       TIME_STEP, K_ACTIONS)
            rl_model.run()

        elif ALGORITHM == 'deeplearning':
            rl_model = RLModelSimulator(Simulator, NetworkSimulator)
            rl_model.run()

        elif ALGORITHM == 'random' or ALGORITHM == 'none':
            from config import TOPOLOGY

            if TOPOLOGY == 'VL2':
                from config import Da, Di
                randomAlgo(Simulator, 0, A_SIZE, K_ACTIONS, TOPOLOGY, Da, Di,
                           ALGORITHM)
            elif TOPOLOGY == 'FATTREE':
                from config import K
                randomAlgo(Simulator, K, A_SIZE, K_ACTIONS, TOPOLOGY, 0, 0,
                           ALGORITHM)

        elif ALGORITHM == 'heuristic_1' or ALGORITHM == 'heuristic_2' or \
             ALGORITHM == 'heuristic_0':
            rl_model = HeuristicSolution(Simulator, TIME_STEP, \
                                            K_ACTIONS, ALGORITHM, DIRECTION)
            rl_model.run()

        elif ALGORITHM == 'edmonds_0' or ALGORITHM == 'edmonds_1':
            rl_model = EdmondsAlgorithm(Simulator, TIME_STEP, K_ACTIONS, \
                                        ALGORITHM, DIRECTION)
            rl_model.run()
        elif ALGORITHM == 'solstice':
            rl_model = SolsticeAlgorithm(Simulator, TIME_STEP, K_ACTIONS,
                                         LINK_BANDWIDTH_AGGR_TOR,
                                         OPTICAL_BANDWIDTH)
            rl_model.run()
        elif ALGORITHM == 'rotornet':
            rl_model = RotorNet(Simulator, int(SIMULATION_TIME / TIME_STEP),
                                TIME_STEP, K_ACTIONS, A_SIZE)
            rl_model.run()
        elif ALGORITHM == 'xWeaver':
            xWeaver_sys = xWeaver(Simulator, int(SIMULATION_TIME / TIME_STEP),
                                  TIME_STEP, K_ACTIONS)
            xWeaver_sys.run()

        elif ALGORITHM == "xWeaver_DG":
            xWeaver_sys = xWeaver_Data_Generation(
                Simulator, int(SIMULATION_TIME / TIME_STEP), TIME_STEP,
                K_ACTIONS)
            xWeaver_sys.run()
        elif ALGORITHM == 'Parsed':
            assert (TRACE_NAME != 'Parsed')
            xWeaver_sys = Parse_Trace(Simulator,
                                      int(SIMULATION_TIME / TIME_STEP),
                                      TIME_STEP, K_ACTIONS)
            xWeaver_sys.run()

    # elif ALGORITHM == 'HopcroftKarp':
    #   rl_model = HopcroftKarp(Simulator,
    #               int(SIMULATION_TIME / TIME_STEP),
    #               TIME_STEP, K_ACTIONS, ALGORITHM)
    #   rl_model.run()


if __name__ == "__main__":
    main()
    # randomAlgo()
