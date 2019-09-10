from network_simulator_interface import *

def randomAlgo(Simulator, k, a_size, k_actions, topo, Da, Di, algo):
    import random

    def parse_actions(actions, K=k, topology=topo, Da=Da, Di=Di):
        res = []
        index = 0

        if topology == 'VL2':
            numSwitches = Da * Di / 4
        elif topology == 'FATTREE':
            numSwitches = K ** 2 / 2

        for i in xrange(0, numSwitches):
          for j in xrange(0, i):
            if index in actions:
              src_sw = 'tor%d' % i
              dst_sw = 'tor%d' % j

              res.append([src_sw, dst_sw, index + 1])
              res.append([dst_sw, src_sw, -1 * index])
            index = index + 1
        return res

    monitor = threading.Event()
    sim = Simulator(monitor, 'RANDOM')
    sim.reset()
    random.seed(1000)

    while monitor.isSet():
        pass

    counter = 0
    while (1):
        if algo == 'random':
            actions = [x for x in xrange(a_size)]
            actions = random.sample(actions, k_actions)

            if counter == 0:
                prevActions = []
            else:
                prevActions = [item for item in prevActions if item not in actions]

        elif algo == 'none':
            actions = []
            prevActions = []

        sim.step(parse_actions(actions), parse_actions(prevActions))

        while (monitor.isSet()):
            pass

        if sim.is_done():
            break

        prevActions = actions
        counter = (counter + 1)