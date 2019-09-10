"""
Configuration File.
This file has all the network
parameters + tunable hyperparameters
of the RL module.
These parameters are accessible from
any file in the codebase.
"""

import time


def timing_function(some_function):
    """Decorator Function."""

    def wrapper(*args, **kwargs):
        """Measure the time a function takes."""
        time_1 = time.time()
        ret = some_function(*args, **kwargs)
        time_2 = time.time()

        if (time_2 - time_1) > 0.01:
            print("Time it took to run " + str(type(args[0]).__name__) + ":" +
                  str((some_function.__name__)) + " : " +
                  str((time_2 - time_1)) + "\n")
        return ret

    return wrapper


"""
    deeplearning, optimal, random
    heuristic_1, heuristic_2, none,
    xWeaver, xWeaver_DG, rotornet
    heuristic_0, Parsed, solstice
    edmonds_0, edmonds_1
"""
ALGORITHM = "deeplearning"  ## SCHEME
WORK_MODE = "topology"  # Topology, Routing
"""
    BI means that if there is an optical link between
    TOR_1 and TOR_2 then it offloads traffic going between
    TOR_1 and TOR_2 in either direction.
    UNI means that there would be two optical links
    between TOR_1 and TOR_2 to serve traffic in
    either direction.
"""
DIRECTION = 'BI'  # UNI, BI
"""
    TOPOLOGY TYPE
    Supported: FATTREE, VL2
"""
TOPOLOGY = "FATTREE"
TF_STATS_FOLDER = "./Topology_Problem/Results/TF_STATS/"
RESULTS_FILE = "./Topology_Problem/Results/TRAIN/Results.txt"
"""
    How many future steps to predict?
    The higher the number, the higher the
    AFCT. Default: 1
"""
STEPS = 1

"""
    If GENERATE_TRACE is True
    then a TM_Trace would be generated and 
    written to File <TRAFFIC_MATRICES>
"""
GENERATE_TRACE = False
TRAFFIC_MATRICES = './Topology_Problem/Results/TRAIN/TM.txt'

TEST_TIMING_INFO = False
TEST_TIMING_FILE = './Topology_Problem/Results/TEST/Times.txt'

FLOW_INFO = False
FLOW_INFO_FILE = './Topology_Problem/Results/TEST/Flows.txt'

LOG_REWARDS = False
REWARDS_FILE = "./Topology_Problem/Results/TRAIN/Rewards.txt"

FLOW_AFCT_FILE = "./Topology_Problem/Results/TRAIN/Flow_Results.txt"
TRACE_NAME = "FB"

PLACEMENT = 'RANDOM'  # RANDOM, SJF
MAX_BYTES = 40 * 1000 * 1000 * 1000

if ALGORITHM == 'xWeaver':
    TRACE_NAME = 'FB'

MEMORY_STATISTICS = False

# Simulation parameter
NEW_TRAINING_METHODOLOGY = False
NUM_WORKERS = 16
TIME_STEP = 20
TOTAL_EPOCHS = 1
TM_TYPE = "tor"
CHECKPOINT_ITERATIONS = 200

TESTING = False
TOTAL_JOB_PERCENTAGE = 20
TESTING_JOB_START = 0
TESTING_JOB_END = 100
INTERARRIVAL_FACTOR = 1.0

SPARSE_INPUT = False
CO_FLOWS = False

NODE2VEC = False
NUM_DIMENSIONS = 16
NODE2VEC_P = 1
NODE2VEC_Q = 1
NUM_WALKS = 80
WALK_LENGTH = 160

INPUT_NORMALIZATION = True
LIVE_FLOW_TIMES = True
FINISHED_FLOW_TIMES = False
FLOW_SIZES = True

K_ACTIONS_PERCENTAGE = 20

FAILURES = False
FAILURES_AUTOMATION = False
FAILURES_FILE = "Failures.txt"
RL_MODEL_FILE = "./Topology_Problem/RL_Model.txt"

MAX_READ_NODE_NUM = 3
WRITE_NODE_NUM = 1

LEARNING_RATE = 1e-5
LEARNING_RATE_DECAY_FACTOR = 0.95
NUM_EPOCHS_PER_DECAY = 10
MAX_REWARD = 100.0
CLIPPING = True
CLIPPING_VALUE = 1e-10
LOAD_MODEL = True
GAMMA = 0.01
NUM_VALID_STEPS = 10

TOPOLOGY_INCLUDED = False
LSTM_INCLUDED = False

TOTAL_FLOWS = 1e100
SIMULATION_TIME = 10e6

LOGGING_TIME = 100  # UNIT: s
DISPLAY_TM = False
FLOW_COMPLETION_REWARD = 0
LOGGING = False
DEBUG_LOGGING = False
INTERPRETABLE_ML = False

INIT_RANDOM = True
RANDOM_SEED = 6

PARSED_TRACE = "./Topology_Problem/Results/TRAIN/Trace.txt"

if TRACE_NAME == 'FB':
    TRACE_NAME = 'FB-2010_samples_24_times_1hr_0.tsv'
elif TRACE_NAME == 'Cloudera':
    TRACE_NAME = 'CC-b-clearnedSummary.tsv'
elif TRACE_NAME == "Synthetic":
    TOTAL_JOBS = 10
    TECHNIQUE = 'RANDOM'  # STRIDE, RANDOM, SHUFFLE, ML
    SHORT_FLOW_PERCENTAGE = 0.5
    SHORT_SIZE_UPPER = 25 * 1000 * 1000  # Bytes
    SHORT_SIZE_LOWER = 0

    LONG_SIZE_LOWER = 25 * 1000 * 1000
    LONG_SIZE_UPPER = 100 * 1000 * 1000 * 1000  # Bytes

    if TECHNIQUE == 'STRIDE':
        STRIDE = 24
elif TRACE_NAME == 'Parsed':
    TOTAL_JOBS = 1

if ALGORITHM == 'xWeaver' or ALGORITHM == 'xWeaver_DG' or \
   ALGORITHM == 'xWeaver_Trace':
    NUM_SAMPLES = 250000

    TRAINING_ITEMS = 4500
    NUM_EPOCHS_SCORE = 100
    NUM_EPOCHS_MAIN = 500
    MAX_DEPTH = 6
    MAX_WIDTH = 4
    ETA = 0.7

    TOTAL_JOBS = 1
    MAX_SIZE = 1000 * 1000
    STRIDE = 3

    MODEL_FOLDER = "./Topology_Problem/Results/CHECKPOINTS/"
    TOPOLOGY_FILE = "./Topology_Problem/Results/TRAIN/Topology.txt"
    TM_FILE = "./Topology_Problem/Results/TRAIN/TM.txt"
elif ALGORITHM == 'optimal':
    TOTAL_EPOCHS += 10e5
"""
    After changing this parameter,
    the job should be refreshed.\
"""
JOB_REFRESH = True

if TOPOLOGY == 'FATTREE':
    K = 4
    HOST_NUM = int(K**3 / 4.0)
    POD_NUM = K
    SW_NUM = int(5 * (K**2) / 4.0)
    ROOT_SW_NUM = int(K**2 / 4.0)
    AGGR_SW_NUM = int(K**2 / 2.0)
    TOR_SW_NUM = int(K**2 / 2.0)

elif TOPOLOGY == 'JELLYFISH':
    TOR_SW_NUM = 8
    EDGES_PER_NODE = 6
    HOSTS_PER_NODE = 2
    HOST_NUM = TOR_SW_NUM * HOSTS_PER_NODE

elif TOPOLOGY == 'VL2':
    Da = 6
    Di = Da
    ROOT_SW_NUM = int(Da / 2.0)
    HOSTS_PER_TOR = 20
    HOST_NUM = (int((Da * Di) / 4.0) * HOSTS_PER_TOR)
    AGGR_SW_NUM = Di
    TOR_SW_NUM = int(Da * Di / 4.0)

NT_SIZE = TOR_SW_NUM

if (TOPOLOGY == 'FATTREE') or (TOPOLOGY == 'VL2'):
    if 'host' == TM_TYPE:
        TM_SIZE = HOST_NUM
    elif 'tor' == TM_TYPE:
        TM_SIZE = TOR_SW_NUM

if DIRECTION == 'BI':
    A_SIZE = int(NT_SIZE * (NT_SIZE - 1) / 2.0)
    K_ACTIONS = int(K_ACTIONS_PERCENTAGE / 100.0 * (A_SIZE * 1.0))
elif DIRECTION == 'UNI':
    A_SIZE = int(NT_SIZE * (NT_SIZE - 1))
    K_ACTIONS = int(NT_SIZE / 2.0)

if TOPOLOGY == 'FATTREE':
    SCALE_OUT_FACTOR = int(K / 4.0)
    SCALE_UP_FACTOR = float(K / 8.0) + 0.5
    RANDOM_FACTOR = int(((K / 2.0) - 2) * 10)
elif TOPOLOGY == 'VL2':
    SCALE_OUT_FACTOR = int((Da - 2) / 4.0)
    SCALE_UP_FACTOR = float(Da / 8.0) + 0.25
    RANDOM_FACTOR = int(((Da / 2.0) - 3) * 10)

if TRACE_NAME == 'FB-2010_samples_24_times_1hr_0.tsv':
    TOTAL_JOBS = int(TOTAL_JOB_PERCENTAGE * 1.0 / 100.0 * 14952.0 *
                     float(SCALE_OUT_FACTOR))
elif TRACE_NAME == 'CC-b-clearnedSummary.tsv':
    TOTAL_JOBS = int(TOTAL_JOB_PERCENTAGE * 1.0 / 100.0 * 2392.0 *
                     float(SCALE_OUT_FACTOR))

LINK_BANDWIDTH_ROOT_AGGR = 40 * 1000 * 1000 * 1000  # Gbps
LINK_BANDWIDTH_AGGR_TOR = 40 * 1000 * 1000 * 1000  # Gbps
LINK_BANDWIDTH_TOR_HOST = 40 * 1000 * 1000 * 1000  # Gbps
OPTICAL_BANDWIDTH = 100 * 1000 * 1000 * 1000  # Gbps

# Facebook trace parameters
CHUNK_SIZE = 256 * 1024 * 1024  # 256MB
TYPE_READ_TASK = 4
TYPE_SHUFFLE_TASK = 4
TYPE_WRITE_TASK = 4
TYPE_READ_FLOW = 4
TYPE_SHUFFLE_FLOW = 5
TYPE_WRITE_FLOW = 6

# Failures Data Reading
if FAILURES:
    if not FAILURES_AUTOMATION:
        f = open(FAILURES_FILE, 'r')
        failuresData = f.read().split()
        FAILURES_DICT = {}

        while "-1" != (failuresData[0]):
            linksList = []
            c = 2

            for i in xrange(0, int(failuresData[1])):
                linksList.append((failuresData[c], failuresData[c + 1],
                                  float(failuresData[c + 2])))
                c += 3

            FAILURES_DICT[int(failuresData[0])] = linksList
            failuresData = failuresData[c:] + failuresData[:c]

    if FAILURES_AUTOMATION:
        FAILURES_TYPE = 'PERCENTAGE'
        FAILURES_VALUE = 0
        FAILURES_SEED = 6

        FAILURES_DATA = [FAILURES_TYPE, FAILURES_VALUE, FAILURES_SEED]

# RL Model Layer Configurations
f = open(RL_MODEL_FILE, 'r')
RL_MODEL = [layer.split(' ') for layer in f.read().split('\n')[1:]]

if SPARSE_INPUT == True and NODE2VEC == True:
    assert False, "Both scalability techniques set to True."