"""
Traffic Matrix Initialer.

Classes:
    Parent Class: FBInitializer
    Inhertied Classes: SyntheticInitializer, ClouderaInitializer

Contains classes to parse specific csv in cases of
both FB and Cloudera traces to initialize the simulator.job_dict

In case of Synethetic Data, there is not trace present
but uses parameters provided in the config file
to initiliaze the simulator.job_dict
"""

import pickle

from ..config import *

if TRACE_NAME == 'Synthetic':
    from ..config import TECHNIQUE, TIME_STEP

    if TECHNIQUE == 'STRIDE':
        from ..config import STRIDE
elif TRACE_NAME == 'Parsed':
    from ..config import PARSED_TRACE


import random

from .job import FBJob
from .job import SyntheticJob
from .job import xWeaverJob
from .job import ParsedTraceJob


class FBInitializer(object):
    """
    FBInitializer Class.

    This class parses the FB Trace
    and initalizes the specific (simulator.job_dict)
    data structures.
    """

    # @timing_function
    def __init__(self, simulator, refresh=True):
        self.refresh = refresh
        self.simulator = simulator

    # @timing_function
    def run(self):
        if self.refresh:
            if "FB-2010_samples_24_times_1hr_0.tsv" in self.simulator.trace:
                self.parse_fb_trace_file()
                self.refresh = False
        self.loadTrace()

    # @timing_function
    def loadTrace(self):
        job_file = open(self.simulator.job_file, "r")
        job_id_count = 0
        job_file_data = job_file.readlines()

        if NEW_TRAINING_METHODOLOGY:
            if not TESTING:
                num_jobs = int(TOTAL_JOBS / (NUM_WORKERS * 1.0))
                job_start = int(self.simulator.worker_id * num_jobs)
                job_end = int(self.simulator.worker_id * num_jobs + num_jobs)
                job_file_data = job_file_data[job_start:job_end]


        if TESTING:
            job_start = (TESTING_JOB_START * 1.0) / 100.0 * len(job_file_data)
            job_end = (TESTING_JOB_END * 1.0) / 100.0 * len(job_file_data)
            job_file_data = job_file_data[int(job_start):int(job_end)]

        start_trace = True
        TIME_FACTOR = 0

        for line in job_file_data:
            try:
                line = line.split("\t")
                job_id = int(line[0])

                arrival_time = int(line[1])

                if NEW_TRAINING_METHODOLOGY:
                    if not TESTING:
                        if start_trace:
                            start_trace = False
                            TIME_FACTOR = arrival_time - 10

                        arrival_time -= TIME_FACTOR

                read_bytes = int(line[2])
                shuffle_bytes = int(line[3])
                write_bytes = int(line[4])

                read_node_list = eval(line[5])
                write_list = eval(line[8])
                mapper_list = eval(line[6])
                reducer_list = eval(line[7])

                if INIT_RANDOM:
                    # Random seed so that the simulator is deterministic.
                    random.seed(RANDOM_SEED)

                count = 0
                while count != SCALE_OUT_FACTOR:
                    job_id = job_id_count
                    arrival_time = arrival_time + random.randint(0, RANDOM_FACTOR)
                    # print arrival_time

                    if TESTING:
                    	arrival_time = int(arrival_time * 1.0 / INTERARRIVAL_FACTOR)
                    job = FBJob(job_id, arrival_time, self.simulator)

                    job.total_read_bytes = int(read_bytes * 1.0 * SCALE_UP_FACTOR)
                    job.total_shuffle_bytes = int(shuffle_bytes * SCALE_UP_FACTOR)
                    job.total_write_bytes = int(write_bytes * 1.0 * SCALE_UP_FACTOR)

                    job.read_node_list = read_node_list
                    job.write_node_list = write_list
                    job.mapper_list = mapper_list
                    job.reducer_list = reducer_list

                    self.simulator.job_dict[job_id] = job

                    count += 1
                    job_id_count += 1

                    if job_id_count == TOTAL_JOBS:
                        break

                if job_id_count == TOTAL_JOBS:
                    break

            except Exception as E:
                print(E)
                continue

        job_file.close()

        self.simulator.job_cutoff = job_id_count

        if LOGGING:
            self.simulator.logger.debug('%d jobs initialized' % len(self.simulator.job_dict))

    # @timing_function
    def parse_fb_trace_file(self):
        MAX_MAPPER_NUM = int(HOST_NUM)
        MAX_REDUCER_NUM =  int(HOST_NUM)

        trace_file = open(self.simulator.trace, "r")
        job_file = open(self.simulator.job_file, "w")
        job_id_count = 0

        trace_file_data = trace_file.readlines()
        trace_file.close()

        for line in trace_file_data:
            line = line.split("\t")
            job_id_str = str(line[0])
            job_id = int(job_id_str[3:])

            arrival_time = int(line[1])

            gap_time = int(line[2])# What is the gap time?
            read_bytes = int(line[3])
            shuffle_bytes = int(line[4])
            write_bytes = int(line[5])
            if SIMULATION_TIME != -1:
                if arrival_time > SIMULATION_TIME:
                    break
            # Override the job has 0 bytes
            if read_bytes == 0 or shuffle_bytes == 0 or write_bytes == 0:
                continue
            # Override very large shuffle jobs (>100G)
            if shuffle_bytes > 100 * 1000 * 1000 * 1000:
                continue

            read_num = read_bytes // CHUNK_SIZE
            if read_num < 1:
                read_num = 1
            elif read_num > MAX_READ_NODE_NUM:
                read_num = MAX_READ_NODE_NUM

            reducer_num = write_bytes // CHUNK_SIZE
            if reducer_num < 1:
                reducer_num = 1
            elif reducer_num > MAX_REDUCER_NUM:
                reducer_num = MAX_REDUCER_NUM

            mapper_num = shuffle_bytes // CHUNK_SIZE
            if mapper_num < 1:
                mapper_num = 1
            elif mapper_num > MAX_MAPPER_NUM:
                mapper_num = MAX_MAPPER_NUM

            # if read_bytes < read_num * mapper_num:
            #   mapper_num = CHUNK_SIZE

            # if shuffle_bytes < mapper_num * reducer_num:
            #   reducer_num = CHUNK_SIZE

            if INIT_RANDOM:
                # Random seed so that the simulator is deterministic.
                random.seed(RANDOM_SEED)

            job_id = job_id_count

            job_id_count += 1
            read_nodes_list = random.sample(self.simulator.host_dict.keys(),
                                            read_num)
            mapper_nodes_list = random.sample(self.simulator.host_dict.keys(),
                                              mapper_num)
            reducer_nodes_list = random.sample(self.simulator.host_dict.keys(),
                                               reducer_num)
            write_nodes_list = random.sample(self.simulator.host_dict.keys(),
                                             WRITE_NODE_NUM)

            line = "%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%s\n" % (job_id, arrival_time,
                                                             read_bytes,
                                                             shuffle_bytes,
                                                             write_bytes,
                                                             str(read_nodes_list),
                                                             str(mapper_nodes_list),
                                                             str(reducer_nodes_list),
                                                             str(write_nodes_list))

            job_file.write(line)

            if job_id_count == TOTAL_JOBS:
                break

        job_file.close()

class SyntheticInitializer(FBInitializer):
    def __init__(self, simulator, refresh = True):
        super(SyntheticInitializer, self).__init__(simulator, refresh)

    def run(self):
        if 'Synthetic' in self.simulator.trace:
            self.loadTrace()

    def loadTrace(self):
        if INIT_RANDOM == True:
            random.seed(RANDOM_SEED)

        hosts = self.simulator.host_dict.keys()

        job_id = 0
        arrival_time = int(TIME_STEP * 2.5)

        if TECHNIQUE == 'STRIDE' or TECHNIQUE == 'RANDOM':
            for _ in xrange(TOTAL_JOBS):
                for idx in xrange(len(hosts)):
                    job = SyntheticJob(job_id, arrival_time, 
                                       self.simulator)

                    job.total_read_bytes = 0
                    job.total_shuffle_bytes = 0
                    job.total_write_bytes = 0

                    job.read_node_list = [hosts[idx]]

                    if TECHNIQUE == 'STRIDE':
                        job.mapper_list = [hosts[(idx + STRIDE) % len(hosts)]]
                    elif TECHNIQUE == 'RANDOM':
                        dest_idx = idx
                        while dest_idx == idx:
                            dest_idx = random.randint(0, len(hosts) - 1)

                        job.mapper_list = [hosts[dest_idx]]
                    self.simulator.job_dict[job_id] = job

                    job_id += 1
                arrival_time += TIME_STEP

        elif TECHNIQUE == 'SHUFFLE':
            for _ in xrange(TOTAL_JOBS):
                for source in hosts:
                    for dest in hosts:
                        if source == dest:
                            continue

                        job = SyntheticJob(job_id, arrival_time, 
                                           self.simulator)

                        job.total_read_bytes = 0
                        job.total_shuffle_bytes = 0
                        job.total_write_bytes = 0

                        job.read_node_list = [source]
                        job.mapper_list = [dest]

                        self.simulator.job_dict[job_id] = job
                        job_id += 1
                arrival_time += TIME_STEP
                    
        self.simulator.job_cutoff = job_id

class ClouderaInitializer(FBInitializer):
    def __init__(self, simulator, refresh = True):
        super(ClouderaInitializer, self).__init__(simulator, refresh)

    def run(self):
        if "CC-b-clearnedSummary.tsv" in self.simulator.trace:
            self.parse_cc_trace_file()
            self.refresh = False
        self.loadTrace()

    def parse_cc_trace_file(self):
        MAX_MAPPER_NUM = int(HOST_NUM)
        MAX_REDUCER_NUM =  int(HOST_NUM)

        MAX_READ_NODE_NUM = 3
        WRITE_NODE_NUM = 1
        trace_file = open(self.simulator.trace, "r")
        job_file = open(self.simulator.job_file, "w")
        job_id_count = 0

        lines = trace_file.readlines()
        start_time = int(lines[0].split("\t")[5])
        lines = [line.split("\t") for line in lines]
        lines = sorted(lines, key = lambda line: int(line[5]))

        #print start_time
        for line in lines:
            #job_id_str = str(line[0])
            #job_id = int(job_id_str[3:])

            arrival_time = int(line[5]) - start_time
            arrival_time = int(arrival_time * 1.0 / float(INTERARRIVAL_FACTOR))

            read_bytes = int(line[2])
            shuffle_bytes = int(line[3])
            write_bytes = int(line[4])
            mapper_num = int(line[10])
            reducer_num = int(line[11])
            if SIMULATION_TIME != -1:
                if arrival_time > SIMULATION_TIME:
                    break
            # Override the job has 0 bytes
            if read_bytes == 0 or write_bytes == 0:
                continue
            # Override very large shuffle jobs (>100G)
            # if shuffle_bytes > 100 * 1000 * 1000 * 1000:
            #   continue

            read_num = read_bytes / CHUNK_SIZE
            if read_num < 1:
                read_num = 1
            elif read_num > MAX_READ_NODE_NUM:
                read_num = MAX_READ_NODE_NUM

            reducer_num = write_bytes / CHUNK_SIZE
            if reducer_num < 1:
                reducer_num = 1
            elif reducer_num > MAX_REDUCER_NUM:
                reducer_num = MAX_REDUCER_NUM

            mapper_num = shuffle_bytes / CHUNK_SIZE
            if mapper_num < 1:
                mapper_num = 1
            elif mapper_num > MAX_MAPPER_NUM:
                mapper_num = MAX_MAPPER_NUM

            # if read_bytes < read_num * mapper_num:
            #   mapper_num = CHUNK_SIZE

            # if shuffle_bytes < mapper_num * reducer_num:
            #   reducer_num = CHUNK_SIZE

            if INIT_RANDOM:
                # Random seed so that the simulator is deterministic.
                random.seed(RANDOM_SEED)

            job_id = job_id_count
            job_id_count += 1
            read_nodes_list = random.sample(self.simulator.host_dict.keys(), read_num)
            mapper_nodes_list = random.sample(self.simulator.host_dict.keys(), mapper_num)
            reducer_nodes_list = random.sample(self.simulator.host_dict.keys(), reducer_num)
            write_nodes_list = random.sample(self.simulator.host_dict.keys(), WRITE_NODE_NUM)

            line = "%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%s\n" % (job_id, arrival_time, read_bytes, shuffle_bytes, write_bytes, str(read_nodes_list), str(mapper_nodes_list), str(reducer_nodes_list), str(write_nodes_list))

            job_file.write(line)

            if job_id_count == TOTAL_JOBS:
                break

        trace_file.close()
        job_file.close()

class ParsedTraceInitializer(FBInitializer):
    def __init__(self, simulator, refresh = True):
        super(ParsedTraceInitializer, self).__init__(simulator, refresh)

    def run(self):
        if 'Parsed' in self.simulator.trace:
            self.loadTrace()

    def loadTrace(self):
        hosts = self.simulator.host_dict.keys()
        job_id = 0
        with open(PARSED_TRACE, 'r') as file_data:
            flows = pickle.loads(file_data.read())

        job = ParsedTraceJob(0, 0, self.simulator)
        job.total_read_bytes = 0

        ## ==========================================================
        ## Using the worker_id to shuffle between the jobs is a hack
        ## but I didn't want to create an extra variable.
        ## ==========================================================
        job.flows = flows[int(self.simulator.worker_id) % len(flows)]
        # print job.flows


        self.simulator.job_dict[job_id] = job
        self.simulator.job_cutoff = 1


class xWeaverInitializer(FBInitializer):
    def __init__(self, simulator, refresh = True):
        super(xWeaverInitializer, self).__init__(simulator, refresh)

    def run(self):
        if 'xWeaver' in self.simulator.trace:
            self.loadTrace()

    def loadTrace(self):
        hosts = self.simulator.host_dict.keys()
        job_id = 0

        for i in xrange(1):
            job = xWeaverJob(job_id, 0, self.simulator)

            job.total_read_bytes = 0
            job.total_shuffle_bytes = 0
            job.total_write_bytes = 0

            # random.seed(RANDOM_SEED)
            job.read_node_list = random.sample(hosts, len(hosts))
            job.mapper_list = random.sample(hosts, random.randint(1, len(hosts)))

        self.simulator.job_dict[job_id] = job
        self.simulator.job_cutoff = 1