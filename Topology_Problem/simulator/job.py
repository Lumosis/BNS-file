#!/usr/bin/env python
import itertools
import random

from ..config import *
from .flow import *
from numpy.random import choice
from numpy.random import seed


class FBJob(object):
    # @timing_function
    def __init__(self, job_id, start_time, simulator):
        self.job_id = job_id
        self.start_time = start_time 
        self.finish_time = None

        self.started = False
        self.finished = False
        self.simulator = simulator

        self.read_node_list = []
        self.mapper_list = []
        self.reducer_list = []
        self.write_node_list = []

        self.read_flow_list = []
        self.shuffle_flow_list = []
        self.write_flow_list = []

        self.total_read_bytes = None
        self.total_shuffle_bytes = None 
        self.total_write_bytes = None

        self.finished_read_bytes = 0
        self.finished_shuffle_bytes = 0
        self.finished_write_bytes = 0
        # self.finshed_job_times = []

    # @timing_function
    def initialize_read_flow(self):
        flow_num = len(self.read_node_list) * len(self.mapper_list)
        tx_bytes_per_flow = int(self.total_read_bytes * 1.0 / flow_num)

        if tx_bytes_per_flow == 0:
        	tx_bytes_per_flow += 1

        self.total_read_bytes = tx_bytes_per_flow * flow_num
        for read_node_name, mapper_name in itertools.product(self.read_node_list, self.mapper_list):
            if self.simulator.flow_id_count >= TOTAL_FLOWS:
                break


            flow = Flow(self.simulator.flow_id_count, self.job_id, TYPE_READ_FLOW, read_node_name, mapper_name, tx_bytes_per_flow, self.simulator)
            self.simulator.flow_id_count += 1
            self.read_flow_list.append(flow)
            flow.start_time = self.start_time

    # @timing_function
    def initialize_shuffle_flow(self):
        flow_num = len(self.mapper_list) * len(self.reducer_list)

        tx_bytes_per_flow = int(self.total_shuffle_bytes * 1.0 / flow_num)

        if tx_bytes_per_flow == 0:
        	tx_bytes_per_flow += 1

        self.total_shuffle_bytes = tx_bytes_per_flow * flow_num

        for mapper_name, reducer_name in itertools.product(self.mapper_list, self.reducer_list):
            if self.simulator.flow_id_count >= TOTAL_FLOWS:
                break

            flow = Flow(self.simulator.flow_id_count, self.job_id, TYPE_SHUFFLE_FLOW, mapper_name, reducer_name, tx_bytes_per_flow, self.simulator)
            self.simulator.flow_id_count += 1
            self.shuffle_flow_list.append(flow)

    # @timing_function
    def initialize_write_flow(self):
        flow_num = len(self.reducer_list) * len(self.write_node_list)
        tx_bytes_per_flow = int(self.total_write_bytes * 1.0 / flow_num)

        ## If the flow_num are more than the total bytes the variable becomes 0.
        if tx_bytes_per_flow == 0:
        	tx_bytes_per_flow += 1

        self.total_write_bytes = tx_bytes_per_flow * flow_num

        for reducer_name, write_node_name in itertools.product(self.reducer_list, self.write_node_list):
            if self.simulator.flow_id_count >= TOTAL_FLOWS:
                break
                
            flow = Flow(self.simulator.flow_id_count, self.job_id, TYPE_WRITE_FLOW, reducer_name, write_node_name, tx_bytes_per_flow, self.simulator)
            self.simulator.flow_id_count += 1
            self.write_flow_list.append(flow)

    # @timing_function
    def schedule_read_flows(self):
        for flow in self.read_flow_list:

            self.simulator.scheduler.schedule_flow(flow)

    # @timing_function
    def schedule_shuffle_flows(self, start_time):
        for flow in self.shuffle_flow_list:
            flow.start_time = start_time
            self.simulator.scheduler.schedule_flow(flow)

    # @timing_function
    def schedule_write_flows(self, start_time):
        for flow in self.write_flow_list:
            flow.start_time = start_time
            self.simulator.scheduler.schedule_flow(flow)

    # @timing_function
    def handle_finished_flow(self, flow):
        if flow.flow_type == TYPE_READ_FLOW:

            self.finished_read_bytes += flow.tx_bytes
            if self.total_read_bytes == self.finished_read_bytes:
                if LOGGING:
                    self.simulator.logger.debug("Time: %f Job: %d reading finished" % (flow.finish_time, self.job_id))

                if self.total_shuffle_bytes != 0:
                    self.initialize_shuffle_flow()
                    self.schedule_shuffle_flows(flow.finish_time)
                else:
                    self.initialize_write_flow()
                    self.schedule_write_flows(flow.finish_time)

        elif flow.flow_type == TYPE_SHUFFLE_FLOW:
            self.finished_shuffle_bytes += flow.tx_bytes
            if self.total_shuffle_bytes == self.finished_shuffle_bytes:
                if LOGGING:
                    self.simulator.logger.debug("Time: %f Job: %d shuffle finished" % (flow.finish_time, self.job_id))

                self.initialize_write_flow()
                self.schedule_write_flows(flow.finish_time)
        elif flow.flow_type == TYPE_WRITE_FLOW:
            self.finished_write_bytes += flow.tx_bytes
            if self.total_write_bytes == self.finished_write_bytes:

                self.finish_time = flow.finish_time
                self.simulator.num_finished_jobs += 1
                self.finished = True

                if CO_FLOWS:
                    self.simulator.finished_job_vals = \
                      np.append(self.simulator.finished_job_vals, self.finish_time - self.start_time)

                if LOGGING:
                    self.simulator.logger.debug("Time: %f Job: %d writing finished" % (flow.finish_time, self.job_id))
                    self.simulator.logger.info("Time: %f Job: %d finished, elapsed time: %f, %d/%d jobs finished" % (self.finish_time, self.job_id, self.finish_time - self.start_time, self.simulator.num_finished_jobs, len(self.simulator.job_dict)))
                    # self.finshed_job_times.append(self.finish_time - self.start_time)
                for flow in self.read_flow_list[:]:
                    del flow
                
                del self.read_flow_list

                for flow in self.shuffle_flow_list[:]:
                    del flow
                
                del self.shuffle_flow_list

                for flow in self.write_flow_list[:]:
                    del flow

                del self.write_flow_list
                del self

class SyntheticJob(FBJob):
    def __init__(self, job_id, start_time, simulator):
        super(SyntheticJob, self).__init__(job_id, start_time, simulator)

    # @timing_function
    def initialize_read_flow(self):
        if INIT_RANDOM == True:
            seed(RANDOM_SEED)

        LONG_FLOW_PERCENTAGE = 1.0 - SHORT_FLOW_PERCENTAGE


        flow_size = choice([0, 1], None, 
                           p = [SHORT_FLOW_PERCENTAGE, LONG_FLOW_PERCENTAGE])
        if flow_size == 0: ## SHORT
            flow_size = random.randint(SHORT_SIZE_LOWER, SHORT_SIZE_UPPER)
        if flow_size == 1:
            flow_size = random.randint(LONG_SIZE_LOWER, LONG_SIZE_UPPER)

        for read_node_name, mapper_name in itertools.product(self.read_node_list, self.mapper_list):

            if self.simulator.flow_id_count >= TOTAL_FLOWS:
                break

            flow = Flow(self.simulator.flow_id_count, self.job_id, TYPE_READ_FLOW, read_node_name, mapper_name, flow_size, self.simulator)

            self.total_read_bytes += flow_size
            self.simulator.flow_id_count += 1
            self.read_flow_list.append(flow)
            flow.start_time = self.start_time

    # @timing_function
    def handle_finished_flow(self, flow):
        if flow.flow_type == TYPE_READ_FLOW:

            self.finished_read_bytes += flow.tx_bytes
            if self.total_read_bytes == self.finished_read_bytes:
                self.finish_time = flow.finish_time
                self.simulator.num_finished_jobs += 1

                if LOGGING:
                    self.simulator.logger.debug("Time: %f Job: %d writing finished" % (flow.finish_time, self.job_id))
                    self.simulator.logger.info("Time: %f Job: %d finished, elapsed time: %f, %d/%d jobs finished" % (self.finish_time, self.job_id, self.finish_time - self.start_time, self.simulator.num_finished_jobs, len(self.simulator.job_dict)))


class ParsedTraceJob(FBJob):
    def __init__(self, job_id, start_time, simulator):
        super(ParsedTraceJob, self).__init__(job_id, start_time, simulator)
        self.flows = []

    def initialize_read_flow(self):
        for flow in self.flows:
            src = ('host' + str(flow[0]))
            dst = ('host' + str(flow[1]))
            size = float(flow[2])

            if self.simulator.flow_id_count >= TOTAL_FLOWS:
                break

            flow = Flow(self.simulator.flow_id_count, self.job_id, TYPE_READ_FLOW, src, dst, size, self.simulator)

            self.total_read_bytes += size
            self.simulator.flow_id_count += 1
            self.read_flow_list.append(flow)
            flow.start_time = self.start_time

    def handle_finished_flow(self, flow):
        if flow.flow_type == TYPE_READ_FLOW:

            self.finished_read_bytes += flow.tx_bytes
            if self.total_read_bytes == self.finished_read_bytes:
                self.finish_time = flow.finish_time
                self.simulator.num_finished_jobs += 1

                if LOGGING:
                    self.simulator.logger.debug("Time: %f Job: %d writing finished" % (flow.finish_time, self.job_id))
                    self.simulator.logger.info("Time: %f Job: %d finished, elapsed time: %f, %d/%d jobs finished" % (self.finish_time, self.job_id, self.finish_time - self.start_time, self.simulator.num_finished_jobs, len(self.simulator.job_dict)))


class xWeaverJob(FBJob):
    def __init__(self, job_id, start_time, simulator):
        super(xWeaverJob, self).__init__(job_id, start_time, simulator)

    # @timing_function
    def initialize_read_flow(self):
        if INIT_RANDOM == True:
            seed(RANDOM_SEED)
        
        flow_size = MAX_SIZE * random.uniform(0, 1)

        for read_node_name, mapper_name in itertools.product(self.read_node_list, self.mapper_list):

            if self.simulator.flow_id_count >= TOTAL_FLOWS:
                break

            flow = Flow(self.simulator.flow_id_count, self.job_id, TYPE_READ_FLOW, read_node_name, mapper_name, flow_size, self.simulator)

            self.total_read_bytes += flow_size
            self.simulator.flow_id_count += 1
            self.read_flow_list.append(flow)
            flow.start_time = self.start_time

    # @timing_function
    def handle_finished_flow(self, flow):
        if flow.flow_type == TYPE_READ_FLOW:

            self.finished_read_bytes += flow.tx_bytes
            if self.total_read_bytes == self.finished_read_bytes:
                self.finish_time = flow.finish_time
                self.simulator.num_finished_jobs += 1

                if LOGGING:
                    self.simulator.logger.debug("Time: %f Job: %d writing finished" % (flow.finish_time, self.job_id))
                    self.simulator.logger.info("Time: %f Job: %d finished, elapsed time: %f, %d/%d jobs finished" % (self.finish_time, self.job_id, self.finish_time - self.start_time, self.simulator.num_finished_jobs, len(self.simulator.job_dict)))