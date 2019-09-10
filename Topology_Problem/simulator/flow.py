#!/usr/bin/env python
import networkx as nx
import numpy as np

from ..config import LOGGING
from ..config import TM_TYPE
from ..config import WORK_MODE
from ..config import CO_FLOWS

class Flow(object):
    # @timing_function
    def __init__(self, flow_id, job_id, flow_type, src_name, dst_name, tx_bytes, simulator):
        self.flow_id = flow_id
        self.job_id = job_id

        self.flow_type = flow_type
        self.src_name = src_name
        self.dst_name = dst_name

        self.tx_bytes = tx_bytes
        self.path = []

        self.current_speed = None  # byte per second
        self.last_update_time = None 
        self.remaining_tx_bytes = tx_bytes

        self.started = False
        self.finished = False

        self.start_time = None
        self.finish_time = None 

        self.simulator = simulator
        self.job = self.simulator.job_dict[self.job_id]
        self.simulator.flow[flow_id] = self

        self.optical_link_id = None
        self.transferred_bytes = 0

        self.routing()

    # @timing_function
    def start(self):
        if self.job.started is False:
            self.simulator.logger.debug("Time: %f Job: %d started" %
                                        (self.job.start_time, self.job.job_id))
            self.job.started = True
        if LOGGING:
            self.simulator.logger.debug("Time: %f Job: %d flow %s -> %s %d bytes, started to transmit" % (self.start_time, self.job.job_id, self.src_name, self.dst_name, self.tx_bytes))

        self.started = True

        if TM_TYPE == 'tor' and len(self.path) <= 3:
            self.finish_time = self.start_time
            self.finish()
            return

        if self.src_name == self.dst_name:
            self.finish_time = self.start_time
            self.finish()
        else:
            if WORK_MODE == "topology":
                src_tor = self.simulator.sw_dict[self.path[1]]
                dst_tor = self.simulator.sw_dict[self.path[-2]]
                src_tor.src_flows.append(self)
                dst_tor.dst_flows.append(self)

    # @timing_function
    def finish(self):
        self.finished = True
        if WORK_MODE == "topology":
            if self.src_name != self.dst_name:
                if TM_TYPE == 'host' or (TM_TYPE == 'tor' and len(self.path) > 3):
                    if self.optical_link_id is None:
                        # Not optical flows
                        src_tor = self.simulator.sw_dict[self.path[1]]
                        dst_tor = self.simulator.sw_dict[self.path[-2]]
                        src_tor.src_flows.remove(self)
                        dst_tor.dst_flows.remove(self)

                    else:
                        optical_link = self.simulator.optical_links[self.optical_link_id]
                        optical_link.flows.remove(self)
        #print self.current_speed
        if LOGGING:
            self.simulator.logger.debug("Time: %f Job: %d flow %s -> %s %d bytes, finished" % (self.finish_time, self.job.job_id, self.src_name, self.dst_name, self.tx_bytes))
        self.job.handle_finished_flow(self)

        if not CO_FLOWS:
            self.simulator.finished_flow_vals = np.append(self.simulator.finished_flow_vals, self.finish_time - self.start_time)
            self.simulator.flow_info.append([self.job.job_id, self.flow_id, self.start_time, self.finish_time, self.tx_bytes])

    # @timing_function
    def routing(self):
        if self.src_name == self.dst_name:
            return False

        self.path = nx.shortest_path(self.simulator.topo, self.src_name, self.dst_name)