#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ..config import *
from .link import *

class FlowHandler:
    # @timing_function
    def __init__(self, simulator, optical_links_to_add = [], optical_links_to_remove = []):
        self.simulator = simulator
        self.optical_links_to_add = optical_links_to_add
        self.optical_links_to_remove = optical_links_to_remove
        self.time_step = int(self.simulator.scheduler.sim_clock * 1.0 / TIME_STEP) - 1 # Since we already add the sim_clock by a time step

    # @timing_function
    def run(self):
        while self.run_once():
            pass

    # @timing_function
    def run_once(self):
        self.schedule_all_flows()
        if GENERATE_TRACE is True:
            self.add_optical_links()
            self.remove_optical_links()
            self.iterate_optical_links()
        self.iterate_tor_sw()
        if len(self.simulator.scheduler.flow_to_schedule[self.time_step]) != 0:
            return True
        return False

    # @timing_function
    def schedule_all_flows(self):
        if self.time_step not in self.simulator.scheduler.flow_to_schedule:
            self.simulator.scheduler.flow_to_schedule[self.time_step] = []
        while len(self.simulator.scheduler.flow_to_schedule[self.time_step]) != 0:
            flow = self.simulator.scheduler.flow_to_schedule[self.time_step].pop()

            if flow.start_time > self.simulator.scheduler.sim_clock:
                self.simulator.logger.error("Flow schedule error, this should never happen")
            flow.start()
    
    # @timing_function
    def iterate_tor_sw(self):
        finished_flows = []
        for tor in self.simulator.tor_sw_list:
            src_flow_num = len(tor.src_flows)

            for flow in tor.src_flows:
                dst_tor = self.simulator.sw_dict[flow.path[-2]]
                dst_flow_num = len(dst_tor.dst_flows)

                srcBandwidthFactor = 1.0
                destBandwidthFactor = 1.0
                
                if len(flow.path) > 3:
                    if (tor.node_name, flow.path[2]) in self.simulator.failed_links:
                        srcBandwidthFactor = self.simulator.failed_links[(tor.node_name, flow.path[2])]

                    if (dst_tor.node_name, flow.path[-3]) in self.simulator.failed_links:
                        destBandwidthFactor = self.simulator.failed_links[(dst_tor.node_name, flow.path[-3])]

                bandwidth = min(float(LINK_BANDWIDTH_AGGR_TOR * srcBandwidthFactor) / src_flow_num, \
                                float(LINK_BANDWIDTH_AGGR_TOR * destBandwidthFactor) / dst_flow_num)

                # If allocated in previous iteration then restore the remaining tx bytes
                if flow.last_update_time == self.simulator.scheduler.sim_clock:
                    # Fix: if the flow starts within this time step, we should only restore the actual part
                    if (self.simulator.scheduler.sim_clock - flow.start_time) < TIME_STEP:
                        flow.remaining_tx_bytes += flow.current_speed * (self.simulator.scheduler.sim_clock - flow.start_time)
                    # Fix: Otherwise, we should restore the whole progress during this time step.
                    else:
                        flow.remaining_tx_bytes += flow.current_speed * TIME_STEP
          
                flow.current_speed = bandwidth / 8.0

                progress = 0
                if (self.simulator.scheduler.sim_clock - flow.start_time) < TIME_STEP:
                    progress = flow.current_speed * (self.simulator.scheduler.sim_clock - flow.start_time)
                else:
                    progress = flow.current_speed * TIME_STEP 

                #print flow.remaining_tx_bytes
                if flow.remaining_tx_bytes > progress:
                    flow.transferred_bytes = progress
                    flow.remaining_tx_bytes -= progress
                    flow.last_update_time = self.simulator.scheduler.sim_clock
                else:
                    flow.transferred_bytes = flow.remaining_tx_bytes
                    # Get the real finish time
                    elapsed_time = flow.remaining_tx_bytes / flow.current_speed
                    if flow.last_update_time is not None:
                        flow.finish_time = flow.last_update_time + elapsed_time  
                    else:
                        flow.finish_time = flow.start_time + elapsed_time
                    flow.remaining_tx_bytes = 0
                    finished_flows.append(flow)

        for flow in finished_flows:
            flow.finish()

    # @timing_function
    def add_optical_links(self):
        for src_tor, dst_tor, link_id in self.optical_links_to_add:
            self.add_optical_link(src_tor, dst_tor, link_id)
        self.optical_links_to_add = []
    
    # @timing_function
    def add_optical_link(self, src_tor, dst_tor, link_id):
        if link_id not in self.simulator.optical_links.keys():
            optical_link = OpticalLink(link_id, src_tor, dst_tor, OPTICAL_BANDWIDTH)
            self.simulator.optical_links[link_id] = optical_link
        else:
            optical_link = self.simulator.optical_links[link_id]
        src_tor = self.simulator.sw_dict[src_tor]

        for flow in src_tor.src_flows[:]:
            if dst_tor != flow.path[-2]:
                continue
                
            self.simulator.sw_dict[flow.path[1]].src_flows.remove(flow)
            self.simulator.sw_dict[flow.path[-2]].dst_flows.remove(flow)
            optical_link.flows.append(flow)
            flow.optical_link_id = link_id
    
    # @timing_function
    def remove_optical_links(self):
        for src_tor, dst_tor, link_id in self.optical_links_to_remove:
            self.remove_optical_link(src_tor, dst_tor, link_id)
        self.optical_links_to_remove = []
    
    # @timing_function
    def remove_optical_link(self, src_tor, dst_tor, link_id):
        if link_id not in self.simulator.optical_links:
            self.simulator.logger.error("Optical link error, this should never happen: " + str(link_id))
            return
        optical_link = self.simulator.optical_links[link_id]

        for flow in optical_link.flows[:]:
            flow.routing()
            # print flow.path[1]
            self.simulator.sw_dict[flow.path[1]].src_flows.append(flow)
            self.simulator.sw_dict[flow.path[-2]].dst_flows.append(flow)
            optical_link.flows.remove(flow)

            flow.optical_link_id = None
        self.simulator.optical_links.pop(link_id)
    
    # @timing_function
    def iterate_optical_links(self):
        finished_flows = []
        for link_id in self.simulator.optical_links:
            optical_link = self.simulator.optical_links[link_id]
            flow_num = len(optical_link.flows)

            for flow in optical_link.flows:

                bandwidth = float(OPTICAL_BANDWIDTH) / flow_num

                # If allocated in previous iteration then restore the remaining tx bytes
                if flow.last_update_time == self.simulator.scheduler.sim_clock:
                    # Fix: if the flow starts within this time step, we should only restore the actual part
                    if (self.simulator.scheduler.sim_clock - flow.start_time) < TIME_STEP:
                        flow.remaining_tx_bytes += flow.current_speed * (self.simulator.scheduler.sim_clock - flow.start_time)
                    # Fix: Otherwise, we should restore the whole progress during this time step.
                    else:
                        flow.remaining_tx_bytes += flow.current_speed * TIME_STEP

                flow.current_speed = bandwidth / 8.0
                
                progress = 0
                if (self.simulator.scheduler.sim_clock - flow.start_time) < TIME_STEP:
                    progress = flow.current_speed * (self.simulator.scheduler.sim_clock - flow.start_time)
                else:
                    progress = flow.current_speed * TIME_STEP

                if flow.remaining_tx_bytes > progress:
                    flow.transferred_bytes = progress
                    flow.remaining_tx_bytes -= progress
                    flow.last_update_time = self.simulator.scheduler.sim_clock

                else:
                    flow.transferred_bytes = flow.remaining_tx_bytes
                    # Get the real finish time
                    elapsed_time = flow.remaining_tx_bytes / flow.current_speed
                    if flow.last_update_time is not None:
                        flow.finish_time = flow.last_update_time + elapsed_time  
                    else:
                        flow.finish_time = flow.start_time + elapsed_time
                    flow.remaining_tx_bytes = 0
                    finished_flows.append(flow)

        for flow in finished_flows:
            flow.finish()