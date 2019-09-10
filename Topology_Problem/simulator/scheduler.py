"""
Scheduler Class.

The scheduler class schedules processes the job dict
and scheduler the flows.
"""

# !/usr/bin/env python
import numpy as np
import os
import random
import threading
import time
import queue

from ..config import *

from .failure import FailureHandler
from .handler import FlowHandler

if FAILURES and FAILURES_AUTOMATION:
    from ..config import FAILURES_DATA


class Scheduler(threading.Thread):
    """Scheduler Class."""

    def __init__(self, simulator, monitor):
        """
        Method: __init__, Class: Simulator.

        Input Variables:
            simulator: Requires a simulator instance.
                       This is to access the job dictionary
                       and other simulator variables.
            sim_name: Simulation Name which contains al the
                      simulation data.
        """
        threading.Thread.__init__(self)
        self.simulator = simulator

        self.job_to_schedule = {}
        self.flow_to_schedule = {}
        self.sim_clock = 0
        self.monitor = monitor
        self.isDone = False

        self.links_to_add = []
        self.links_to_remove = []
        self.current_paths = []

        self.isStopped = False

    def schedule_flow(self, flow):
        flow_time_slot = int(flow.start_time * 1.0 / TIME_STEP)

        if flow_time_slot not in self.flow_to_schedule:
            self.flow_to_schedule[flow_time_slot] = []
        self.flow_to_schedule[flow_time_slot].append(flow)

    # @timing_function
    def pre_process_job_dict(self):
        # Distribute the job dictionary into each time slot
        for job_id in self.simulator.job_dict:
            job = self.simulator.job_dict[job_id]
            job_time_slot = int(job.start_time * 1.0 / TIME_STEP)
            # print "Time Slot: %d" % job_time_slot

            if job_time_slot not in self.job_to_schedule:
                self.job_to_schedule[job_time_slot] = []
            self.job_to_schedule[job_time_slot].append(job)

    # @timing_function
    def run(self):
        time_f = open(self.simulator.time_file, "w")
        start_log = "Simulator started at %s\n" % time.ctime()
        time_f.write(start_log)
        time_f.close()

        self.pre_process_job_dict()

        self.sim_clock -= TIME_STEP  # For Initialization

        while self.sim_clock < SIMULATION_TIME:
            time_slot = int(self.sim_clock * 1.0 / TIME_STEP)

            self.sim_clock += TIME_STEP

            if (self.sim_clock >= SIMULATION_TIME):
                self.isDone = True

            if (self.simulator.job_cutoff <= self.simulator.num_finished_jobs):
                self.isDone = True

            if PLACEMENT == 'SJF':
                try:
                    find_bytes = (lambda job: job.total_read_bytes + job.
                                  total_shuffle_bytes + job.total_write_bytes)
                    jobs = self.job_to_schedule[int(time_slot)]
                    jobs = sorted(jobs, key=find_bytes)
                    total_allocated_bytes = 0

                    for job in jobs[:]:
                        if job.start_time > self.sim_clock:
                            print("Error")

                        job.start_time = self.sim_clock - TIME_STEP + 1
                        job.initialize_read_flow()
                        job.schedule_read_flows()
                        total_allocated_bytes += find_bytes(job)
                        jobs.remove(job)

                        if total_allocated_bytes > MAX_BYTES:
                            break

                    if (int(time_slot) + 1) not in self.job_to_schedule:
                        self.job_to_schedule[int(time_slot) + 1] = jobs
                    else:
                        self.job_to_schedule[int(time_slot) + 1].extend(jobs)
                except KeyError as e:
                    pass

            elif PLACEMENT == 'RANDOM':
                try:
                    for job in self.job_to_schedule[int(time_slot)]:
                        if job.start_time > self.sim_clock:
                            print("Error")
                            self.simulator.logger.error(
                                "Scheduler error, this should never happen")
                        job.initialize_read_flow()
                        job.schedule_read_flows()
                except KeyError as e:
                    pass

            if self.sim_clock >= SIMULATION_TIME or \
               self.simulator.job_cutoff <= self.simulator.num_finished_jobs:
                break

            self.monitor.clear()
            self.monitor.wait()

            if self.isStopped:
                break

            if WORK_MODE == "topology":
                # if int(self.sim_clock) % LOGGING_TIME == 0:
                print("Date: %0.5f" % time.time(),)
                print("Worker-%d" % int(self.simulator.worker_id),)
                print("Current Epoch Number: %s" % self.simulator.epochCount,)
                print("Current Epoch Time: %s" % self.sim_clock)

                if FAILURES:
                    failure_handler = FailureHandler(self.simulator)

                flow_handler = FlowHandler(
                    self.simulator,
                    optical_links_to_add=self.links_to_add,
                    optical_links_to_remove=self.links_to_remove)

            if FAILURES:
                if FAILURES_AUTOMATION is False:
                    from config import FAILURES_DICT
                    if self.sim_clock in FAILURES_DICT:
                        failure_handler.run(FAILURES_DICT[self.sim_clock], [])
                if FAILURES_AUTOMATION:
                    if self.sim_clock == 0:
                        totalLinks = list(self.simulator.links.keys())

                        totalLinks = [
                            link for link in totalLinks
                            if (link[0].find('tor') != -1
                                and link[1].find('aggr') != -1)
                        ]

                        if FAILURES_DATA[0] == 'PERCENTAGE':
                            numFailedLinks = int(
                                (FAILURES_DATA[1]) / 100.0 * len(totalLinks))
                        elif FAILURES_DATA[0] == 'VALUE':
                            numFailedLinks = int(FAILURES_DATA[1])

                        print("Number of Failed Links: " + str(numFailedLinks))

                        random.seed(FAILURES_DATA[2])
                        failedLinks = random.sample(totalLinks, numFailedLinks)
                        failedLinks = [(link[0], link[1], 0.5)
                                       for link in failedLinks]
                        failure_handler.run(failedLinks, [])

            flow_handler.run()

        if CO_FLOWS:
            data = self.simulator.finished_job_vals
        else:
            data = self.simulator.finished_flow_vals

        if data.size != 0:
            if ALGORITHM != 'xWeaver_DG':
                results = open(RESULTS_FILE + '_' + str(os.getpid()), "a")
            else:
                results = open(RESULTS_FILE, "a")

            if FAILURES == True and True == FAILURES_AUTOMATION:
                results.write("FailedLinks: %s\n" % failedLinks)

            results.write("Counter: " + str(data.size) + " ")
            results.write(" " + str(np.min(data)))
            results.write(" " + str(np.percentile(data, 25)))
            results.write(" " + str(np.median(data)))
            results.write(" " + str(np.percentile(data, 75)))
            results.write(" " + str(np.percentile(data, 95)))
            results.write(" " + str(np.percentile(data, 99)))
            results.write(" " + str(np.percentile(data, 100)))
            results.write(" " + str(np.max(data)))
            results.write(" AFCT: " + str((np.average(data))) + '\n')
            results.close()

            if ALGORITHM != 'optimal':
                if TESTING and FLOW_INFO:
                    with open(FLOW_INFO_FILE, 'a') as f:
                        f.write(str(self.simulator.flow_info))
                    del self.simulator.flow_info

            # if ALGORITHM != 'xWeaver_DG':
            #     flowResults = open(FLOW_AFCT_FILE + '_' + str(os.getpid()), "a")
            #     flowResults.write(str(list(data)))
            #     flowResults.close()

        time_f = open(self.simulator.time_file, "a")
        start_log = "Simulator ended at %s\n" % time.ctime()
        time_f.write(start_log)
        time_f.close()

        self.monitor.clear()