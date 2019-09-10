"""
Simulator Class.

The simulator class has all the main data structures which contain
the topology, the jobs and flows data. It also drives the scheduler
so that the whole simulation runs.
"""

import gc
import logging
import numpy as np
import os

from ..config import *
from .initializer import Initializer
from .scheduler import Scheduler


class Simulator:
    """Simulator Class."""

    def __init__(self, monitor, sim_name="", worker_id = 0):
        """
        Method: __init__, Class: Simulator.

        Parameters:
            monitor: Semaphore to pause / unpause the simulator
            sim_name: Simulation Name which contains al the
                      simulation data.
        """
        self.topo = None  # Initialized while simulation starts
        self.worker_id = worker_id * 1.0

        self.scheduler = None  # Initialized when reset is called.
        self.name = sim_name
        self.sw_dict = {}
        self.tor_sw_list = []
        self.host_dict = {}

        self.num_finished_jobs = 0

        self.flow_info = []
        self.flow = {}
        self.failed_links = {}
        self.job_refresh = JOB_REFRESH

        self.epochCount = 0
        self.monitor = monitor

        self.job_cutoff = -1

        root_path = "./%s/" % sim_name
        if os.path.exists(root_path) is False:
            os.mkdir(root_path)
        if os.path.exists(root_path + "outputs/") is False:
            os.mkdir(root_path + "outputs/")
        if os.path.exists(root_path + "logs/") is False:
            os.mkdir(root_path + "logs/")

        # self.trace = "./simulator/data/FB-2010_samples_24_times_1hr_0.tsv"
        self.trace = "./Topology_Problem/simulator/data/" + TRACE_NAME
        self.job_file = ("./Topology_Problem/simulator/data/job_%d.tsv" % os.getpid())
        self.flow_file = "./Topology_Problem/simulator/data/flow.tsv"
        self.time_file = root_path + "logs/sim_time.log"

        debug_file = root_path + "logs/debug.log"
        info_file = root_path + "logs/info.log"
        self.logger = logging.getLogger("")
        self.logger.setLevel(logging.DEBUG)
        info_handler = logging.FileHandler(info_file, mode="w")
        info_formatter = logging.Formatter('%(levelname)-8s %(message)s')
        info_handler.setFormatter(info_formatter)
        info_handler.setLevel(logging.INFO)

        debug_handler = logging.FileHandler(debug_file, mode="w")
        # debug_handler = logging.NullHandler()
        debug_formatter = logging.Formatter('%(levelname)-8s %(message)s')
        debug_handler.setFormatter(debug_formatter)
        debug_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)-8s %(message)s',)
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(console_handler)

    def end(self):
        """
        End method for the simulator class.

        Ends the program.
        """
        if self.scheduler is not None:
            self.scheduler.isStopped = True
        self.monitor.set()

    def update_worker_id(self, worker_id):
        self.worker_id = worker_id * 1.0

    def reset(self):
        """
        Reset method for the simulator class.

        Resets the simulator for the next epoch.
        All the data structures are reset in this
        method.
        """
        # print "Current Time: %s" % time.localtime()
        gc.collect()  # Collect all unreferenced memory.
        self.epochCount += 1

        print("Epoch Number: %s" % self.epochCount)

        if self.epochCount == TOTAL_EPOCHS + 1:
            return None

        if (self.scheduler != None):
            self.scheduler.isStopped = True
        self.monitor.set() ## Monitor Set to True.

        self.scheduler = Scheduler(self, self.monitor)
        self.topo = None
        self.sw_dict = {}
        self.links = {}
        self.tor_sw_list = []
        self.host_dict = {}
        self.job_dict = {}

        self.num_finished_jobs = 0

        self.flow = {}
        self.optical_links = {}
        self.flow_id_count = 0
        
        if CO_FLOWS:
            self.finished_job_vals = np.array([])
        else:
            self.finished_flow_vals = np.array([])

        Initializer(self, self.job_refresh).run()

        """
        self.job_refresh should always be
        false in the later epochs.

        This is done so that the trace
        file isn't parsed in each epoch
        """
        self.job_refresh = False
        self.scheduler.isDone = False
        self.scheduler.start()

        return True

    def step(self, links_to_add, links_to_remove):
        """
        Method: step, Class: Simulator.

        Adds + Removes sets of optical links and runs
        the next step of the current epoch.
        """
        self.scheduler.links_to_add = links_to_add
        self.scheduler.links_to_remove = links_to_remove
        self.monitor.set()

    # @timing_function
    def is_done(self):
        """
        Method: isDone, Class: Simulator.

        Returns True when the current epoch
        is finished.
        """
        return self.scheduler.isDone
