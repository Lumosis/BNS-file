#!/usr/bin/env python
from .initializer_tm import *
from .initializer_topo import *
from .link import *
from .node import *

class Initializer:
    
    # @timing_function
    def __init__(self, simulator, job_refresh):
        self.simulator = simulator
        self.job_refresh = job_refresh

    # @timing_function
    def run(self):
        if LOGGING:
            self.simulator.logger.info('Initialization started')
        
        if TOPOLOGY == 'FATTREE':
            FatTreeInitializer(self.simulator).run()
        elif TOPOLOGY == 'VL2':
            VL2Initializer(self.simulator).run()
        elif TOPOLOGY == 'JELLYFISH':
            JellyfishInitialer(self.simulator).run()

        if TRACE_NAME == 'FB-2010_samples_24_times_1hr_0.tsv':
            FBInitializer(self.simulator, self.job_refresh).run()
        elif TRACE_NAME == 'CC-b-clearnedSummary.tsv':
            ClouderaInitializer(self.simulator, self.job_refresh).run()
        elif TRACE_NAME == 'Synthetic':
            SyntheticInitializer(self.simulator, self.job_refresh).run()
        elif TRACE_NAME == 'xWeaver':
            xWeaverInitializer(self.simulator, self.job_refresh).run()
        elif TRACE_NAME == 'Parsed':
            ParsedTraceInitializer(self.simulator, self.job_refresh).run()
            
        if LOGGING:
            self.simulator.logger.info('Initialization finished')
