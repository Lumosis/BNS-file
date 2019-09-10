"""
Failure Module.

This module contains the failure handler class.
"""


class FailureHandler:
    """
    FailureHandlerClass.

    This module is used to fail tor - aggr links.
    One can specific the link degradation from
    0 [No Failure] to 1 [Complete Failure] for
    any tor - aggr link.
    """

    def __init__(self, simulator):
        self.simulator = simulator

    def run(self, failedLinks, recoveredLinks):
        self.setLinkFailures(failedLinks)
        self.setLinkRecoveries(recoveredLinks)

    def setLinkFailures(self, links):
        for src, dst, degradation in links:

            if src[0:3] != 'tor' or dst[0:4] != 'aggr':
                self.simulator.logger.error('Incorrect Link Failure')

            if (src, dst) not in self.simulator.topo.edges:
                self.simulator.logger.error('Link not in Topology.')

            self.simulator.failed_links[(src, dst)] = float(degradation)

    def setLinkRecoveries(self, links):
        for src, dst in links:
            if src[0:3] != 'tor' or dst[0:4] != 'aggr':
                self.simulator.logger.error('Incorrect Link Recovery')

            if (src, dst) not in self.simulator.topo.edges:
                self.simulator.logger.error('Link not in Topology.')

            self.simulator.failed_links[(src, dst)] = 1.0