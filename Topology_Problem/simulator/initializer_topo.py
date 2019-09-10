#!/usr/bin/env python
import copy
import networkx as nx

from ..config import *
from .link import *
from .node import *

class VL2Initializer:
    def __init__(self, simulator):
        self.simulator = simulator
        self.root = []
        self.aggr = []
        self.tor = []
        self.host = []
        self.simulator.topo = nx.DiGraph()

    def run(self):
        if LOGGING:
            self.simulator.logger.debug('Initializing VL2 topology with Da = %d and Di = %d' % (Da, Di))
        self.initialize_root_sw()
        self.initialize_aggr_sw()
        self.initialize_tor_sw()
        self.initialize_host()
        self.connect_aggr_to_root()
        self.connect_tor_to_aggr()
        self.connect_host_to_tor()
        if LOGGING:
            self.simulator.logger.debug('VL2 topology with %d hosts, %d tor switches, %d aggr switches, %d root switches initialized' % (len(self.host), len(self.tor), len(self.aggr), len(self.root)))

        del self.root
        del self.aggr
        del self.tor
        del self.host

    def initialize_root_sw(self):
        for i in range(ROOT_SW_NUM):
            node_name = "root" + str(i)
            root = RootSwitch(node_name)
            self.simulator.sw_dict[node_name] = root
            self.simulator.topo.add_node(node_name)
            self.root.append(root)

    def initialize_aggr_sw(self):
        for i in range(AGGR_SW_NUM):
            node_name = "aggr" + str(i)
            aggr = AggrSwitch(node_name)
            self.simulator.sw_dict[node_name] = aggr
            self.simulator.topo.add_node(node_name)
            self.aggr.append(aggr)

    def initialize_tor_sw(self):
        for i in range(TOR_SW_NUM):
            node_name = "tor" + str(i)
            tor = ToRSwitch(node_name)
            self.simulator.sw_dict[node_name] = tor
            self.simulator.topo.add_node(node_name)
            self.tor.append(tor)
            self.simulator.tor_sw_list.append(tor)

    def initialize_host(self):
        for i in range(HOST_NUM):
            node_name = "host" + str(i)
            host = Host(node_name)
            self.simulator.host_dict[node_name] = host
            self.simulator.topo.add_node(node_name)
            self.host.append(host)

    def connect_aggr_to_root(self):
        for aggr in self.aggr:
            for root in self.root:
                    src_name = aggr.node_name
                    sink_name = root.node_name
                    bandwidth = LINK_BANDWIDTH_AGGR_TOR
                    up_link = Link(src_name, sink_name, bandwidth)
                    down_link = Link(sink_name, src_name, bandwidth)
                    aggr.up_link[sink_name] = up_link
                    root.down_link[src_name] = down_link

                    self.simulator.links[(src_name, sink_name)] = up_link
                    self.simulator.links[(sink_name, src_name)] = down_link

                    self.simulator.topo.add_edges_from([(src_name, sink_name)])
                    self.simulator.topo.add_edges_from([(sink_name, src_name)])

    def connect_tor_to_aggr(self):
        for i in range(0, Di / 2):
            for j in range(0, Da * Di / 8):
                aggr = self.aggr[i]
                tor = self.tor[j]

                src_name = tor.node_name
                sink_name = aggr.node_name
                bandwidth = LINK_BANDWIDTH_AGGR_TOR
                up_link = Link(src_name, sink_name, bandwidth)
                down_link = Link(sink_name, src_name, bandwidth)
                tor.up_link[sink_name] = up_link
                aggr.down_link[src_name] = down_link

                self.simulator.links[(src_name, sink_name)] = up_link
                self.simulator.links[(sink_name, src_name)] = down_link

                self.simulator.topo.add_edges_from([(src_name, sink_name)])
                self.simulator.topo.add_edges_from([(sink_name, src_name)])

        for i in range(Di / 2, Di):
            for j in range(Da*Di/ 8, Da*Di/4):
                aggr = self.aggr[i]
                tor = self.tor[j]

                src_name = tor.node_name
                sink_name = aggr.node_name
                bandwidth = LINK_BANDWIDTH_AGGR_TOR
                up_link = Link(src_name, sink_name, bandwidth)
                down_link = Link(sink_name, src_name, bandwidth)
                tor.up_link[sink_name] = up_link
                aggr.down_link[src_name] = down_link

                self.simulator.links[(src_name, sink_name)] = up_link
                self.simulator.links[(sink_name, src_name)] = down_link

                self.simulator.topo.add_edges_from([(src_name, sink_name)])
                self.simulator.topo.add_edges_from([(sink_name, src_name)])

    def connect_host_to_tor(self):
        for i in range(0, Da * Di / 4):
            for j in range(0,20):
                host = self.host[i*20+j]
                tor = self.tor[i]

                src_name = host.node_name
                sink_name = tor.node_name
                bandwidth = LINK_BANDWIDTH_TOR_HOST
                up_link = Link(src_name, sink_name, bandwidth)
                down_link = Link(sink_name, src_name, bandwidth)
                host.up_link[sink_name] = up_link
                tor.down_link[src_name] = down_link

                self.simulator.links[(src_name, sink_name)] = up_link
                self.simulator.links[(sink_name, src_name)] = down_link

                self.simulator.topo.add_edges_from([(src_name, sink_name)])
                self.simulator.topo.add_edges_from([(sink_name, src_name)])


class FatTreeInitializer:
    # @timing_function
    def __init__(self,simulator):
        self.simulator = simulator

        self.root = []
        self.aggr = []
        self.tor = []
        self.host = []
        self.pod = []
        self.simulator.topo = nx.DiGraph()

    # @timing_function
    def run(self):
        if LOGGING:
            self.simulator.logger.debug('Initializing FatTree topology with k = %d' % (K))
        self.initialize_root_sw()
        self.initialize_aggr_sw()
        self.initialize_tor_sw()
        self.initialize_host()
        self.initialize_pod()
        self.connect_aggr_to_root()
        self.connect_tor_to_aggr()
        self.connect_host_to_tor()
        # self.populate_path_dict()
        if LOGGING:
            self.simulator.logger.debug('FatTree with %d hosts, \
                                         %d tor switches, %d aggr switches, \
                                         %d root switches initialized' % 
                                         (len(self.host), 
                                          len(self.tor), 
                                          len(self.aggr), 
                                          len(self.root)))

        del self.root
        del self.aggr
        del self.tor
        del self.host
        del self.pod ## Clear Memory -- Variables aren't needed afterwards.

    # @timing_function
    def initialize_root_sw(self):
        
        for i in range(ROOT_SW_NUM):
            node_name = "root" + str(i)
            root = RootSwitch(node_name)
            self.simulator.sw_dict[node_name] = root
            self.simulator.topo.add_node(node_name)
            self.root.append(root)
        # print self.simulator.topo

    # @timing_function
    def initialize_aggr_sw(self):
        for i in range(AGGR_SW_NUM):
            node_name = "aggr" + str(i)
            aggr = AggrSwitch(node_name)
            self.simulator.sw_dict[node_name] = aggr
            self.simulator.topo.add_node(node_name)
            self.aggr.append(aggr)

    # @timing_function
    def initialize_tor_sw(self):
        for i in range(TOR_SW_NUM):
            node_name = "tor" + str(i)
            tor = ToRSwitch(node_name)
            self.simulator.sw_dict[node_name] = tor
            self.simulator.topo.add_node(node_name)
            self.tor.append(tor)
            self.simulator.tor_sw_list.append(tor)

    # @timing_function
    def initialize_host(self):
        for i in range(HOST_NUM):
            node_name = "host" + str(i)
            host = Host(node_name)
            self.simulator.host_dict[node_name] = host
            self.simulator.topo.add_node(node_name)
            self.host.append(host)

    # @timing_function
    def initialize_pod(self):
        aggr_num = 0
        tor_num = 0
        for i in range(POD_NUM):
            aggr = []
            tor = []
            for j in range(AGGR_SW_NUM // POD_NUM):
                aggr.append(self.aggr[aggr_num])
                aggr_num += 1
            for j in range(TOR_SW_NUM // POD_NUM):
                tor.append(self.tor[tor_num])
                tor_num += 1

            self.pod.append((aggr, tor))

    # @timing_function
    def connect_aggr_to_root(self):
        for pod in self.pod:
            aggr_num = len(pod[0])
            each_aggr_connect_num = ROOT_SW_NUM // aggr_num
            root_num = 0
            for aggr in pod[0]:
                for i in range(each_aggr_connect_num):
                    root = self.root[root_num]
                    src_name = aggr.node_name
                    sink_name = root.node_name
                    bandwidth = LINK_BANDWIDTH_AGGR_TOR
                    up_link = Link(src_name, sink_name, bandwidth)
                    down_link = Link(sink_name, src_name, bandwidth)
                    aggr.up_link[sink_name] = up_link
                    root.down_link[src_name] = down_link
                    root_num += 1

                    self.simulator.links[(src_name, sink_name)] = up_link
                    self.simulator.links[(sink_name, src_name)] = down_link
                    self.simulator.topo.add_edges_from([(src_name, sink_name)])
                    self.simulator.topo.add_edges_from([(sink_name, src_name)])

    # @timing_function
    def connect_tor_to_aggr(self):
        for pod in self.pod:
            for aggr in pod[0]:
                for tor in pod[1]:
                    src_name = tor.node_name
                    sink_name = aggr.node_name
                    bandwidth = LINK_BANDWIDTH_AGGR_TOR
                    up_link = Link(src_name, sink_name, bandwidth)
                    down_link = Link(sink_name, src_name, bandwidth)
                    tor.up_link[sink_name] = up_link
                    aggr.down_link[src_name] = down_link

                    self.simulator.links[(src_name, sink_name)] = up_link
                    self.simulator.links[(sink_name, src_name)] = down_link
                    self.simulator.topo.add_edges_from([(src_name, sink_name)])
                    self.simulator.topo.add_edges_from([(sink_name, src_name)])

    # @timing_function
    def connect_host_to_tor(self):
        each_tor_connect_num = HOST_NUM // TOR_SW_NUM
        host_id = 0
        for pod in self.pod:
            for tor in pod[1]:
                for i in range(each_tor_connect_num):
                    host = self.host[host_id]
                    src_name = host.node_name
                    sink_name = tor.node_name
                    bandwidth = LINK_BANDWIDTH_TOR_HOST
                    up_link = Link(src_name, sink_name, bandwidth)
                    down_link = Link(sink_name, src_name, bandwidth)
                    host.up_link[sink_name] = up_link
                    tor.down_link[src_name] = down_link
                    host_id += 1

                    self.simulator.links[(src_name, sink_name)] = up_link
                    self.simulator.links[(sink_name, src_name)] = down_link
                    self.simulator.topo.add_edges_from([(src_name, sink_name)])
                    self.simulator.topo.add_edges_from([(sink_name, src_name)])

class JellyfishInitialer:
    def __init__(self, simulator):
        self.simulator = simulator
        self.tor = []
        self.open = []
        self.closed = []
        
        self.host = []
        self.simulator.topo = nx.DiGraph()

    def run(self):
        if LOGGING:
            self.simulator.logger.debug('Initializing Jellyfish topology with Switches = %d ' +
                'and Ports per switches: %d' % (1,1))
        self.initialize_tor_sw()
        self.initialize_host()
        self.connect_tor_to_tor()
        self.connect_host_to_tor()

        if LOGGING:
            self.simulator.logger.debug('VL2 topology with %d hosts, %d tor switches.' % (len(self.host), len(self.tor)))

    def initialize_tor_sw(self):
        for i in range(TOR_SW_NUM):
            node_name = "tor" + str(i)
            tor = ToRSwitch(node_name)
            self.simulator.sw_dict[node_name] = tor
            self.simulator.topo.add_node(node_name)
            self.tor.append(tor)
            self.open.append(tor.node_name)
            self.simulator.tor_sw_list.append(tor)

    def initialize_host(self):  
        for i in range(HOST_NUM):
            node_name = "host" + str(i)
            host = Host(node_name)
            self.simulator.host_dict[node_name] = host
            self.simulator.topo.add_node(node_name)
            self.host.append(host)

    def connect_tor_to_tor(self):
        self.add_links() # ToDO: Complete This!

        node = random.choice(self.open)
        self.swapLinks(node) # Implement this!

    def swapLinks(self, node):
        pass

    def add_links(self):
        oldOpen = copy.deepcopy(self.open)

        for src_name in oldOpen:
            if src_name not in self.open:
                continue

            for sink_name in oldOpen:
                if src_name not in self.open:
                    break
                if sink_name not in self.open:
                    continue

                if (src_name != sink_name) and \
                    self.simulator.topo.has_edge(src_name, sink_name) == False:
                        self.simulator.topo.add_edges_from([(src_name, sink_name)])
                        self.simulator.topo.add_edges_from([(sink_name, src_name)])

                        self.relocate_nodes([src_name, sink_name])

    def relocate_nodes(self, nodes):
        for node in nodes:
            if len(self.simulator.topo[node]) < EDGES_PER_NODE and node in self.closed:
                self.closed.remove(node)
                self.open.append(node)
            elif len(self.simulator.topo[node]) == EDGES_PER_NODE and node in self.open:
                self.open.remove(node)
                self.closed.append(node)

    def connect_host_to_tor(self):
        for tor_node in xrange(0, len(self.tor)):
            for host_node in xrange(tor_node * HOSTS_PER_NODE, tor_node * HOSTS_PER_NODE + HOSTS_PER_NODE):
                src = self.host[host_node]
                sink = self.tor[tor_node]

                src_name = src.node_name
                sink_name = sink.node_name
                
                bandwidth = LINK_BANDWIDTH_TOR_HOST
                
                up_link = Link(src_name, sink_name, bandwidth)
                down_link = Link(sink_name, src_name, bandwidth)
                
                src.up_link[sink_name] = up_link
                sink.down_link[src_name] = down_link

                self.simulator.topo.add_edges_from([(src_name, sink_name)])
                self.simulator.topo.add_edges_from([(sink_name, src_name)])