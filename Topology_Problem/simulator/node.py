#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ..config import *

class Switch:
    # @timing_function
    def __init__(self, node_name):
        self.node_name = node_name

class RootSwitch(Switch):
    # @timing_function
    def __init__(self, node_name):
        Switch.__init__(self, node_name)
        self.down_link = {}

class AggrSwitch(Switch):
    # @timing_function
    def __init__(self, node_name):
        Switch.__init__(self, node_name)
        self.up_link = {}
        self.down_link = {}

class ToRSwitch(Switch):
    # @timing_function
    def __init__(self, node_name):
        Switch.__init__(self, node_name)
        self.src_flows = []
        self.dst_flows = []
        self.up_link = {}
        self.down_link = {}

        self.optical_src_flows = []
        self.optical_dst_flows = []


class Host:
    # @timing_function
    def __init__(self, node_name):
        self.node_name = node_name
        self.up_link = {}