class Link:
    # @timing_function
    def __init__(self, src, sink, bandwidth):
        self.src = src 
        self.sink = sink 
        self.bandwidth = bandwidth
        self.flows = []

class OpticalLink:
    # @timing_function
    def __init__(self, link_id, src, sink, bandwidth):
        self.link_id = link_id
        self.src = src 
        self.sink = sink 
        self.bandwidth = bandwidth
        self.flows = []