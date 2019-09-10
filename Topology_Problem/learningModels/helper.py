import numpy as np
import networkx as nx

def maxWeightMatching(trafficMatrix):
    # Input: Traffic Matrix (N x N) {Numpy}
    # Output: List of Edges to Connect.
    numNodes = np.size(trafficMatrix, 0)

    G = nx.Graph()
    G.add_nodes_from([_ for _ in range(numNodes * 2)])

    nonZeroIndices = np.nonzero(trafficMatrix)
    nonZeroIndices = zip(*nonZeroIndices)

    for idx in nonZeroIndices:
        G.add_edge(idx[0], 
                   idx[1] + numNodes, 
                   weight = trafficMatrix[idx[0], idx[1]])

    temp = nx.max_weight_matching(G)

    res = []

    for edge in temp:
        if edge[0] < numNodes:
            res.append([edge[0], edge[1] - numNodes])
        else:
            res.append([edge[1], edge[0] - numNodes])

    del G
    del temp

    return res