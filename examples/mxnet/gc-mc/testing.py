import time
import math
import numpy as np
import scipy.sparse as sp
import networkx as nx
import dgl


"""
   0  1  2  3  4
   -- -- -- -- --
0 |1 |2 |  |4 |  |
1 |  |  |3 |  |5 |
2 |1 |  |  |4 |  |
3 |  |2 |  |  |5 |
"""
def gen_bipartite():
    user_item_R = np.array([[1,2,0,4,0], [0,0,3,0,5], [1,0,0,4,0], [0,2,0,0,5]])
    user_item_coo = sp.coo_matrix(user_item_R)
    g = dgl.DGLBipartiteGraph(metagraph = nx.MultiGraph([('user', 'item', 'rating')]),
                              num_typed_nodes = {'user': 4, 'item': 5},
                              edge_connections_by_type = {('user', 'item', 'rating'): user_item_coo},
                              readonly = True)
    print("#users: {}".format(g['user'].number_of_nodes()))
    print("#items: {}".format(g['item'].number_of_nodes()))
    print("#ratings: {}".format(g.number_of_nodes()))
    print(g.edges())



gen_bipartite()