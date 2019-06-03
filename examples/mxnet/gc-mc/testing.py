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
    n_user = 4
    n_item = 5
    user_item_R = np.array([[1,2,0,4,0], [0,0,3,0,5], [1,0,0,4,0], [0,2,0,0,5]])
    user_item_coo = sp.coo_matrix(user_item_R)
    g = dgl.DGLBipartiteGraph(metagraph = nx.MultiGraph([('user', 'item', 'rating')]),
                              number_of_nodes_by_type = {'user': n_user, 'item': n_item},
                              edge_connections_by_type = {('user', 'item', 'rating'): user_item_coo},
                              node_frame = {"user": np.eye(n_user), "item": np.eye(n_item)},
                              readonly = True)
    print("#users: {}".format(g['user'].number_of_nodes()))
    print("#items: {}".format(g['item'].number_of_nodes()))
    print("#ratings: {}".format(g.number_of_edges()))
    print(g.edges())


gen_bipartite()