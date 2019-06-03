import time
import math
import numpy as np
import scipy.sparse as sp
import networkx as nx
import dgl
import sys


"""
   0  1  2  3  4
   -- -- -- -- --
0 |1 |2 |  |4 |  |
1 |  |  |3 |  |5 |
2 |1 |  |  |4 |  |
3 |  |2 |  |  |5 |
"""

def _globally_normalize_bipartite_adjacency(adjacencies, symmetric=True):
    """ Globally Normalizes set of bipartite adjacency matrices """

    # degree_u and degree_v are row and column sums of adj+I

    adj_tot = np.sum(adj for adj in adjacencies) ## it is just the original training adj
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

    if symmetric:
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]

    return adj_norm

def compute_support(adj_train, num_link, symmetric):
    support_l = []
    adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32)
    for i in range(num_link):
        # build individual binary rating matrices (supports) for each rating
        support_unnormalized = sp.csr_matrix(adj_train_int == i + 1, dtype=np.float32)
        support_l.append(support_unnormalized)

    support_l = _globally_normalize_bipartite_adjacency(support_l, symmetric=symmetric)

    num_support = len(support_l)
    print("num_support:", num_support)
    for idx, sup in enumerate(support_l):
        print("support{}:\n".format(idx), sup.toarray(), "\n")
    #support = sp.hstack(support, format='csr')
    return support_l

def gen_bipartite():
    n_user = 4
    n_item = 5
    num_link = 5
    sym = True
    user_item_R = np.array([[1,2,0,4,0], [0,0,3,0,5], [1,0,0,4,0], [0,2,0,0,5]])
    user_item_pair = np.array([[0, 0, 0, 1, 1, 2, 2, 3, 3],
                               [0, 1, 3, 2, 4, 0, 3, 1, 4]])
    user_item_ratings = np.array([1,2,4,3,5,1,4,2,5])
    g = dgl.DGLBipartiteGraph(metagraph = nx.MultiGraph([('user', 'item', 'rating')]),
                              number_of_nodes_by_type = {'user': n_user, 'item': n_item},
                              edge_connections_by_type = {('user', 'item', 'rating'): (user_item_pair[0, :],
                                                                                       user_item_pair[1, :])},
                              readonly = True)
    g.edata["rating"] = user_item_ratings

    support_l = compute_support(user_item_R, num_link, sym)
    for idx, support in enumerate(support_l):
        sup_coo = support.tocoo()
        g.edges[sup_coo.row, sup_coo.col].data['support_{}'.format(idx)] = sup_coo.data

    print("#users: {}".format(g['user'].number_of_nodes()))
    print("#items: {}".format(g['item'].number_of_nodes()))
    print("#ratings: {}".format(g.number_of_edges()))
    # print("g.edges('all', 'eid')", g.edges('all', 'eid'))
    # print("g.edges('all', 'srcdst')", g.edges('all', 'srcdst'))
    # print("g.edges('uv', 'eid')", g.edges('uv', 'eid'))
    # print("g.edges('uv', 'srcdst')", g.edges('uv', 'srcdst'))
    print("g.edata", g.edata)


    return g




g = gen_bipartite()
