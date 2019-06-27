import time
import math
import numpy as np
import scipy.sparse as sp
import networkx as nx
import dgl
import sys
import mxnet as mx
import dgl.function as fn


"""
   0  1  2  3  4
   -- -- -- -- --
0 |1 |2 |  |4 |  |
1 |  |  |3 |  |5 |
2 |1 |  |  |4 |  |
3 |  |2 |  |  |5 |
"""
# user_item_pair = np.array([[0, 0, 0, 1, 1, 2, 2, 3, 3],
#                            [0, 1, 3, 2, 4, 0, 3, 1, 4]])


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
    ctx = mx.cpu()

    user_item_R = np.array([[1,2,0,4,0], [0,0,3,0,5], [1,0,0,4,0], [0,2,0,0,5]])
    user_item_pair = np.array([[0, 0, 0, 1, 1, 2, 2, 3, 3],
                               [0, 1, 3, 2, 4, 0, 3, 1, 4]])
    user_item_ratings = np.array([1,2,4,3,5,1,4,2,5])
    g = dgl.DGLBipartiteGraph(metagraph = nx.MultiGraph([('user', 'item', 'rating'),
                                                         ('item', 'user', 'rating')]),
                              number_of_nodes_by_type = {'user': n_user, 'item': n_item},
                              edge_connections_by_type = {('user', 'item', 'rating'): (user_item_pair[0, :],
                                                                                       user_item_pair[1, :]),
                                                          ('item', 'user', 'rating'): (user_item_pair[1, :],
                                                                                       user_item_pair[0, :])},
                              readonly = True)
    g['user', 'item', 'rating'].edata["R"] = mx.nd.array(user_item_ratings, ctx=ctx, dtype=np.float32)
    g['item', 'user', 'rating'].edata["R"] = mx.nd.array(user_item_ratings, ctx=ctx, dtype=np.float32)
    print("#users: {}".format(g['user'].number_of_nodes()))
    print("#items: {}".format(g['item'].number_of_nodes()))
    # print("#\t(user-->item) ratings: {}".format(g['user', 'item', 'rating'].number_of_edges()))
    # print("#\t(item-->user) ratings: {}".format(g['item', 'user', 'rating'].number_of_edges()))
    g['user'].ndata['fea'] = mx.nd.ones((g['user'].number_of_nodes(), g['user'].number_of_nodes()), ctx=ctx)*2
    g['item'].ndata['fea'] = mx.nd.ones((g['item'].number_of_nodes(), g['item'].number_of_nodes()), ctx=ctx)*10

    def msg_func(edges):
        return {'m': edges.src['fea']}
    def apply_node_func(nodes):
        return {'res': nodes.data['accum']}

    g1 = g['user', 'item', 'rating']
    g2 = g['item', 'user', 'rating']
    g2.ndata.update({{'res' : ft, 'a1' : a1, 'a2' : a2}})


    print("For g1 ......")
    g1.send_and_recv(g1.edges(),
                     msg_func, fn.sum("m", "accum"), apply_node_func)
    print('g1["item"]', g1["item"].ndata.pop('res'))
    # print('g1["user"]', g1["user"].ndata.pop('res'))
    print("For g2 ......")
    g2.send_and_recv(g2.edges(),
                     msg_func, fn.sum("m", "accum"), apply_node_func)
    #g2.update_all(msg_func, fn.sum("m", "accum"), apply_node_func)
    print('g2["user"]', g2["user"].ndata.pop('res'))
    #print('g2["user"]', g2["user"].ndata.pop('res'))

    #print('g["item"]', g["item"].ndata.pop('res'))

    print("=========================\n\n\n")

    """
       0  1  2  3  4
       -- -- -- -- --
    0 |  |2 |  |  |  |
    1 |  |  |3 |  |  |
    2 |1 |  |  |4 |  |
    3 |  |  |  |  |5 |
    """
    # user_item_pair = np.array([[0, 1, 2, 2, 3],
    #                            [1, 2, 0, 3, 4]])

    user_item_pair = np.array([[0, 1, 2, 2, 3],
                               [1, 2, 0, 3, 4]])
    sub_g = g.edge_subgraph({('user', 'item', 'rating'): g1.edge_ids(user_item_pair[0, :],
                                                                     user_item_pair[1, :]),
                             ('item', 'user', 'rating'): g2.edge_ids(user_item_pair[1, :],
                                                                     user_item_pair[0, :])})
    sub_g.copy_from_parent()
    user_item_sub_g = sub_g['user', 'item', 'rating']
    print(user_item_sub_g.edges())
    print(user_item_sub_g.edata['R'])
    # print(sub_g['item', 'user', 'rating'].edges("all", "srcdst"))
    #print("sub_g['user'].ndata['fea']", sub_g['user'].ndata['fea'])
    print("user_item_sub_g.parent_nid('item')", sub_g.parent_nid('item'))
    print("user_item_sub_g.parent_nid('user')", sub_g.parent_nid('user'))


    return g




g = gen_bipartite()
