import time
import math
import numpy as np
import scipy.sparse as sp
import networkx as nx
import dgl
import sys
import mxnet as mx
import dgl.function as fn
import mx.nd as F
from numpy.testing import assert_array_equal

def edge_pair_input(sort=False):
    if sort:
        src = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 7, 7, 7, 9, 9]
        dst = [4, 6, 9, 3, 5, 3, 7, 5, 8, 1, 3, 4, 9, 1, 9, 6, 2, 8, 9, 2, 10]
        return src, dst
    else:
        src = [0, 0, 4, 5, 0, 4, 7, 4, 4, 3, 2, 7, 7, 5, 3, 2, 1, 9, 6, 1, 9]
        dst = [9, 6, 3, 9, 4, 4, 9, 9, 1, 8, 3, 2, 8, 1, 5, 7, 3, 2, 6, 5, 10]
        return src, dst

def gen_from_edgelist(directed):
    src, dst = edge_pair_input()
    num_typed_nodes = {'src': max(src) + 1, 'dst': max(dst) + 1}
    src = np.array(src, np.int64)
    dst = np.array(dst, np.int64)
    metagraph = nx.MultiGraph([('src', 'dst', 'e'), ('dst', 'src', 'e')])
    if directed:
        g = dgl.DGLBipartiteGraph(metagraph, num_typed_nodes,
                                  {('src', 'dst', 'e'): (src, dst)},
                                  readonly=True)
    else:
        g = dgl.DGLBipartiteGraph(metagraph, num_typed_nodes,
                                  {('src', 'dst', 'e'): (src, dst),
                                   ('dst', 'src', 'e'): (dst, src)},
                                  readonly=True)
    return g

g = gen_from_edgelist(False)
src_g = g['src']
dst_g = g['dst']
print("src_g", src_g)
print("dst_g", dst_g)
print("src_g.number_of_nodes()", src_g.number_of_nodes())
print("dst_g.number_of_nodes()", dst_g.number_of_nodes())

src_g.ndata['nid'] = F.arange(0, src_g.number_of_nodes())
dst_g.ndata['nid'] = F.arange(src_g.number_of_nodes(),
                              src_g.number_of_nodes() + dst_g.number_of_nodes())
srcdst_g = g['src', 'dst', 'e']
dstsrc_g = g['dst', 'src', 'e']

print("srcdst_g", srcdst_g)
print("dstsrc_g", dstsrc_g)
srcdst_g.edata['eid'] = F.arange(0, srcdst_g.number_of_edges())
dstsrc_g.edata['eid'] = F.arange(srcdst_g.number_of_edges(),
                                 srcdst_g.number_of_edges() * 2)
subg_eid = dgl.utils.toindex([0, 2, 7, 9])
subg_src, subg_dst = srcdst_g.find_edges(subg_eid)
print("(subg_dst, subg_src)", (subg_dst, subg_src))
print("dstsrc_g.edge_ids(subg_dst, subg_src)", dstsrc_g.edge_ids(subg_dst, subg_src))
subg = g.edge_subgraph({('src', 'dst', 'e'): subg_eid,
                        ('dst', 'src', 'e'): dstsrc_g.edge_ids(subg_dst, subg_src)})
subg.copy_from_parent()
assert_array_equal(np.unique(F.asnumpy(subg['src'].ndata['nid'])),
                   np.unique(F.asnumpy(subg_src)))
assert_array_equal(np.unique(F.asnumpy(subg['dst'].ndata['nid'])),
                   np.unique(F.asnumpy(subg_dst)) + src_g.number_of_nodes())
assert_array_equal(F.asnumpy(subg['src', 'dst', 'e'].edata['eid']),
                       subg_eid.tonumpy())
assert_array_equal(F.asnumpy(subg['dst', 'src', 'e'].edata['eid']),
                       subg_eid.tonumpy() + srcdst_g.number_of_edges())