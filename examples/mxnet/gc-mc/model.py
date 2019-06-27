import mxnet.ndarray as F
import numpy as np
import warnings
from mxnet.gluon import nn, HybridBlock, Block
from utils import get_activation
import mxnet as mx
import dgl.function as fn

class MultiLinkGCNAggregator(Block):
    def __init__(self, src_key, dst_key, units, src_in_units, dst_in_units, num_links,
                 dropout_rate=0.0, accum='stack', act=None, **kwargs):
        super(MultiLinkGCNAggregator, self).__init__(**kwargs)
        self._src_key = src_key
        self._dst_key = dst_key
        self._accum = accum
        self._num_links = num_links
        self._units = units
        if accum == "stack":
            assert units % num_links == 0, 'units should be divisible by the num_links '
            self._units = self._units // num_links

        with self.name_scope():
            self.dropout = nn.Dropout(dropout_rate) ### dropout before feeding the out layer
            self.act = get_activation(act)
            ### TODO kwargs only be supported in hybridBlock
            # for i in range(num_links):
                # self.__setattr__('weight{}'.format(i),
                #                  self.params.get('weight{}'.format(i),
                #                                  shape=(units, 0),
                #                                  dtype=np.float32,
                #                                  allow_deferred_init=True))
                # self.__setattr__('bias{}'.format(i),
                #                  self.params.get('bias{}'.format(i),
                #                                  shape=(units,),
                #                                  dtype=np.float32,
                #                                  init='zeros',
                #                                  allow_deferred_init=True))
            self.src_dst_weights = self.params.get('src_dst_weight',
                                           shape=(num_links, self._units, src_in_units),
                                           dtype=np.float32,
                                           allow_deferred_init=True)
            self.dst_src_weights = self.params.get('dst_dst_weight',
                                                   shape=(num_links, self._units, dst_in_units),
                                                   dtype=np.float32,
                                                   allow_deferred_init=True)
            # self.biases = self.params.get('bias',
            #                               shape=(num_links, units, ),
            #                               dtype=np.float32,
            #                               init='zeros',
            #                               allow_deferred_init=True)

    def forward(self, g):

        def src_node_update(nodes):
            Ndata = {}
            for i in range(self._num_links):
                # w = kwargs['weight{}'.format(i)]
                w = self.src_dst_weights.data()[i] ## 500 * #nodes
                Ndata = {'w{}'.format(i): mx.nd.dot(self.dropout(nodes.data['fea']), w, transpose_b=True)}
            return Ndata
        def dst_node_update(nodes):
            Ndata = {}
            for i in range(self._num_links):
                w = self.dst_src_weights.data()[i] ## 500 * #nodes
                Ndata = {'w{}'.format(i): mx.nd.dot(self.dropout(nodes.data['fea']), w, transpose_b=True)}
            return Ndata

        def src_dst_msg_func(edges):
            msgs = []
            for i in range(self._num_links): ## 5
                print("edges.src['fea']", edges.src['fea'])
                msgs.append(mx.nd.reshape(edges.data['support{}'.format(i)], shape=(-1, 1)) \
                            * edges.src['w{}'.format(i)]) ## #edge * (100 * 5)
            if self._accum == "sum":
                mess_func = {'msg': mx.nd.add_n(*msgs)}
            elif self._accum == "stack":
                mess_func = {'msg': mx.nd.concat(*msgs, dim=1)}
            else:
                raise NotImplementedError
            return mess_func
        # def dst_src_msg_func(edges):
        #     msgs = []
        #     for i in range(self._num_links):
        #         w = self.dst_src_weights.data()[i]
        #         msgs.append(mx.nd.reshape(edges.data['support{}'.format(i)], shape=(-1, 1)) \
        #                     * mx.nd.dot(self.dropout(edges.src['fea']), w, transpose_b=True))
        #     if self._accum == "sum":
        #         mess_func = {'msg': mx.nd.add_n(*msgs)}
        #     elif self._accum == "stack":
        #         mess_func = {'msg': mx.nd.concat(*msgs, dim=1)}
        #     else:
        #         raise NotImplementedError
        #     return mess_func

        def apply_node_func(nodes):
            return {'h': self.act(nodes.data['accum'])}

        src_dst_g = g[self._src_key, self._dst_key, 'rating']
        dst_src_g = g[self._dst_key, self._src_key, 'rating']
        g[self._src_key].apply_nodes(src_node_update)
        print("g[self._src_key].ndata['w0']", g[self._src_key].ndata['w0'])
        src_dst_g.send_and_recv(src_dst_g.edges(),
                                src_dst_msg_func, fn.sum('msg', 'accum'),
                                apply_node_func)

        dst_src_g[self._dst_key].apply_nodes(dst_node_update)
        dst_src_g.send_and_recv(dst_src_g.edges(),
                                src_dst_msg_func, fn.sum('msg', 'accum'),
                                apply_node_func)

        dst_h = src_dst_g[self._dst_key].ndata.pop('h')
        src_h = dst_src_g[self._src_key].ndata.pop('h')

        return src_h, dst_h

class GCMCLayer(Block):
    def __init__(self, src_key, dst_key, src_in_units, dst_in_units, agg_units, out_units, num_links,
                 dropout_rate=0.0, agg_accum='stack', agg_act=None, out_act=None,
                 # agg_ordinal_sharing=False, share_agg_weights=False, share_out_fc_weights=False,
                 **kwargs):
        super(GCMCLayer, self).__init__(**kwargs)
        self._out_act = get_activation(out_act)
        self._src_key = src_key
        self._dst_key = dst_key
        with self.name_scope():
            self.dropout = nn.Dropout(dropout_rate)
            self.aggregator = MultiLinkGCNAggregator(src_key=src_key,
                                                     dst_key=dst_key,
                                                     units = agg_units,
                                                     src_in_units=src_in_units,
                                                     dst_in_units=dst_in_units,
                                                     num_links=num_links,
                                                     dropout_rate=dropout_rate,
                                                     accum=agg_accum,
                                                     act=agg_act,
                                                     prefix='agg_')
            self.user_out_fcs = nn.Dense(out_units, flatten=False, prefix='user_out_')
            self.item_out_fcs = nn.Dense(out_units, flatten=False, prefix='item_out_')
            self._out_act = get_activation(out_act)

    def forward(self, graph):
        user_h, item_h = self.aggregator(graph)
        out_user = self._out_act(self.user_out_fcs(user_h))
        out_item = self._out_act(self.item_out_fcs(item_h))
        return out_user, out_item


class BiDecoder(HybridBlock):
    def __init__(self, in_units, out_units, num_basis_functions=2, prefix=None, params=None):
        super(BiDecoder, self).__init__(prefix=prefix, params=params)
        self._num_basis_functions = num_basis_functions
        with self.name_scope():
            for i in range(num_basis_functions):
                self.__setattr__('weight{}'.format(i),
                                 self.params.get('weight{}'.format(i), shape=(in_units, in_units),
                                                 init=mx.initializer.Orthogonal(scale=1.1,
                                                                                rand_type='normal'),
                                                 allow_deferred_init=True))
            self.rate_out = nn.Dense(units=out_units, flatten=False, use_bias=False, prefix="rate_")

    def hybrid_forward(self, F, data1, data2, **kwargs):
        basis_outputs_l = []
        for i in range(self._num_basis_functions):
            basis_out = F.sum(F.dot(data1, kwargs["weight{}".format(i)]) * data2,
                              axis=1, keepdims=True)
            basis_outputs_l.append(basis_out)
        basis_outputs = F.concat(*basis_outputs_l, dim=1)
        out = self.rate_out(basis_outputs)
        return out


class InnerProductLayer(HybridBlock):
    def __init__(self, mid_units=None, **kwargs):
        super(InnerProductLayer, self).__init__(**kwargs)
        self._mid_units = mid_units
        if self._mid_units is not None:
            self._mid_map = nn.Dense(mid_units, flatten=False)

    def hybrid_forward(self, F, data1, data2):
        if self._mid_units is not None:
            data1 = self._mid_map(data1)
            data2 = self._mid_map(data2)
        score = F.sum(data1 * data2, axis=1, keepdims=True)
        return score

