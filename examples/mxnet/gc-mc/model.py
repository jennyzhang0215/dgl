import mxnet.ndarray as F
import numpy as np
import warnings
from mxnet.gluon import nn, HybridBlock, Block
from utils import get_activation
import mxnet as mx
import dgl.function as fn

class LayerDictionary(Block):
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        input_dims : dict
        output_dims : dict
        """
        super(LayerDictionary, self).__init__(**kwargs)
        self._key2idx = dict()
        with self.name_scope():
            self._layers = nn.Sequential()
        self._nlayers = 0

    def __len__(self):
        return len(self._layers)

    def __setitem__(self, key, layer):
        if key in self._key2idx:
            warnings.warn('Duplicate Key. Need to test the code!')
            self._layers[self._key2idx[key]] = layer
        else:
            self._layers.add(layer)
            self._key2idx[key] = self._nlayers
            self._nlayers += 1

    def __getitem__(self, key):
        return self._layers[self._key2idx[key]]

    def __contains__(self, key):
        return key in self._key2idx


class MultiLinkGCNAggregator(Block):
    def __init__(self, src_key, dst_key, units, in_units, num_links,
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
            self.weights = self.params.get('weight',
                                           shape=(num_links, self._units, in_units),
                                           dtype=np.float32,
                                           allow_deferred_init=True)
            # self.biases = self.params.get('bias',
            #                               shape=(num_links, units, ),
            #                               dtype=np.float32,
            #                               init='zeros',
            #                               allow_deferred_init=True)

    def forward(self, g, src_input, dst_input):
        src_input = self.dropout(src_input)
        dst_input = self.dropout(dst_input)
        #print("self._src_key", self._src_key)
        #print("self._dst_key", self._dst_key)
        g[self._src_key].ndata['fea'] = src_input
        g[self._dst_key].ndata['fea'] = dst_input

        def message_func(edges):
            #print("\n\n In the message function ...")
            msgs = []
            for i in range(self._num_links):
                # w = kwargs['weight{}'.format(i)]
                w = self.weights.data()[i]
                msgs.append(mx.nd.reshape(edges.data['support{}'.format(i)], shape=(-1, 1)) \
                               * mx.nd.dot(edges.src['fea'], w, transpose_b=True))
            if self._accum == "sum":
                mess_func = {'msg': mx.nd.add_n(*msgs)}

            elif self._accum == "stack":
                mess_func = {'msg': mx.nd.concat(*msgs, dim=1)}
            else:
                raise NotImplementedError
            assert isinstance(mess_func, dict)

            return mess_func

        def apply_node_func(nodes):
            return {'h': self.act(nodes.data['accum'])}

        g.send_and_recv(g.edges('uv', 'srcdst'),
                        message_func,
                        fn.sum('msg', 'accum'),
                        apply_node_func)
        # g.register_message_func(message_func)
        # g[self._dst_key].register_reduce_func(fn.sum('msg', 'accum'))
        # g[self._dst_key].register_apply_node_func(apply_node_func)
        # g.send_and_recv()
        h = g[self._dst_key].ndata.pop('h')
        return h

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
            self.user_aggregator = MultiLinkGCNAggregator(src_key=src_key,
                                                                    dst_key=dst_key,
                                                                    units = agg_units,
                                                                    in_units=src_in_units,
                                                                    num_links=num_links,
                                                                    dropout_rate=dropout_rate,
                                                                    accum=agg_accum,
                                                                    act=agg_act,
                                                                    prefix='user_agg_')
            self.item_aggregator = MultiLinkGCNAggregator(src_key=dst_key,
                                                                    dst_key=src_key,
                                                                    in_units=dst_in_units,
                                                                    units=agg_units,
                                                                    num_links=num_links,
                                                                    dropout_rate=dropout_rate,
                                                                    accum=agg_accum,
                                                                    act=agg_act,
                                                                    prefix='item_agg_')
            self.user_out_fcs = nn.Dense(out_units, flatten=False, prefix='user_out_')
            self.item_out_fcs = nn.Dense(out_units, flatten=False, prefix='item_out_')
            self._out_act = get_activation(out_act)

    def forward(self, uv_graph, vu_graph, user_fea, movie_fea):
        movie_h = self.user_aggregator(uv_graph, user_fea, movie_fea)
        user_h = self.item_aggregator(vu_graph, movie_fea, user_fea)
        out_user = self._out_act(self.user_out_fcs(user_h))
        out_movie = self._out_act(self.item_out_fcs(movie_h))
        return out_user, out_movie


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

