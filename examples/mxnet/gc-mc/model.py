import mxnet.ndarray as F
import numpy as np
import warnings
from mxnet.gluon import nn, HybridBlock, Block
from utils import get_activation


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
            self._layers = nn.HybridSequential()
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
    def __init__(self, units, num_links,
                 dropout_rate=0.0, accum='stack', act=None, **kwargs):
        super(MultiLinkGCNAggregator, self).__init__(**kwargs)

        with self.name_scope():
            self.dropout = nn.Dropout(dropout_rate) ### dropout before feeding the out layer
            self._accum = accum
            self._num_links = num_links
            self._act = get_activation(act)
            for i in range(num_links):
                self.__setattr__('weight{}'.format(i),
                                 self.params.get('weight{}'.format(i),
                                                 shape=(units, 0),
                                                 dtype=np.float32,
                                                 allow_deferred_init=True))
                self.__setattr__('bias{}'.format(i),
                                 self.params.get('bias{}'.format(i),
                                                 shape=(units,),
                                                 dtype=np.float32,
                                                 init='zeros',
                                                 allow_deferred_init=True))


    def forward(self, g, dst_key, **kwargs):
        def message_func(edges):
            msg_dic = {}
            for i in range(self._num_links):
                w = kwargs['weight{}'.format(i)]
                msg_dic['msg{}'.format(i)] = w * edges.src['h']* edges.data['support{}'.format(i)]
            return msg_dic

        def reduce_func(nodes):
            out_l = []
            for i in range(self._num_links):
                out_l.append(F.sum(nodes.mailbox['msg{}'.format(i)], 1)+\
                             kwargs['bias{}'.format(i)]
)
            if self._accum == "sum":
                return {'accum': F.add_n(*out_l)}
            elif self._accum == "stack":
                return {'accum': F.concat(*out_l, dim=1)}
            else:
                raise NotImplementedError

        def apply_node_func(nodes):
            return {'h' : self._act(nodes.data['accum'])}

        g.register_message_func(message_func)
        g[dst_key].register_reduce_func(reduce_func)
        g[dst_key].register_apply_node_func(apply_node_func)
        g.send_and_recv(g.edges('uv', 'srcdst'))

        h = g[dst_key].ndata.pop('h')
        return h


class GCMCLayer(Block):
    def __init__(self, agg_units, out_units, num_links, src_key, dst_key,
                 dropout_rate=0.0, agg_accum='stack', agg_act=None,
                 agg_ordinal_sharing=False, share_agg_weights=False,
                 share_out_fc_weights = False, out_act = None,
                 **kwargs):
        super(GCMCLayer, self).__init__(**kwargs)
        self._out_act = get_activation(out_act)
        with self.name_scope():
            self.dropout = nn.Dropout(dropout_rate) ### dropout before feeding the out layer
            self._aggregators = LayerDictionary(prefix='agg_')
            with self._aggregators.name_scope():
                self._aggregators[(src_key, dst_key)] = MultiLinkGCNAggregator(units = agg_units,
                                                                               num_links=num_links,
                                                                               dropout_rate=dropout_rate,
                                                                               accum=agg_accum,
                                                                               act=agg_act,
                                                                               prefix='{}_{}_'.format(src_key, dst_key))
                self._aggregators[(dst_key, src_key)] = MultiLinkGCNAggregator(units=agg_units,
                                                                               num_links=num_links,
                                                                               dropout_rate=dropout_rate,
                                                                               accum=agg_accum,
                                                                               act=agg_act,
                                                                               prefix='{}_{}_'.format(dst_key, src_key))
            self._out_fcs = LayerDictionary(prefix='out_fc_')
            with self._out_fcs.name_scope():
                self._out_fcs[src_key] = nn.Dense(out_units, flatten=False,
                                                  prefix='{}_'.format(src_key))
                self._out_fcs[dst_key] = nn.Dense(out_units, flatten=False,
                                                  prefix='{}_'.format(dst_key))

            self._out_act = get_activation(out_act)

    def forward(self, src_g, dst_g, src_key, dst_key):
        dst_h = self._aggregators[(src_key, dst_key)](src_g, dst_key)
        src_h = self._aggregators[(dst_key, src_key)](dst_g, src_key)
        out_dst = self._out_act(self._out_fcs[dst_key](dst_h))
        out_src = self._out_act(self._out_fcs[dst_key](src_h))
        src_g[src_key].ndata['h'] = out_src
        src_g[dst_key].ndata['h'] = out_dst
        dst_g[src_key].ndata['h'] = out_dst
        dst_g[dst_key].ndata['h'] = out_src
        return out_src, out_dst


