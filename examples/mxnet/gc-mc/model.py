from mxnet as gluon
import mxnet.ndarray as F
import dgl.function as fn
from gluon import nn
import numpy as np

from utils import get_activation

class MultiLinkGCNAggregator(gluon.Block):
    def __init__(self, agg_units, num_links,
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
                                                 shape=(agg_units, 0),
                                                 dtype=np.float32,
                                                 allow_deferred_init=True))
                self.__setattr__('bias{}'.format(i),
                                 self.params.get('bias{}'.format(i),
                                                 shape=(agg_units,),
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


class GCMCLayer():
    def __init__(self):

