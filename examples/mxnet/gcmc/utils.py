import math
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
import mxnet.gluon as gluon
from mxnet.gluon import nn, HybridBlock, Block
import ast
import os
import inspect
import logging
import re
import mxnet.ndarray as nd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier
import csv
import os
from collections import OrderedDict

class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()

class IdentityActivation(HybridBlock):
    def hybrid_forward(self, F, x):
        return x


class ELU(HybridBlock):
    r"""
    Exponential Linear Unit (ELU)
        "Fast and Accurate Deep Network Learning by Exponential Linear Units", Clevert et al, 2016
        https://arxiv.org/abs/1511.07289
        Published as a conference paper at ICLR 2016
    Parameters
    ----------
    alpha : float
        The alpha parameter as described by Clevert et al, 2016
    Inputs:
        - **data**: input tensor with arbitrary shape.
    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, alpha=1.0, **kwargs):
        super(ELU, self).__init__(**kwargs)
        self._alpha = alpha

    def hybrid_forward(self, F, x):
        return - self._alpha * F.relu(1.0 - F.exp(x)) + F.relu(x)


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or HybridBlock

    Returns
    -------
    ret: HybridBlock
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'identity':
            return IdentityActivation()
        elif act == 'elu':
            return ELU()
        elif act in ['relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']:
            return nn.Activation(act)
        else:
            raise NotImplementedError
    else:
        return act


class StackFCBlock(HybridBlock):
    def __init__(self, units, num_layers, dropout_rate=0.0,
                 mid_act='relu', out_act=None, flatten=False, prefix=None, params=None):
        super(StackFCBlock, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_layers = num_layers
        self._dropout_rate = dropout_rate
        self._mid_act = get_activation(mid_act)
        self._out_act = get_activation(out_act)
        self._dropout = nn.Dropout(dropout_rate)
        with self.name_scope():
            self.layers = nn.HybridSequential('stack_fc_')
            with self.layers.name_scope():
                for i in range(num_layers):
                    self.layers.add(nn.Dense(units, flatten=flatten, prefix='l{}_'.format(i)))

    def hybrid_forward(self, F, x):
        for i in range(self._num_layers):
            x = self.layers[i](x)
            if i < self._num_layers - 1:
                x = self._mid_act(x)
                x = self._dropout(x)
            else:
                x = self._out_act(x)
        return x


class DenseNetBlock(HybridBlock):
    def __init__(self, units, layer_num, act, flatten=False, prefix=None, params=None):
        super(DenseNetBlock, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._layer_num = layer_num
        self._act = get_activation(act)
        print("layer_num", layer_num)
        with self.name_scope():
            self.layers = nn.HybridSequential('dblock_')
            with self.layers.name_scope():
                for _ in range(layer_num):
                    self.layers.add(nn.Dense(units, flatten=flatten))

    def hybrid_forward(self, F, x):
        layer_in_l = [x]
        layer_out = None
        for i in range(self._layer_num):
            if len(layer_in_l) == 1:
                layer_in = layer_in_l[0]
            else:
                layer_in = F.concat(*layer_in_l, dim=-1)
            layer_out = self._act(self.layers[i](layer_in))
            layer_in_l.append(layer_out)
        return layer_out


class HeterDenseNetBlock(Block):
    def __init__(self, units, layer_num, act, num_set, flatten=False, prefix=None, params=None):
        super(HeterDenseNetBlock, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_set = num_set
        self._layer_num = layer_num
        self._act = get_activation(act)
        with self.name_scope():
            self.layers = nn.Sequential('hdblock_')
            with self.layers.name_scope():
                for _ in range(layer_num):
                    self.layers.add(nn.Dense(units*num_set, flatten=flatten))

    def forward(self, x, mask):
        """

        Parameters
        ----------
        F
        x: Shape(batch_size, num_node, input_dim)
        mask: Shape(batch_size, num_node, num_set, 1)

        Returns
        -------

        """
        layer_in_l = [x]
        layer_out = None
        for i in range(self._layer_num):
            if len(layer_in_l) == 1:
                layer_in = layer_in_l[0]
            else:
                layer_in = nd.concat(*layer_in_l, dim=-1)
            ### TODO assume batch_size=1
            x_mW = nd.reshape(self.layers[i](layer_in), shape=(0, 0, self._num_set, self._units))
            layer_out = self._act( nd.sum(nd.broadcast_mul(x_mW, mask), axis=-2) )
            layer_in_l.append(layer_out)
        return layer_out

class L2Normalization(HybridBlock):
    def __init__(self, axis=-1, eps=1E-6, prefix=None, params=None):
        super(L2Normalization, self).__init__(prefix=prefix, params=params)
        self._axis = axis
        self._eps = eps

    def hybrid_forward(self, F, x):
        ret = F.broadcast_div(x, F.sqrt(F.sum(F.square(x), axis=self._axis, keepdims=True)
                                        + self._eps))
        return ret



def copy_params_to_nd(params, ctx=None):
    return {k: v.data(ctx).copy() for k, v in params.items()}


def copy_nd_to_params(nd_params, params):
    for k, v in params.items():
        v.set_data(nd_params[k])


def safe_eval(expr):
    if type(expr) is str:
        return ast.literal_eval(expr)
    else:
        return expr


def get_name_id(dir_path):
    name_id = 0
    file_path = os.path.join(dir_path, 'cfg%d.yml' % name_id)
    while os.path.exists(file_path):
        name_id += 1
        file_path = os.path.join(dir_path, 'cfg%d.yml' % name_id)
    return name_id


def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):
    """

    Parameters
    ----------
    folder : str or None
    name : str or None
    level : int
    console_level
    no_console: bool
        Whether to disable the console log

    Returns
    -------

    """
    if name is None:
        name = inspect.stack()[1][1].split('.')[0]
    if folder is None:
        folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Remove all the current handlers
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".log")
    print("All Logs will be saved to %s" %logpath)
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)
    if not no_console:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder


def parse_ctx(ctx_args):
    ctx = re.findall('([a-z]+)(\d*)', ctx_args)
    ctx = [(device, int(num)) if len(num) > 0 else (device, 0) for device, num in ctx]
    ctx = [mx.Context(*ele) for ele in ctx]
    return ctx


def gluon_total_param_num(net):
    return sum([np.prod(v.shape) for v in net.collect_params().values()])


def gluon_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(gluon_total_param_num(net)) +\
               'Params:\n'
    for k, v in net.collect_params().items():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str


def params_clip_global_norm(param_dict, clip, ctx):
    grads = [p.grad(ctx) for p in param_dict.values()]
    gnorm = gluon.utils.clip_global_norm(grads, clip)
    return gnorm


def get_global_norm(arrays):
    ctx = arrays[0].context
    total_norm = nd.add_n(*[nd.dot(x, x).as_in_context(ctx)
                            for x in (arr.reshape((-1,)) for arr in arrays)])
    total_norm = nd.sqrt(total_norm).asscalar()
    return total_norm


def div_up(a, b):
    return (a + b - 1) // b


def copy_to_ctx(data, ctx, dtype=None):
    if isinstance(data, (list, tuple)):
        if dtype is None:
            dtype = data[0].dtype
        return [nd.array(ele, dtype=dtype, ctx=ctx) for ele in data]
    elif isinstance(data, dict):
        if dtype is None:
            return {k: copy_to_ctx(v, ctx) for k, v in data.items()}
        else:
            return {k: copy_to_ctx(v, ctx, dtype) for k, v in data.items()}
    else:
        if dtype is None:
            dtype = data.dtype
        return nd.array(data, dtype=dtype, ctx=ctx)


def nd_acc(pred, label):
    """Evaluate accuracy using mx.nd.NDArray

    Parameters
    ----------
    pred : nd.NDArray
    label : nd.NDArray
    class_num : int

    Returns
    -------
    acc : float
    """
    return nd.sum(pred == label).asscalar() / float(pred.size)


def nd_f1(pred, label, num_class, average="micro"):
    """Evaluate F1 using mx.nd.NDArray

    Parameters
    ----------
    pred : nd.NDArray
        Shape (num, label_num) or (num,)
    label : nd.NDArray
        Shape (num, label_num) or (num,)
    num_class : int
    average : str

    Returns
    -------
    f1 : float
    """
    if pred.dtype != np.float32:
        pred = pred.astype(np.float32)
        label = label.astype(np.float32)
    assert num_class > 1
    assert pred.ndim == label.ndim
    if num_class == 2 and average == "micro":
        tp = nd.sum((pred == 1) * (label == 1)).asscalar()
        fp = nd.sum((pred == 1) * (label == 0)).asscalar()
        fn = nd.sum((pred == 0) * (label == 1)).asscalar()
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        assert num_class is not None
        pred_onehot = nd.one_hot(indices=pred, depth=num_class)
        label_onehot = nd.one_hot(indices=label, depth=num_class)
        tp = pred_onehot * label_onehot
        fp = pred_onehot * (1 - label_onehot)
        fn = (1 - pred_onehot) * label_onehot
        if average == "micro":
            tp = nd.sum(tp).asscalar()
            fp = nd.sum(fp).asscalar()
            fn = nd.sum(fn).asscalar()
            precision = float(tp) / (tp + fp)
            recall = float(tp) / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
        elif average == "macro":
            if tp.ndim == 3:
                tp = nd.sum(tp, axis=(0, 1))
                fp = nd.sum(fp, axis=(0, 1))
                fn = nd.sum(fn, axis=(0, 1))
            else:
                tp = nd.sum(tp, axis=0)
                fp = nd.sum(fp, axis=0)
                fn = nd.sum(fn, axis=0)
            precision = nd.mean(tp / (tp + fp)).asscalar()
            recall = nd.mean(tp / (tp + fn)).asscalar()
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            raise NotImplementedError
    return f1


def sklearn_logistic_regression(dataname,
                                train_embeds, train_labels,
                                valid_embeds, valid_labels,
                                test_embeds, test_labels,
                                max_iter=None, tol=0.001, alpha=0.0001):
    if not isinstance(train_embeds, np.ndarray):
        train_embeds = train_embeds.asnumpy()
    if not isinstance(valid_embeds, np.ndarray):
        valid_embeds = valid_embeds.asnumpy()
    if not isinstance(test_embeds, np.ndarray):
        test_embeds = test_embeds.asnumpy()
    if dataname == "ppi":
        classifier = MultiOutputClassifier(
            SGDClassifier(loss="log", alpha=alpha, n_jobs=-1, max_iter=max_iter, tol=tol))
        classifier.fit(train_embeds, train_labels)
    elif dataname == "cora" or dataname == "reddit":
        classifier = SGDClassifier(loss="log", alpha=alpha, n_jobs=-1, max_iter=max_iter, tol=tol)
        classifier.fit(train_embeds, train_labels)
    else:
        raise NotImplementedError
    train_pred = classifier.predict(train_embeds)
    valid_pred = classifier.predict(valid_embeds)
    test_pred = classifier.predict(test_embeds)

    train_acc = accuracy_score(y_true=train_labels.reshape((-1,)), y_pred=train_pred.reshape((-1,)))
    valid_acc = accuracy_score(y_true=valid_labels.reshape((-1,)), y_pred=valid_pred.reshape((-1,)))
    test_acc = accuracy_score(y_true=test_labels.reshape((-1,)), y_pred=test_pred.reshape((-1,)))

    train_f1 = f1_score(y_true=train_labels, y_pred=train_pred, average='micro')
    valid_f1 = f1_score(y_true=valid_labels, y_pred=valid_pred, average='micro')
    test_f1 = f1_score(y_true=test_labels, y_pred=test_pred, average='micro')

    return train_acc, train_f1, valid_acc, valid_f1, test_acc, test_f1
