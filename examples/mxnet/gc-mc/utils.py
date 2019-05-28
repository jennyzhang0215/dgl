import ast
import os
import io
import inspect
import logging
import re
import mxnet.ndarray as nd
from mxnet import gluon
import mxnet as mx
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier


class ExponentialMovingAverage(object):
    def __init__(self, decay=0.999, num_updates=None, dynamic_decay=False, params=None, ctx=None):
        """Maintain a set of shadow variables "v" that is calculated by

        v[:] -= (1 - decay) * (v - param)

        Parameters
        ----------
        decay : float
        num_updates : int or None
            The number of total training updates. It will affect the decay.
        dynamic_decay : bool
            Whether the decay is related to the updating steps
            If dynamic_decay is False:
                decay = min(decay, (1.0 + num_updates) / (10.0 + num_updates))
            Else:
                decay = 1.0 / (step + 1) # Step starts with 0
        params : ParameterDict or None
            The initial parameters. If it's None, it will be set based by self.initialize()
        ctx : None or mx.context.Context
        """
        self._decay = decay
        self._num_updates = num_updates
        self._dynamic_decay = dynamic_decay
        self._ctx = ctx
        self._initialized = False
        self._average_params = None
        self._step = -1
        if params is not None:
            self.initialize(params)

    def initialize(self, params):
        """

        Parameters
        ----------
        params : gluon.parameter.ParameterDict

        """
        self._initialized = True
        self._average_params = copy_params_to_nd(params, self._ctx)

    def step(self, params):
        self._step += 1
        if self._dynamic_decay:
            decay = 1.0 / (self._step + 1.0)
        else:
            decay = min(self._decay, (1.0 + self._num_updates) / (10.0 + self._num_updates))
        for name, average_param in self._average_params.items():
            average_param[:] += decay * (params[name].data(self._ctx) - average_param)

    def copy_to(self, params):
        copy_nd_to_params(self._average_params, params)


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
