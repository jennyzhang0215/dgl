import os
import argparse
import logging
import random
import string
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

from graph import HeterGraph, merge_node_ids_dict, set_seed
from datasets import LoadData
from layers import HeterGCNLayer, BiDecoder, StackedHeterGCNLayers, LayerDictionary, InnerProductLayer
from iterators import HeterIterator
from utils import get_activation, parse_ctx, \
    gluon_net_info, gluon_total_param_num, params_clip_global_norm, \
    logging_config, MetricLogger


def load_dataset(args):
    dataset = LoadData(args.data_name, seed = args.seed,
                       test_ratio = args.data_test_ratio,
                       val_ratio = args.data_valid_ratio,
                       force_download = False)
    all_graph = dataset.graph
    name_user = dataset.name_user
    name_item = dataset.name_item
    logging.info(dataset)
    # !IMPORTANT. We need to check that ids in all_graph are continuous from 0 to #Node - 1.
    # We will later use these ids to take the embedding vectors
    all_graph.check_continous_node_ids()
    feature_dict = dict()
    nd_user_indices = mx.nd.arange(all_graph.features[name_user].shape[0], ctx=args.ctx)
    nd_item_indices = mx.nd.arange(all_graph.features[name_item].shape[0], ctx=args.ctx)
    user_item_total = all_graph.features[name_user].shape[0] + all_graph.features[name_item].shape[0]
    feature_dict[name_user] = mx.nd.one_hot(nd_user_indices, user_item_total)
    feature_dict[name_item] = mx.nd.one_hot(nd_item_indices + nd_user_indices.shape[0], user_item_total)

    info_line = "Feature dim: "
    info_line += "\n" + name_user + ": {}".format(feature_dict[name_user].shape)
    info_line += "\n" + name_item + ": {}".format(feature_dict[name_item].shape)
    logging.info(info_line)

    return dataset, all_graph, feature_dict


def gen_graph_sampler_args(meta_graph):
    ret = dict()
    for src_key in meta_graph:
        for dst_key in meta_graph[src_key]:
            ret[(src_key, dst_key)] = -1
    return ret


def gen_pair_key(src_key, dst_key):
    if src_key < dst_key:
        return src_key, dst_key
    else:
        return dst_key, src_key



class Net(nn.Block):
    def __init__(self, all_graph, nratings, name_user, name_item, **kwargs):
        super(Net, self).__init__(**kwargs)
        self._nratings = nratings
        self._name_user = name_user
        self._name_item = name_item
        self._act = get_activation(args.model_activation)
        with self.name_scope():
            # Construct Encoder
            self.encoder = StackedHeterGCNLayers(prefix='enc_')
            with self.encoder.name_scope():
                self.encoder.add(HeterGCNLayer(meta_graph=all_graph.meta_graph,
                                               multi_link_structure=all_graph.get_multi_link_structure(),
                                               dropout_rate=args.gcn_dropout,
                                               agg_type='gcn',
                                               agg_units=args.gcn_agg_units,
                                               out_units=args.gcn_out_units,
                                               source_keys=all_graph.meta_graph.keys(),
                                               agg_ordinal_sharing=args.gcn_agg_ordinal_share,
                                               share_agg_weights=args.gcn_agg_share_weights,
                                               agg_accum=args.gcn_agg_accum,
                                               agg_act=args.model_activation,
                                               accum_self=args.gcn_out_accum_self,
                                               out_act=args.model_activation,
                                               layer_accum=args.gcn_out_accum,
                                               layer_norm=False,
                                               prefix='l0_'))


            if args.gen_r_use_classification:
                self.gen_ratings = BiDecoder(in_units=args.gcn_out_units,
                                             out_units=nratings,
                                             num_basis_functions=args.gen_r_num_basis_func,
                                             prefix='gen_rating')
            else:
                self.gen_ratings = InnerProductLayer(prefix='gen_rating')


    def forward(self, graph, feature_dict, rating_node_pairs=None, graph_sampler_args=None, symm=None):
        """

        Parameters
        ----------
        graph : HeterGraph
        feature_dict : dict
            Dictionary contains the base features of all nodes
        rating_node_pairs : np.ndarray or None
            Shape: (2, #Edges), First row is user and the second row is item
        graph_sampler_args : dict or None
            Arguments for graph sampler
        symm : bool
            Whether to calculate the support in the symmetric formula

        Returns
        -------
        pred_ratings : list of mx.nd.ndarray
            The predicted ratings. If we use the stacked hourglass AE structure.
             it will return a list with multiple predicted ratings
        """
        if symm is None:
            symm = args.gcn_agg_norm_symm
        ctx = next(iter(feature_dict.values())).context
        req_node_ids_dict = dict()

        uniq_node_ids_dict, encoder_fwd_indices = \
            merge_node_ids_dict([{self._name_user: rating_node_pairs[0],
                                  self._name_item: rating_node_pairs[1]},
                                 req_node_ids_dict])
        req_node_ids_dict, encoder_fwd_plan = self.encoder.gen_plan(graph=graph,
                                                                    sel_node_ids_dict=uniq_node_ids_dict,
                                                                    graph_sampler_args=graph_sampler_args,
                                                                    symm=symm)

        input_dict = dict()
        for key, req_node_ids in req_node_ids_dict.items():
            input_dict[key] = mx.nd.take(feature_dict[key],
                                         mx.nd.array(req_node_ids, ctx=ctx, dtype=np.int32))

        output_dict = self.encoder.heter_sage(input_dict, encoder_fwd_plan)
        rating_idx_dict, req_idx_dict = encoder_fwd_indices

        # Generate the predicted ratings
        assert rating_node_pairs is not None
        rating_user_fea = mx.nd.take(output_dict[self._name_user],
                                     mx.nd.array(rating_idx_dict[self._name_user], ctx=ctx, dtype=np.int32))
        rating_item_fea = mx.nd.take(output_dict[self._name_item],
                                     mx.nd.array(rating_idx_dict[self._name_item], ctx=ctx, dtype=np.int32))
        pred_ratings = self.gen_ratings(rating_user_fea, rating_item_fea)

        return pred_ratings



def evaluate(args, net, feature_dict, data_iter, segment='valid'):
    rating_mean = data_iter._train_ratings.mean()
    rating_std = data_iter._train_ratings.std()
    rating_sampler = data_iter.rating_sampler(batch_size=args.train_rating_batch_size, segment=segment,
                                              sequential=True)
    possible_rating_values = data_iter.possible_rating_values
    nd_possible_rating_values = mx.nd.array(possible_rating_values, ctx=args.ctx, dtype=np.float32)
    eval_graph = data_iter.val_graph if segment == 'valid' else data_iter.test_graph
    graph_sampler_args = gen_graph_sampler_args(data_iter.all_graph.meta_graph)
    # Evaluate RMSE
    cnt = 0
    rmse = 0

    for rating_node_pairs, gt_ratings in rating_sampler:
        nd_gt_ratings = mx.nd.array(gt_ratings, dtype=np.float32, ctx=args.ctx)
        cnt += rating_node_pairs.shape[1]
        pred_ratings = net.forward(graph=eval_graph,
                                   feature_dict=feature_dict,
                                   rating_node_pairs=rating_node_pairs,
                                   graph_sampler_args=graph_sampler_args,
                                   symm=args.gcn_agg_norm_symm)
        if args.gen_r_use_classification:
            real_pred_ratings = (mx.nd.softmax(pred_ratings, axis=1) *
                                 nd_possible_rating_values.reshape((1, -1))).sum(axis=1)
            rmse += mx.nd.square(real_pred_ratings - nd_gt_ratings).sum().asscalar()
        else:
            rmse += mx.nd.square(mx.nd.clip(pred_ratings.reshape((-1,)) * rating_std + rating_mean,
                                            possible_rating_values.min(),
                                            possible_rating_values.max()) - nd_gt_ratings).sum().asscalar()
    rmse  = np.sqrt(rmse / cnt)
    return rmse

def train(args):
    dataset, all_graph, feature_dict = load_dataset(args)
    valid_node_pairs, _ = dataset.valid_data
    test_node_pairs, _ = dataset.test_data
    data_iter = HeterIterator(all_graph=all_graph,
                              name_user=dataset.name_user,
                              name_item=dataset.name_item,
                              test_node_pairs=test_node_pairs,
                              valid_node_pairs=valid_node_pairs,
                              seed=args.seed)
    logging.info(data_iter)
    ### build the net
    possible_rating_values = data_iter.possible_rating_values
    nd_possible_rating_values = mx.nd.array(possible_rating_values, ctx=args.ctx, dtype=np.float32)
    net = Net(all_graph=all_graph, nratings=possible_rating_values.size,
              name_user=dataset.name_user, name_item=dataset.name_item)
    net.initialize(init=mx.init.Xavier(factor_type='in'), ctx=args.ctx)
    net.hybridize()
    if args.gen_r_use_classification:
        rating_loss_net = gluon.loss.SoftmaxCELoss()
    else:
        rating_loss_net = gluon.loss.L2Loss()
    rating_loss_net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), args.train_optimizer, {'learning_rate': args.train_lr})
    ### prepare the logger
    train_loss_logger = MetricLogger(['iter', 'loss', 'rmse', ], ['%d', '%.4f', '%.4f'],
                                     os.path.join(args.save_dir, 'train_loss%d.csv' % args.save_id))
    valid_loss_logger = MetricLogger(['iter', 'rmse'], ['%d', '%.4f'],
                                     os.path.join(args.save_dir, 'valid_loss%d.csv' % args.save_id))
    test_loss_logger = MetricLogger(['iter', 'rmse'], ['%d', '%.4f'],
                                    os.path.join(args.save_dir, 'test_loss%d.csv' % args.save_id))
    ### initialize the iterator
    rating_sampler = data_iter.rating_sampler(batch_size=args.train_rating_batch_size,
                                              segment='train')
    graph_sampler_args = gen_graph_sampler_args(all_graph.meta_graph)
    rating_mean = data_iter._train_ratings.mean()
    rating_std = data_iter._train_ratings.std()
    ### declare the loss information
    best_valid_rmse = np.inf
    no_better_valid = 0
    best_iter = -1
    avg_gnorm = 0
    count_rmse = 0
    count_num = 0
    count_loss = 0


    for iter_idx in range(1, args.train_max_iter):
        rating_node_pairs, gt_ratings = next(rating_sampler)
        nd_gt_ratings = mx.nd.array(gt_ratings, ctx=args.ctx, dtype=np.float32)
        if args.gen_r_use_classification:
            nd_gt_label = mx.nd.array(np.searchsorted(possible_rating_values, gt_ratings),
                                      ctx=args.ctx, dtype=np.int32)
        iter_graph = data_iter.train_graph
        ## remove the batch rating pair and link prediction pair (optional)
        if rating_node_pairs.shape[1] < data_iter._train_node_pairs.shape[1] and args.model_remove_rating:
            if iter_idx == 1:
                logging.info("Removing training edges within the batch...")
            iter_graph = iter_graph.remove_edges_by_id(src_key=dataset.name_user,
                                                       dst_key=dataset.name_item,
                                                       node_pair_ids=rating_node_pairs)

        with mx.autograd.record():
            pred_ratings = net.forward(graph=iter_graph,
                                       feature_dict=feature_dict,
                                       rating_node_pairs=rating_node_pairs,
                                       graph_sampler_args=graph_sampler_args,
                                       symm=args.gcn_agg_norm_symm)
            if args.gen_r_use_classification:
                loss = rating_loss_net(pred_ratings, nd_gt_label).mean()
            else:
                loss = rating_loss_net(mx.nd.reshape(pred_ratings, shape=(-1,)),
                                        (nd_gt_ratings - rating_mean) / rating_std ).mean()
            loss.backward()

        count_loss += loss.asscalar()
        gnorm = params_clip_global_norm(net.collect_params(), args.train_grad_clip, args.ctx)
        avg_gnorm += gnorm
        trainer.step(1.0) #, ignore_stale_grad=True)

        if iter_idx == 1:
            logging.info("Total #Param of net: %d" % (gluon_total_param_num(net)))
            logging.info(gluon_net_info(net, save_path=os.path.join(args.save_dir, 'net%d.txt' % args.save_id)))

        if args.gen_r_use_classification:
            real_pred_ratings = (mx.nd.softmax(pred_ratings, axis=1) *
                                 nd_possible_rating_values.reshape((1, -1))).sum(axis=1)
            rmse = mx.nd.square(real_pred_ratings - nd_gt_ratings).sum()
        else:
            rmse = mx.nd.square(pred_ratings.reshape((-1,)) * rating_std + rating_mean - nd_gt_ratings).sum()
        count_rmse += rmse.asscalar()
        count_num += pred_ratings.shape[0]

        if iter_idx % args.train_log_interval == 0:
            train_loss_logger.log(iter=iter_idx,
                                  loss=count_loss/(iter_idx+1), rmse=count_rmse/count_num)
            logging_str = "Iter={}, gnorm={:.3f}, loss={:.4f}, rmse={:4f}".format(
                iter_idx, avg_gnorm/args.train_log_interval, count_loss/(iter_idx+1), count_rmse/count_num)
            avg_gnorm = 0
            count_rmse = 0
            count_num = 0

        if iter_idx % args.train_valid_interval == 0:
            valid_rmse = evaluate(args=args,
                                  net=net,
                                  feature_dict=feature_dict,
                                  data_iter=data_iter,
                                  segment='valid')
            valid_loss_logger.log(**dict([('iter', iter_idx), ('rmse', valid_rmse)]))
            logging_str += ',\tVal RMSE={:.4f}.format(valid_rmse)'

            if valid_rmse < best_valid_rmse:
                best_valid_rmse = valid_rmse
                no_better_valid = 0
                best_iter = iter_idx
                #net.save_parameters(filename=os.path.join(args.save_dir, 'best_valid_net{}.params'.format(args.save_id)))
                test_rmse = evaluate(args=args, net=net, feature_dict=feature_dict, data_iter=data_iter, segment='test')
                best_test_rmse = test_rmse
                test_loss_logger.log(iter=iter_idx, rmse=test_rmse)
                logging_str += ', Test RMSE={:.4f}'.format(test_rmse)
            else:
                no_better_valid += 1
                if no_better_valid > args.train_early_stopping_patience\
                    and trainer.learning_rate <= args.train.min_lr:
                    logging.info("Early stopping threshold reached. Stop training.")
                    break
                if no_better_valid > args.train.decay_patience:
                    new_lr = max(trainer.learning_rate * args.train.decay_factor, args.train.min_lr)
                    if new_lr < trainer.learning_rate:
                        logging.info("\tChange the LR to %g" % new_lr)
                        trainer.set_learning_rate(new_lr)
                        no_better_valid = 0
        if iter_idx  % args.train_log_interval == 0:
            logging.info(logging_str)
    logging.info('Best Iter Idx={}, Best Valid RMSE={:.4f}, Best Test RMSE={:.4f}'.format(
        best_iter, best_valid_rmse, best_test_rmse))

    train_loss_logger.close()
    valid_loss_logger.close()
    test_loss_logger.close()


def config():
    parser = argparse.ArgumentParser(description='Run the baseline method.')

    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--ctx', dest='ctx', default='gpu', type=str,
                        help='Running Context. E.g `--ctx gpu` or `--ctx gpu0,gpu1` or `--ctx cpu`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--silent', action='store_true')

    parser.add_argument('--data_name', default='ml-100k', type=str,
                        help='The dataset name: ml-100k, ml-1m, ml-10m')
    parser.add_argument('--data_test_ratio', type=float, default=0.1)
    parser.add_argument('--data_valid_ratio', type=float, default=0.1)

    parser.add_argument('--model_remove_rating', type=bool, default=True)
    parser.add_argument('--model_activation', type=str, default="leaky")

    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--gcn_agg_units', type=int, default=500)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_agg_share_weights', type=bool, default=False)
    parser.add_argument('--gcn_agg_ordinal_share', type=bool, default=False)
    parser.add_argument('--gcn_out_accum_self', type=bool, default=False)
    parser.add_argument('--gcn_out_units', type=int, default=75)
    parser.add_argument('--gcn_out_accum', type=str, default="stack")
    parser.add_argument('--gcn_out_share_weights', type=bool, default=False)

    parser.add_argument('--gen_r_use_classification', type=int, default=2)
    parser.add_argument('--gen_r_num_basis_func', type=int, default=2)

    parser.add_argument('--train_rating_batch_size', type=int, default=10000)
    parser.add_argument('--train_max_iter', type=int, default=100000)
    parser.add_argument('--train_log_interval', type=int, default=10)
    parser.add_argument('--train_valid_interval', type=int, default=10)
    parser.add_argument('--train_optimizer', type=str, default="adam")
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_early_stopping_patience', type=int, default=150)
    parser.add_argument('--train_grad_clip', type=float, default=10.0)

    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)[0]


    ### configure save_fir to save all the info
    if args.save_dir is None:
        args.save_dir = args.data_name+"_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
    if args.save_id is None:
        args.save_id = random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == '__main__':
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    args = config()
    logging_config(folder=args.save_dir, name='log', no_console=args.silent)
    ### TODO save the args
    np.random.seed(args.seed)
    mx.random.seed(args.seed, args.ctx)
    set_seed(args.seed)
    train(args)
