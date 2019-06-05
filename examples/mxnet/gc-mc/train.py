import os
import argparse
import logging
import random
import string
import numpy as np
import mxnet as mx
from mxnet import gluon
from data import MovieLens
from model import GCMCLayer, BiDecoder, InnerProductLayer
from utils import get_activation, parse_ctx, \
    gluon_net_info, gluon_total_param_num, params_clip_global_norm, \
    logging_config, MetricLogger
from mxnet.gluon import nn, HybridBlock, Block


def load_dataset(args):
    dataset = MovieLens(args.data_name)
    # !IMPORTANT. We need to check that ids in all_graph are continuous from 0 to #Node - 1.
    # We will later use these ids to take the embedding vectors
    feature_dict = dict()
    nd_user_indices = mx.nd.arange(dataset.user_features.shape[0], ctx=args.ctx)
    nd_item_indices = mx.nd.arange(dataset.movie_features.shape[0], ctx=args.ctx)
    user_item_total = dataset.user_features.shape[0] + dataset.movie_features.shape[0]
    feature_dict["user"] = mx.nd.one_hot(nd_user_indices, user_item_total)
    feature_dict["movie"] = mx.nd.one_hot(nd_item_indices + nd_user_indices.shape[0], user_item_total)

    info_line = "Feature dim: "
    info_line += "\nUser: {}".format(feature_dict["user"].shape)
    info_line += "\nMovie: {}".format(feature_dict["movie"].shape)
    print(info_line)

    return dataset, feature_dict

class Net(HybridBlock):
    def __init__(self, nratings, name_user, name_item, args, **kwargs):
        super(Net, self).__init__(**kwargs)
        self._nratings = nratings
        self._name_user = name_user
        self._name_item = name_item
        self._act = get_activation(args.model_activation)
        with self.name_scope():
            # Construct Encoder
            self.encoder = GCMCLayer(agg_units=args.gcn_agg_units,
                                     out_units=args.gcn_out_units,
                                     num_links=5,
                                     src_key="user",
                                     dst_key="movie",
                                     dropout_rate=args.gcn_dropout,
                                     agg_accum=args.gcn_agg_accum,
                                     agg_act=args.model_activation,
                                     out_act = args.model_activation,
                                     prefix='enc_')
            if args.gen_r_use_classification:
                self.gen_ratings = BiDecoder(in_units=args.gcn_out_units,
                                             out_units=nratings,
                                             num_basis_functions=args.gen_r_num_basis_func,
                                             prefix='gen_rating')
            else:
                self.gen_ratings = InnerProductLayer(prefix='gen_rating')


    def hybrid_forward(self, F, uv_graph, vu_graph, rating_node_pairs):
        output_l = self.encoder(uv_graph, vu_graph, "user", "movie")
        # Generate the predicted ratings
        rating_user_fea = F.take(output_l[0], rating_node_pairs[0])
        rating_item_fea = F.take(output_l[1], rating_node_pairs[1])
        pred_ratings = self.gen_ratings(rating_user_fea, rating_item_fea)

        return pred_ratings



# def evaluate(args, net, feature_dict, data_iter, segment='valid'):
#     rating_mean = data_iter._train_ratings.mean()
#     rating_std = data_iter._train_ratings.std()
#     rating_sampler = data_iter.rating_sampler(batch_size=args.train_rating_batch_size, segment=segment,
#                                               sequential=True)
#     possible_rating_values = data_iter.possible_rating_values
#     nd_possible_rating_values = mx.nd.array(possible_rating_values, ctx=args.ctx, dtype=np.float32)
#     eval_graph = data_iter.val_graph if segment == 'valid' else data_iter.test_graph
#     graph_sampler_args = gen_graph_sampler_args(data_iter.all_graph.meta_graph)
#     # Evaluate RMSE
#     cnt = 0
#     rmse = 0
#
#     for rating_node_pairs, gt_ratings in rating_sampler:
#         nd_gt_ratings = mx.nd.array(gt_ratings, dtype=np.float32, ctx=args.ctx)
#         cnt += rating_node_pairs.shape[1]
#         pred_ratings = net.forward(graph=eval_graph,
#                                    feature_dict=feature_dict,
#                                    rating_node_pairs=rating_node_pairs,
#                                    graph_sampler_args=graph_sampler_args,
#                                    symm=args.gcn_agg_norm_symm)
#         if args.gen_r_use_classification:
#             real_pred_ratings = (mx.nd.softmax(pred_ratings, axis=1) *
#                                  nd_possible_rating_values.reshape((1, -1))).sum(axis=1)
#             rmse += mx.nd.square(real_pred_ratings - nd_gt_ratings).sum().asscalar()
#         else:
#             rmse += mx.nd.square(mx.nd.clip(pred_ratings.reshape((-1,)) * rating_std + rating_mean,
#                                             possible_rating_values.min(),
#                                             possible_rating_values.max()) - nd_gt_ratings).sum().asscalar()
#     rmse  = np.sqrt(rmse / cnt)
#     return rmse

def train(args):
    dataset, feature_dict = load_dataset(args)

    ### build the net
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = mx.nd.array(possible_rating_values, ctx=args.ctx, dtype=np.float32)
    net = Net(nratings=possible_rating_values.size,
              name_user="user", name_item="movie",
              args=args)
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

    ### prepare data
    train_rating_pair = mx.nd.array(dataset.train_rating_pairs, ctx=args.ctx, dtype=np.int64)
    nd_gt_ratings = mx.nd.array(dataset.train_rating_values, ctx=args.ctx, dtype=np.float32)
    rating_mean = dataset.train_rating_values.mean()
    rating_std = dataset.train_rating_values.std()

    uv_train_graph = dataset.uv_train_graph
    uv_train_support_l = dataset.compute_support(uv_train_graph.adjacency_matrix_scipy(("user", "movie", "rating")),
                                                 dataset.num_link, args.gcn_agg_norm_symm)
    """
    for idx, support in enumerate(uv_train_support_l):
        sup_coo = support.tocoo()
        uv_train_graph.edges[np.array(sup_coo.row, dtype=np.int64),
                             np.array(sup_coo.col, dtype=np.int64)].data['support{}'.format(idx)] = \
            mx.nd.array(sup_coo.data, ctx=args.ctx, dtype=np.float32)
    """
    uv_train_graph["user"].ndata["h"] = mx.nd.array(feature_dict["user"], ctx=args.ctx, dtype=np.float32)
    uv_train_graph["movie"].ndata["h"] = mx.nd.array(feature_dict["movie"], ctx=args.ctx, dtype=np.float32)

    vu_train_graph = dataset.vu_train_graph
    vu_train_support_l = dataset.compute_support(vu_train_graph.adjacency_matrix_scipy(("movie", "user", "rating")),
                                                 dataset.num_link, args.gcn_agg_norm_symm)
    """
    for idx, support in enumerate(vu_train_support_l):
        sup_coo = support.tocoo()
        vu_train_graph.edges[np.array(sup_coo.row, dtype=np.int64),
                             np.array(sup_coo.col, dtype=np.int64)].data['support{}'.format(idx)] = \
            mx.nd.array(sup_coo.data, ctx=args.ctx, dtype=np.float32)
    """
    vu_train_graph["movie"].ndata["h"] = mx.nd.array(feature_dict["movie"], ctx=args.ctx, dtype=np.float32)
    vu_train_graph["user"].ndata["h"] = mx.nd.array(feature_dict["user"], ctx=args.ctx, dtype=np.float32)

    ### declare the loss information
    best_valid_rmse = np.inf
    no_better_valid = 0
    best_iter = -1
    avg_gnorm = 0
    count_rmse = 0
    count_num = 0
    count_loss = 0

    for iter_idx in range(1, args.train_max_iter):
        if args.gen_r_use_classification:
            nd_gt_label = mx.nd.array(np.searchsorted(possible_rating_values, nd_gt_ratings),
                                      ctx=args.ctx, dtype=np.int32)

        with mx.autograd.record():
            pred_ratings = net.forward(uv_graph=uv_train_graph,
                                       vu_graph=vu_train_graph,
                                       rating_node_pairs=train_rating_pair)
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
            print("Total #Param of net: %d" % (gluon_total_param_num(net)))
            print(gluon_net_info(net, save_path=os.path.join(args.save_dir, 'net%d.txt' % args.save_id)))

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
    #
    #     if iter_idx % args.train_valid_interval == 0:
    #         valid_rmse = evaluate(args=args,
    #                               net=net,
    #                               feature_dict=feature_dict,
    #                               data_iter=data_iter,
    #                               segment='valid')
    #         valid_loss_logger.log(iter = iter_idx, rmse = valid_rmse)
    #         logging_str += ',\tVal RMSE={:.4f}'.format(valid_rmse)
    #
    #         if valid_rmse < best_valid_rmse:
    #             best_valid_rmse = valid_rmse
    #             no_better_valid = 0
    #             best_iter = iter_idx
    #             #net.save_parameters(filename=os.path.join(args.save_dir, 'best_valid_net{}.params'.format(args.save_id)))
    #             test_rmse = evaluate(args=args, net=net, feature_dict=feature_dict, data_iter=data_iter, segment='test')
    #             best_test_rmse = test_rmse
    #             test_loss_logger.log(iter=iter_idx, rmse=test_rmse)
    #             logging_str += ', Test RMSE={:.4f}'.format(test_rmse)
    #         else:
    #             no_better_valid += 1
    #             if no_better_valid > args.train_early_stopping_patience\
    #                 and trainer.learning_rate <= args.train_min_lr:
    #                 logging.info("Early stopping threshold reached. Stop training.")
    #                 break
    #             if no_better_valid > args.train_decay_patience:
    #                 new_lr = max(trainer.learning_rate * args.train_lr_decay_factor, args.train_min_lr)
    #                 if new_lr < trainer.learning_rate:
    #                     logging.info("\tChange the LR to %g" % new_lr)
    #                     trainer.set_learning_rate(new_lr)
    #                     no_better_valid = 0
        if iter_idx  % args.train_log_interval == 0:
            logging.info(logging_str)
    # logging.info('Best Iter Idx={}, Best Valid RMSE={:.4f}, Best Test RMSE={:.4f}'.format(
    #     best_iter, best_valid_rmse, best_test_rmse))

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
    parser.add_argument('--gcn_agg_share_weights', type=bool, default=True)
    parser.add_argument('--gcn_agg_ordinal_share', type=bool, default=False)
    parser.add_argument('--gcn_out_accum_self', type=bool, default=False)
    parser.add_argument('--gcn_out_share_weights', type=bool, default=True)
    parser.add_argument('--gcn_out_units', type=int, default=75)
    parser.add_argument('--gcn_out_accum', type=str, default="stack")

    parser.add_argument('--gen_r_use_classification', type=int, default=2)
    parser.add_argument('--gen_r_num_basis_func', type=int, default=2)

    parser.add_argument('--train_rating_batch_size', type=int, default=10000)
    parser.add_argument('--train_max_iter', type=int, default=100000)
    parser.add_argument('--train_log_interval', type=int, default=10)
    parser.add_argument('--train_valid_interval', type=int, default=10)
    parser.add_argument('--train_optimizer', type=str, default="adam")
    parser.add_argument('--train_grad_clip', type=float, default=10.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.0001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=50)
    parser.add_argument('--train_early_stopping_patience', type=int, default=150)

    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)[0]


    ### configure save_fir to save all the info
    if args.save_dir is None:
        args.save_dir = args.data_name+"_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
    if args.save_id is None:
        args.save_id = np.random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == '__main__':
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    args = config()
    #logging_config(folder=args.save_dir, name='log', no_console=args.silent)
    ### TODO save the args
    np.random.seed(args.seed)
    mx.random.seed(args.seed, args.ctx)
    train(args)
