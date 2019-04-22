import numpy as np
import mxnet as mx
import os
import argparse
import logging
from mxnet import gluon
from mxnet.gluon import nn
from datasets import LoadData
from graph import HeterGraph, merge_nodes, merge_node_ids_dict, empty_as_zero, NodeFeatures, set_seed
from mxgraph.layers import HeterGCNLayer, BiDecoder, StackedHeterGCNLayers, LayerDictionary, InnerProductLayer
from mxgraph.layers.common import get_activation, StackFCBlock
from mxgraph.iterators import HeterIterator
from mxgraph.utils import copy_to_ctx, ExponentialMovingAverage, copy_params_to_nd,\
    copy_nd_to_params, gluon_net_info, parse_ctx, logging_config, gluon_total_param_num, params_clip_global_norm
from mxgraph.helpers.metric_logger import MetricLogger
import random
import string



def load_dataset(args):
    dataset = LoadData(args.data.name, seed=args.seed,
                       use_input_test_set=True,
                       val_ratio=args.data.valid_ratio)
    all_graph = dataset.graph
    name_user = dataset.name_user
    name_item = dataset.name_item
    logging.info(dataset)
    # !IMPORTANT. We need to check that ids in all_graph are continuous from 0 to #Node - 1.
    # We will later use these ids to take the embedding vectors
    all_graph.check_continous_node_ids()
    feature_dict = dict()
    info_line = "Feature dim: "
    nd_user_indices = mx.nd.arange(all_graph.features[name_user].shape[0], ctx=args.ctx)
    nd_item_indices = mx.nd.arange(all_graph.features[name_item].shape[0], ctx=args.ctx)
    user_item_total = all_graph.features[name_user].shape[0] + all_graph.features[name_item].shape[0]
    feature_dict[name_user] = mx.nd.one_hot(nd_user_indices, user_item_total)
    feature_dict[name_item] = mx.nd.one_hot(nd_item_indices + nd_user_indices.shape[0], user_item_total)
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
        self._act = get_activation(args.model.activation)
        with self.name_scope():
            # Construct Encoder
            self.encoders = nn.Sequential(prefix='enc_')
            with self.encoders.name_scope():
                num_enc_blocks = 1
                for block_id in range(num_enc_blocks):
                    recurrent_layer_num = None
                    encoder = StackedHeterGCNLayers(recurrent_layer_num=recurrent_layer_num,
                                                    prefix='b{}_'.format(block_id))
                    with encoder.name_scope():
                        for i, (agg_units, out_units) in enumerate(zip(_GCN.AGG.UNITS, _GCN.OUT.UNITS)):
                            if (i == len(_GCN.AGG.UNITS) - 1):
                                source_keys = [name_user, name_item] ### For HeterGCN without link prediction training
                            else:
                                source_keys = all_graph.meta_graph.keys()
                            encoder.add(HeterGCNLayer(meta_graph=all_graph.meta_graph,
                                                      multi_link_structure=all_graph.get_multi_link_structure(),
                                                      dropout_rate=_GCN.DROPOUT,
                                                      agg_type='gcn',
                                                      agg_units=agg_units,
                                                      out_units=None,
                                                      source_keys=source_keys,
                                                      agg_ordinal_sharing=_GCN.AGG.ORDINAL_SHARE,
                                                      share_agg_weights=_GCN.AGG.SHARE_WEIGHTS,
                                                      agg_accum=_GCN.AGG.ACCUM,
                                                      agg_act=_MODEL.ACTIVATION,
                                                      accum_self=_GCN.OUT.ACCUM_SELF,
                                                      out_act=_MODEL.ACTIVATION,
                                                      layer_accum=_GCN.OUT.ACCUM,
                                                      share_out_fc_weights=_GCN.OUT.SHARE_WEIGHTS,
                                                      layer_norm=False,
                                                      prefix='l{}_'.format(i)))

                    self.encoders.add(encoder)


            if _GEN_RATING.USE_CLASSIFICATION:
                self.gen_ratings = BiDecoder(in_units=_GEN_RATING.MID_MAP,
                                             out_units=nratings,
                                             num_basis_functions=_GEN_RATING.NUM_BASIS_FUNC,
                                             prefix='gen_rating')
            else:
                self.gen_ratings = InnerProductLayer(prefix='gen_rating')

            self.rating_user_projs = nn.Sequential(prefix='rating_user_proj_')
            self.rating_item_projs = nn.Sequential(prefix='rating_item_proj_')
            for rating_proj in [self.rating_user_projs, self.rating_item_projs]:
                with rating_proj.name_scope():
                    num_blocks = 1
                    for block_id in range(num_blocks):
                        ele_proj = nn.HybridSequential(prefix='b{}_'.format(block_id))
                        with ele_proj.name_scope():
                            ele_proj.add(nn.Dense(units=_GEN_RATING.MID_MAP,
                                                  flatten=False))
                        rating_proj.add(ele_proj)



    def forward(self, graph, feature_dict, rating_node_pairs=None,
                graph_sampler_args=None, symm=None):
        """

        Parameters
        ----------
        graph : HeterGraph
        feature_dict : dict
            Dictionary contains the base features of all nodes
        rating_node_pairs : np.ndarray or None
            Shape: (2, #Edges), First row is user and the second row is item
        embed_noise_dict : dict or None
            Dictionary that contains the noises of all nodes that is used to replace the node ids for masked embedding
            {key: (#all node ids, ) the shape and order is the same as the node ids in the whole graph}
        recon_node_ids_dict: dict or None
            Dictionary that contains the nodes ids that we need to reconstruct the embedding
        all_masked_node_ids_dict : dict or None
            Dictionary that contains the node ids of all masked nodes
        graph_sampler_args : dict or None
            Arguments for graph sampler
        symm : bool
            Whether to calculate the support in the symmetric formula
        output_inner_result : bool
            Whether to output the inner results
        input_node_ids_dict : dict

        Returns
        -------
        pred_ratings : list of mx.nd.ndarray
            The predicted ratings. If we use the stacked hourglass AE structure.
             it will return a list with multiple predicted ratings
        pred_embeddings : list of dict
            The predicted embeddings. Return a list of predicted embeddings
             if we use the stacked hourglass AE structure.
        gt_embeddings : dict
            The ground-truth embedding of the target node ids.
        """
        if symm is None:
            symm = _GCN.AGG.NORM_SYMM
        ctx = next(iter(feature_dict.values())).context
        req_node_ids_dict = dict()
        encoder_fwd_plan = [None for _ in range(_MODEL.NBLOCKS)]
        encoder_fwd_indices = [None for _ in range(_MODEL.NBLOCKS)]
        pred_ratings = []
        pred_embeddings = []
        block_req_node_ids_dict = [None for _ in range(_MODEL.NBLOCKS)]


        # From top to bottom, generate the forwarding plan
        for block_id in range(_MODEL.NBLOCKS - 1, -1, -1):
            # Backtrack the encoders
            encoder = self.encoders[block_id]

            uniq_node_ids_dict, encoder_fwd_indices[block_id] = \
                merge_node_ids_dict([{self._name_user: rating_node_pairs[0],
                                      self._name_item: rating_node_pairs[1]},
                                     req_node_ids_dict])

            block_req_node_ids_dict[block_id] = req_node_ids_dict
            req_node_ids_dict, encoder_fwd_plan[block_id]\
                = encoder.gen_plan(graph=graph,
                                   sel_node_ids_dict=uniq_node_ids_dict,
                                   graph_sampler_args=graph_sampler_args,
                                   symm=symm)

        input_dict = dict()
        for key, req_node_ids in req_node_ids_dict.items():
            input_dict[key] = mx.nd.take(feature_dict[key],
                                         mx.nd.array(req_node_ids, ctx=ctx, dtype=np.int32))
        for block_id in range(_MODEL.NBLOCKS):
            encoder = self.encoders[block_id]
            output_dict = encoder.heter_sage(input_dict, encoder_fwd_plan[block_id])
            rating_idx_dict, req_idx_dict = encoder_fwd_indices[block_id]


            # Generate the predicted ratings
            if rating_node_pairs is not None:
                rating_user_fea = mx.nd.take(output_dict[self._name_user],
                                             mx.nd.array(rating_idx_dict[self._name_user], ctx=ctx, dtype=np.int32))
                rating_item_fea = mx.nd.take(output_dict[self._name_item],
                                             mx.nd.array(rating_idx_dict[self._name_item], ctx=ctx, dtype=np.int32))
                user_proj = self.rating_user_projs[block_id]
                item_proj = self.rating_item_projs[block_id]
                rating_user_fea = user_proj(rating_user_fea)
                rating_item_fea = item_proj(rating_item_fea)
                block_pred_ratings = self.gen_ratings(rating_user_fea, rating_item_fea)
                pred_ratings.append(block_pred_ratings)

        else:
            return pred_ratings, pred_embeddings, None



def evaluate(net, feature_dict, ctx, data_iter, segment='valid'):
    rating_mean = data_iter._train_ratings.mean()
    rating_std = data_iter._train_ratings.std()
    rating_sampler = data_iter.rating_sampler(batch_size=_TRAIN.RATING_BATCH_SIZE, segment=segment,
                                              sequential=True)
    possible_rating_values = data_iter.possible_rating_values
    nd_possible_rating_values = mx.nd.array(possible_rating_values, ctx=args.ctx, dtype=np.float32)
    eval_graph = data_iter.val_graph if segment == 'valid' else data_iter.test_graph
    graph_sampler_args = gen_graph_sampler_args(data_iter.all_graph.meta_graph)
    # Evaluate RMSE
    rmse_l = [0 for _ in range(_MODEL.NBLOCKS)]
    cnt = 0

    for rating_node_pairs, gt_ratings in rating_sampler:
        nd_gt_ratings = mx.nd.array(gt_ratings, dtype=np.float32, ctx=ctx)
        cnt += rating_node_pairs.shape[1]

        pred_ratings, _, _ \
            = net.forward(graph=eval_graph,
                          feature_dict=feature_dict,
                          rating_node_pairs=rating_node_pairs,
                          graph_sampler_args=graph_sampler_args,
                          symm=_GCN.AGG.NORM_SYMM)
        for i in range(_MODEL.NBLOCKS):
            if _GEN_RATING.USE_CLASSIFICATION:
                real_pred_ratings = (mx.nd.softmax(pred_ratings[i], axis=1) *
                                     nd_possible_rating_values.reshape((1, -1))).sum(axis=1)
                rmse_l[i] += mx.nd.square(real_pred_ratings - nd_gt_ratings).sum().asscalar()
            else:
                rmse_l[i] +=\
                    mx.nd.square(mx.nd.clip(pred_ratings[i].reshape((-1,)) * rating_std + rating_mean,
                                            possible_rating_values.min(),
                                            possible_rating_values.max()) - nd_gt_ratings).sum().asscalar()
    for i in range(_MODEL.NBLOCKS):
        rmse_l[i] = np.sqrt(rmse_l[i] / cnt)
    return rmse_l


def log_str(loss_l, loss_name):
    return ', ' + \
           ', '.join(
               ["{}{}={:.3f}".format(loss_name, i, loss_l[i][0] / loss_l[i][1])
                for i in range(len(loss_l))])

def train(seed):
    dataset, all_graph, feature_dict = load_dataset(seed)
    valid_node_pairs, _ = dataset.valid_data
    test_node_pairs, _ = dataset.test_data
    data_iter = HeterIterator(all_graph=all_graph,
                              name_user=dataset.name_user,
                              name_item=dataset.name_item,
                              test_node_pairs=test_node_pairs,
                              valid_node_pairs=valid_node_pairs,
                              seed=seed)

    logging.info(data_iter)
    ### build the net
    possible_rating_values = data_iter.possible_rating_values
    nd_possible_rating_values = mx.nd.array(possible_rating_values, ctx=args.ctx, dtype=np.float32)
    net = Net(all_graph=all_graph, nratings=possible_rating_values.size,
              name_user=dataset.name_user, name_item=dataset.name_item)
    net.initialize(init=mx.init.Xavier(factor_type='in'), ctx=args.ctx)
    net.hybridize()
    if _GEN_RATING.USE_CLASSIFICATION:
        rating_loss_net = gluon.loss.SoftmaxCELoss()
    else:
        rating_loss_net = gluon.loss.L2Loss()
    rating_loss_net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), _TRAIN.OPTIMIZER,
                            {'learning_rate': _TRAIN.LR, 'wd': _TRAIN.WD})
    ### prepare the logger
    train_loss_logger = MetricLogger(['iter', 'loss'] + sum([['rmse{}'.format(i),
                                                      'rating_loss{}'.format(i)] for i in range(_MODEL.NBLOCKS)],
                                                    []),
                                     ['%d', '%.4f'] + ['%.4f', '%.4f', '%.4f'] * _MODEL.NBLOCKS,
                                     os.path.join(args.save_dir, 'train_loss%d.csv' % args.save_id))
    valid_loss_logger = MetricLogger(['iter'] + ['rmse{}'.format(i) for i in range(_MODEL.NBLOCKS)],
                                     ['%d'] + ['%.4f'] * _MODEL.NBLOCKS,
                                     os.path.join(args.save_dir, 'valid_loss%d.csv' % args.save_id))
    test_loss_logger = MetricLogger(['iter'] + ['rmse{}'.format(i) for i in range(_MODEL.NBLOCKS)],
                                    ['%d'] + ['%.4f'] * _MODEL.NBLOCKS,
                                    os.path.join(args.save_dir, 'test_loss%d.csv' % args.save_id))
    ### initialize the iterator
    rating_sampler = data_iter.rating_sampler(batch_size=_TRAIN.RATING_BATCH_SIZE,
                                              segment='train')
    graph_sampler_args = gen_graph_sampler_args(all_graph.meta_graph)
    rating_mean = data_iter._train_ratings.mean()
    rating_std = data_iter._train_ratings.std()
    ### declare the loss information
    best_valid_rmse = np.inf
    best_test_rmse_l = None
    no_better_valid = 0
    best_iter = -1
    avg_gnorm = 0
    avg_rmse_l = [[0, 0] for _ in range(_MODEL.NBLOCKS)]
    avg_rating_loss_l = [[0, 0] for _ in range(_MODEL.NBLOCKS)]

    for iter_idx in range(1, _TRAIN.MAX_ITER):
        rating_node_pairs, gt_ratings = next(rating_sampler)
        nd_gt_ratings = mx.nd.array(gt_ratings, ctx=args.ctx, dtype=np.float32)
        if _GEN_RATING.USE_CLASSIFICATION:
            nd_gt_label = mx.nd.array(np.searchsorted(possible_rating_values, gt_ratings),
                                      ctx=args.ctx, dtype=np.int32)

        iter_graph = data_iter.train_graph
        ## remove the batch rating pair and link prediction pair (optional)
        if rating_node_pairs.shape[1] < data_iter._train_node_pairs.shape[1] and _MODEL.REMOVE_RATING:
            if iter_idx == 1:
                logging.info("Removing training edges within the batch...")
            iter_graph = iter_graph.remove_edges_by_id(src_key=dataset.name_user,
                                                       dst_key=dataset.name_item,
                                                       node_pair_ids=rating_node_pairs)

        with mx.autograd.record():
            pred_ratings, pred_embeddings, gt_embeddings\
                = net.forward(graph=iter_graph,
                              feature_dict=feature_dict,
                              rating_node_pairs=rating_node_pairs,
                              graph_sampler_args=graph_sampler_args,
                              symm=_GCN.AGG.NORM_SYMM)
            rating_loss_l = []
            for i in range(_MODEL.NBLOCKS):
                if _GEN_RATING.USE_CLASSIFICATION:
                    ele_loss = rating_loss_net(pred_ratings[i], nd_gt_label).mean()
                else:
                    ele_loss = rating_loss_net(mx.nd.reshape(pred_ratings[i], shape=(-1,)),
                                               (nd_gt_ratings - rating_mean) / rating_std ).mean()

                rating_loss_l.append( ele_loss)
            loss = sum(rating_loss_l)

            loss.backward()
        gnorm = params_clip_global_norm(net.collect_params(), _TRAIN.GRAD_CLIP, args.ctx)
        avg_gnorm += gnorm
        trainer.step(1.0) #, ignore_stale_grad=True)

        if iter_idx == 1:
            logging.info("Total #Param of net: %d" % (gluon_total_param_num(net)))
            logging.info(gluon_net_info(net, save_path=os.path.join(args.save_dir, 'net%d.txt' % args.save_id)))
        # Calculate the avg losses
        for i in range(_MODEL.NBLOCKS):
            if _GEN_RATING.USE_CLASSIFICATION:
                real_pred_ratings = (mx.nd.softmax(pred_ratings[i], axis=1) *
                                     nd_possible_rating_values.reshape((1, -1))).sum(axis=1)
                rmse = mx.nd.square(real_pred_ratings - nd_gt_ratings).sum()
            else:
                rmse = mx.nd.square(pred_ratings[i].reshape((-1,)) * rating_std + rating_mean
                                    - nd_gt_ratings).sum()
            avg_rmse_l[i][0] += rmse.asscalar()
            avg_rmse_l[i][1] += pred_ratings[i].shape[0]

            avg_rating_loss_l[i][0] += rating_loss_l[i].asscalar()
            avg_rating_loss_l[i][1] += 1

        if iter_idx % _TRAIN.LOG_INTERVAL == 0:
            train_loss_info = dict({'iter': iter_idx})
            train_loss_info['loss'] = loss.asscalar()
            for i in range(_MODEL.NBLOCKS):
                train_loss_info['rmse{}'.format(i)] = np.sqrt(avg_rmse_l[i][0] / avg_rmse_l[i][1])
                train_loss_info['rating_loss{}'.format(i)] = avg_rating_loss_l[i][0] / avg_rating_loss_l[i][1]

            train_loss_logger.log(**train_loss_info)

            logging_str = "Iter={}, gnorm={:.3f}, loss={:.3f}".format(iter_idx,
                                                                      avg_gnorm / _TRAIN.LOG_INTERVAL, loss.asscalar())
            logging_str += log_str(avg_rating_loss_l, "RT")

            logging_str += ', '  + ', '.join(["RMSE{}={:.4f}".format(i, np.sqrt(avg_rmse_l[i][0] / avg_rmse_l[i][1]))
                                              for i in range(_MODEL.NBLOCKS)])

            avg_gnorm = 0
            avg_rmse_l = [[0, 0] for _ in range(_MODEL.NBLOCKS)]
        if iter_idx % _TRAIN.VALID_INTERVAL == 0:
            valid_rmse_l = evaluate(net=net,
                                    feature_dict=feature_dict,
                                    ctx=args.ctx,
                                    data_iter=data_iter,
                                    segment='valid')
            valid_loss_logger.log(**dict([('iter', iter_idx)] + [('rmse{}'.format(i), ele_rmse)
                                                                 for i, ele_rmse in enumerate(valid_rmse_l)]))
            logging_str += ',\t' + ', '.join(["Val RMSE{}={:.3f}".format(i, ele_rmse)
                                             for i, ele_rmse in enumerate(valid_rmse_l)])

            if valid_rmse_l[-1] < best_valid_rmse:
                best_valid_rmse = valid_rmse_l[-1]
                no_better_valid = 0
                best_iter = iter_idx
                #net.save_parameters(filename=os.path.join(args.save_dir, 'best_valid_net{}.params'.format(args.save_id)))
                test_rmse_l = evaluate(net=net, feature_dict=feature_dict, ctx=args.ctx,
                                       data_iter=data_iter, segment='test')
                best_test_rmse_l = test_rmse_l
                test_loss_logger.log(**dict([('iter', iter_idx)] + [('rmse{}'.format(i), ele_rmse)
                                                                    for i, ele_rmse in enumerate(test_rmse_l)]))
                logging_str += ', ' + ', '.join(["Test RMSE{}={:.4f}".format(i, ele_rmse)
                                                 for i, ele_rmse in enumerate(test_rmse_l)])
            else:
                no_better_valid += 1
                if no_better_valid > _TRAIN.EARLY_STOPPING_PATIENCE\
                    and trainer.learning_rate <= _TRAIN.MIN_LR:
                    logging.info("Early stopping threshold reached. Stop training.")
                    break
                if no_better_valid > _TRAIN.DECAY_PATIENCE:
                    new_lr = max(trainer.learning_rate * _TRAIN.LR_DECAY_FACTOR, _TRAIN.MIN_LR)
                    if new_lr < trainer.learning_rate:
                        logging.info("\tChange the LR to %g" % new_lr)
                        trainer.set_learning_rate(new_lr)
                        no_better_valid = 0
        if iter_idx  % _TRAIN.LOG_INTERVAL == 0:
            logging.info(logging_str)
    logging.info('Best Iter Idx={}, Best Valid RMSE={:.4f}, '.format(best_iter, best_valid_rmse) +
                 ', '.join(["Best Test RMSE{}={:.4f}".format(i, ele_rmse)
                            for i, ele_rmse in enumerate(best_test_rmse_l)]))

    train_loss_logger.close()
    valid_loss_logger.close()
    test_loss_logger.close()


def config():
    parser = argparse.ArgumentParser(description='Run the baseline method.')

    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--ctx', dest='ctx', default='gpu', type=str,
                        help='Running Context. E.g `--ctx gpu` or `--ctx gpu0,gpu1` or `--ctx cpu`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--silent', action='store_true')

    parser.add_argument('--data.name', type=str, help='The dataset name: ml-100k, ml-1m, ml-10m', default='ml-100k')
    parser.add_argument('--data.valid_ratio', type=float, default=0.1)

    parser.add_argument('--model.remove_rating', type=bool, default=True)
    parser.add_argument('--model.activation', type=str, default="leaky")

    parser.add_argument('--gcn.dropout', type=float, default=0.7)
    parser.add_argument('--gcn.agg.norm_symm', type=bool, default=True)
    parser.add_argument('--gcn.agg.units', type=int, default=500)
    parser.add_argument('--gcn.agg.accum', type=str, default="sum")
    parser.add_argument('--gcn.agg.share_weights', type=bool, default=False)
    parser.add_argument('--gcn.agg.ordinal_share', type=bool, default=False)
    parser.add_argument('--gcn.out.accum_self', type=bool, default=False)
    parser.add_argument('--gcn.out.units', type=int, default=75)
    parser.add_argument('--gcn.out.accum', type=str, default="stack")
    parser.add_argument('--gcn.out.share_weights', type=bool, default=False)

    parser.add_argument('--train.rating_batch_size', type=int, default=10000)
    parser.add_argument('--train.max_iter', type=int, default=100000)
    parser.add_argument('--train.log_interval', type=int, default=10)
    parser.add_argument('--train.valid_interval', type=int, default=10)
    parser.add_argument('--train.optimizer', type=str, default="adam")
    parser.add_argument('--train.lr', type=float, default=0.01)
    parser.add_argument('--train.lr', type=float, default=0.01)
    parser.add_argument('--train.lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train.early_stopping_patience', type=int, default=150)
    parser.add_argument('--train.grad_clip', type=float, default=10.0)

    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)[0]



    ### configure save_fir to save all the info
    if args.save_dir is None:
        args.save_dir = args.data.name+"_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
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
    train(seed=args.seed)
