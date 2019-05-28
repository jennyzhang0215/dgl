import numpy as np
import mxnet as mx
import time
import os
import argparse
import logging
from mxnet import gluon
from mxnet.gluon import nn
import random
import string

import dgl
from utils import parse_ctx,
def config():
    parser = argparse.ArgumentParser(description='Run the baseline method.')
    parser.add_argument('--cfg', dest='cfg_file', help='Optional configuration file',
                        default=None, type=str)
    parser.add_argument('--ctx', dest='ctx', default='gpu',
                        help='Running Context. E.g `--ctx gpu` or `--ctx gpu0,gpu1` or `--ctx cpu`', type=str)
    parser.add_argument('--save_dir', help='The saving directory', type=str)
    parser.add_argument('--dataset', help='The dataset name: ml-100k, ml-1m, ml-10m', type=str,
                        default='ml-100k')
    parser.add_argument('--save_embed', dest='save_embed', help='Whether to save the inner embedding '
                                                                'after the model is trained.',
                        action='store_true')
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--silent', action='store_true')
    args = parser.parse_args()
    args.ctx = parse_ctx(args.ctx)[0]

    cfg = edict()
    cfg.SEED = args.seed
    cfg.DATASET = edict()
    cfg.DATASET.NAME = args.dataset  # e.g. ml-100k
    cfg.DATASET.VALID_RATIO = 0.1

    cfg.MODEL = edict()
    cfg.MODEL.REMOVE_RATING = True
    cfg.MODEL.NBLOCKS = 1  # Number of AE blocks. NBLOCK = 3 ==> AE1 --> AE2 --> AE3 Like the hourglass structure
    cfg.MODEL.ACTIVATION = "leaky"

    cfg.GRAPH_SAMPLER = edict()  # Sample a random number of neighborhoods for mini-batch training
    cfg.GRAPH_SAMPLER.NUM_NEIGHBORS = -1


    cfg.GCN = edict()
    cfg.GCN.TYPE = 'gcn'
    cfg.GCN.DROPOUT = 0.7
    cfg.GCN.USE_RECURRENT = False  # Whether to use recurrent connections
    cfg.GCN.AGG = edict()
    cfg.GCN.AGG.NORM_SYMM = True
    cfg.GCN.AGG.UNITS = [500]  # Number of aggregator units
    cfg.GCN.AGG.ACCUM = "sum"
    cfg.GCN.AGG.SHARE_WEIGHTS = False
    cfg.GCN.AGG.ORDINAL_SHARE = False
    cfg.GCN.OUT = edict()
    cfg.GCN.OUT.ACCUM_SELF = False
    cfg.GCN.OUT.UNITS = [75]  # [50, 100] ### the hidden state of FC
    cfg.GCN.OUT.ACCUM = "stack"
    cfg.GCN.OUT.SHARE_WEIGHTS = False

    cfg.GEN_RATING = edict()
    cfg.GEN_RATING.MID_MAP = 64

    cfg.TRAIN = edict()
    cfg.TRAIN.RATING_BATCH_SIZE = 10000
    cfg.TRAIN.MAX_ITER = 1000000  ### Need to tune
    cfg.TRAIN.LOG_INTERVAL = 10
    cfg.TRAIN.VALID_INTERVAL = 10
    cfg.TRAIN.OPTIMIZER = "adam"
    cfg.TRAIN.LR = 1E-2  # initial learning rate
    cfg.TRAIN.WD = 0.0
    cfg.TRAIN.DECAY_PATIENCE = 100
    cfg.TRAIN.MIN_LR = 5E-4
    cfg.TRAIN.LR_DECAY_FACTOR = 0.5
    cfg.TRAIN.EARLY_STOPPING_PATIENCE = 150
    cfg.TRAIN.GRAD_CLIP = 10.0

    # if args.cfg_file is not None:
    #     cfg_from_file(args.cfg_file, target=cfg)

    ### configure save_fir to save all the info
    if args.save_dir is None:
        if args.cfg_file is None:
            #args.save_dir = "100k_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
            raise ValueError("Must set --cfg if not set --save_dir")
        args.save_dir = os.path.splitext(args.cfg_file)[0]
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    #args.save_id = save_cfg_dir(args.save_dir, source=cfg)

    ### Some Default flag
    cfg.DATASET.USE_INPUT_TEST_SET = True
    cfg.DATASET.TEST_RATIO = 0.2
    cfg.GEN_RATING.USE_CLASSIFICATION = False
    cfg.GEN_RATING.NUM_BASIS_FUNC = 2
    return cfg, args

def load_dataset():
