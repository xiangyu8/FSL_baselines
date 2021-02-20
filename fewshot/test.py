"""
Test script.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data.sampler

import constants
from data.datamgr import SetDataManager, TransformLoader
from io_utils import get_assigned_file, get_best_file, model_dict, parse_args

from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet

if __name__ == "__main__":
    args = parse_args("test")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    acc_all = []
    
    test_few_shot_args = dict(n_way = args.test_n_way, n_support = args.n_shot)
    if args.method in ['protonet','matchingnet']:
        if args.method == 'matchingnet':
            model  = MatchingNet( model_dict[args.model], **test_few_shot_args )
        elif args.method == 'protonet':
            model = ProtoNet( model_dict[args.model], **test_few_shot_args )
    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    if args.save_iter != -1:
        modelfile = get_assigned_file(args.checkpoint_dir, args.save_iter)
    else:
        modelfile = get_best_file(args.checkpoint_dir)
        print(args.checkpoint_dir)

    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp["state"])
        print("best epoch: ", tmp["epoch"])
    # Run the test loop for 600 iterations
    ITER_NUM = 600
    N_QUERY = 15

    test_datamgr = SetDataManager(
        84,
        n_query=N_QUERY,
        n_way=args.test_n_way,
        n_support=args.n_shot,
        n_episode=ITER_NUM,
        args=args,
    )
    test_loader = test_datamgr.get_data_loader(
        os.path.join(constants.DATA_DIR, f"{args.split}.json"),
        aug=False,
    )
    model.eval()
    acc_mean,_, acc_std = model.test_loop(test_loader, return_std = True)

    with open(args.record_file, "a") as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        acc_ci = 1.96 * acc_std / np.sqrt(ITER_NUM)
        f.write(
            json.dumps(
                {
                    "time": timestamp,
                    "split": args.split,
                    "setting": args.checkpoint_dir,
                    "iter_num": ITER_NUM,
                    "acc": acc_mean,
                    "acc_ci": acc_ci,
                    "acc_all": list(acc_all),
                    "acc_std": acc_std,
                },
                sort_keys=True,
            )
        )
        f.write("\n")
