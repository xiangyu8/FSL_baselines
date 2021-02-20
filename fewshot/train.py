"""
Train script.
"""

import json
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim
from tqdm import tqdm

import constants
from data.datamgr import SetDataManager
from io_utils import get_resume_file, model_dict, parse_args
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet

def get_optimizer(model, args):
    """
    Get the optimizer for the model based on arguments. Specifically, if
    needed, we split up training into (1) main parameters, (2) RNN-specific
    parameters, with different learning rates if specified.

    :param model: nn.Module to train
    :param args: argparse.Namespace - other args passed to the script

    :returns: a torch.optim.Optimizer
    """
    # Get params
    main_params = {"params": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        else:
            main_params["params"].append(param)
    params_to_optimize = [main_params]

    # Define optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)
    elif args.optimizer == "amsgrad":
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, amsgrad=True)
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(params_to_optimize, lr=args.lr)
    else:
        raise NotImplementedError("optimizer = {}".format(args.optimizer))
    return optimizer


def train(
    base_loader,
    val_loader,
    model,
    start_epoch,
    stop_epoch,
    args,
    metrics_fname="metrics.json",
):
    """
    Main training script.

    :param base_loader: torch.utils.data.DataLoader for training set, generated
        by data.datamgr.SetDataManager
    :param val_loader: torch.utils.data.DataLoader for validation set,
        generated by data.datamgr.SetDataManager
    :param model: nn.Module to train
    :param start_epoch: which epoch we started at
    :param stop_epoch: which epoch to end at
    :param args: other arguments passed to the script
    "param metrics_fname": where to save metrics
    """
    optimizer = get_optimizer(model, args)

    max_val_acc = 0
    best_epoch = 0

    val_accs = []
    val_losses = []
    all_metrics = defaultdict(list)
    for epoch in tqdm(
        range(start_epoch, stop_epoch), total=stop_epoch - start_epoch, desc="Train"
    ):
        model.train()
        metric = model.train_loop(epoch, base_loader, optimizer)
        for m, val in metric.items():
            all_metrics[m].append(val)
        model.eval()

        os.makedirs(args.checkpoint_dir, exist_ok=True)

        val_acc, val_loss = model.test_loop(val_loader,)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        if val_acc > max_val_acc:
            best_epoch = epoch
            tqdm.write("best model! save...")
            max_val_acc = val_acc
            outfile = os.path.join(args.checkpoint_dir, "best_model.tar")
            torch.save({"epoch": epoch, "state": model.state_dict()}, outfile)

        if epoch and (epoch % args.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(args.checkpoint_dir, "{:d}.tar".format(epoch))
            torch.save({"epoch": epoch, "state": model.state_dict()}, outfile)
        tqdm.write("")

        # Save metrics
        metrics = {
            "train_acc": all_metrics["train_acc"],
            "current_train_acc": all_metrics["train_acc"][-1],
            "train_loss": all_metrics["train_loss"],
            "current_train_loss": all_metrics["train_loss"][-1],
            "current_epoch": epoch,
            "val_acc": val_accs,
            "val_loss": val_losses,
            "current_val_loss": val_losses[-1],
            "current_val_acc": val_acc,
            "best_epoch": best_epoch,
            "best_val_acc": max_val_acc,
        }
        with open(os.path.join(args.checkpoint_dir, metrics_fname), "w") as fout:
            json.dump(metrics, fout, sort_keys=True, indent=4, separators=(",", ": "))

        # Save a copy to current metrics too
        if (
            metrics_fname != "metrics.json"
            and metrics_fname.startswith("metrics_")
            and metrics_fname.endswith(".json")
        ):
            metrics["n"] = int(metrics_fname[8])
            with open(os.path.join(args.checkpoint_dir, "metrics.json"), "w") as fout:
                json.dump(
                    metrics, fout, sort_keys=True, indent=4, separators=(",", ": ")
                )

    # If didn't train, save model anyways
    if stop_epoch == 0:
        outfile = os.path.join(args.checkpoint_dir, "best_model.tar")
        torch.save({"epoch": stop_epoch, "state": model.state_dict()}, outfile)


if __name__ == "__main__":
    args = parse_args("train")

    base_file = os.path.join(constants.DATA_DIR, "base.json")
    val_file = os.path.join(constants.DATA_DIR, "val.json")

    # if test_n_way is smaller than train_n_way, reduce n_query to keep batch
    # size small
    n_query = max(1, int(16 * args.test_n_way / args.train_n_way))

    train_few_shot_args = dict(n_way=args.train_n_way, n_support=args.n_shot)
    base_datamgr = SetDataManager(84, n_query=n_query, **train_few_shot_args, args=args)
    print("Loading train data")

    base_loader = base_datamgr.get_data_loader(base_file,aug=True)
    val_datamgr = SetDataManager(84,n_query=n_query,n_way=args.test_n_way,n_support=args.n_shot,args=args)
    print("Loading val data\n")
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

    if args.method in ['protonet','matchingnet']:
        if args.method == 'matchingnet':
            model  = MatchingNet( model_dict[args.model], **train_few_shot_args )
        elif args.method == 'protonet':
            model = ProtoNet( model_dict[args.model], **train_few_shot_args )
    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    start_epoch = args.start_epoch
    stop_epoch = args.stop_epoch

    if args.resume:
        resume_file = get_resume_file(args.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp["epoch"] + 1
            model.load_state_dict(tmp["state"])

    metrics_fname = "metrics_{}.json".format(args.n)

    train(
        base_loader,
        val_loader,
        model,
        start_epoch,
        stop_epoch,
        args,
        metrics_fname=metrics_fname
    )