"""
Contains argument parsers and utilities for saving and loading metrics and
models.
"""

import argparse
import glob
import os

import numpy as np

import backbone


model_dict = dict(
    Conv4=backbone.Conv4,
    Conv4NP=backbone.Conv4NP,
    Conv4S=backbone.Conv4S,
    Conv6=backbone.Conv6,
    ResNet10=backbone.ResNet10,
    ResNet18=backbone.ResNet18,
    PretrainedResNet18=backbone.PretrainedResNet18,
    ResNet34=backbone.ResNet34,
    ResNet50=backbone.ResNet50,
    ResNet101=backbone.ResNet101,
)


def parse_args(script):
    parser = argparse.ArgumentParser(
        description="few-shot script %s" % (script),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Specify checkpoint dir (if none, automatically generate)",
    )
    parser.add_argument("--model", default="Conv4", help="Choice of backbone")
    parser.add_argument("--method", default="matchingnet", help="Choice of algorithms: matchingnet/protonet")
    parser.add_argument(
        "--freeze_emb", action="store_true", help="Freeze LM word embedding layer"
    )

    parser.add_argument(
        "--train_n_way", default=5, type=int, help="class num to classify for training"
    )
    parser.add_argument(
        "--test_n_way",
        default=5,
        type=int,
        help="class num to classify for testing (validation) ",
    )
    parser.add_argument(
        "--n_shot",
        default=1,
        type=int,
        help="number of labeled data in each class, same as n_support",
    )
    parser.add_argument(
        "--n_workers",
        default=1,
        type=int,
        help="Use this many workers for loading data",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed (torch only; not numpy)"
    )

    if script == "train":
        parser.add_argument(
            "--n", default=1, type=int, help="Train run number (used for metrics)"
        )
        parser.add_argument(
            "--optimizer",
            default="adam",
            choices=["adam", "amsgrad", "rmsprop"],
            help="Optimizer",
        )
        parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
        parser.add_argument("--save_freq", default=50, type=int, help="Save frequency")
        parser.add_argument("--start_epoch", default=0, type=int, help="Starting epoch")
        parser.add_argument(
            "--stop_epoch", default=600, type=int, help="Stopping epoch"
        )  # for meta-learning methods, each epoch contains 100 episodes
        parser.add_argument(
            "--resume",
            action="store_true",
            help="continue from previous trained model with largest epoch",
        )
    elif script == "test":
        parser.add_argument(
            "--split",
            default="novel",
            choices=["base", "val", "novel"],
            help="which split to evaluate on",
        )
        parser.add_argument(
            "--save_iter",
            default=-1,
            type=int,
            help="saved feature from the model trained in x epoch, use the best model if x is -1",
        )
        parser.add_argument(
            "--record_file",
            default="./record/results.txt",
            help="Where to write results to",
        )
    else:
        raise ValueError("Unknown script")

    args = parser.parse_args()
    return args


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, "{:d}.tar".format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, "*.tar"))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != "best_model.tar"]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, "{:d}.tar".format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, "best_model.tar")
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
