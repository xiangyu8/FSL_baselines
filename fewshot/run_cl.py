"""
Run everything on codalab.
"""

import json
import os
from subprocess import check_call


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Run everything on codalab",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    cl_parser = parser.add_argument_group(
        "Codalab args", "args to control high level codalab eval"
    )
    cl_parser.add_argument(
        "--no_train", action="store_true", help="Don't run the train command"
    )
    cl_parser.add_argument(
        "--log_dir", default="./test/", help="Where to save metrics/models"
    )
    cl_parser.add_argument("--n", default=1, type=int, help="Number of runs")

    fparser = parser.add_argument_group(
        "Few shot args", "args to pass to few shot scripts"
    )
    fparser.add_argument("--model", default="Conv4")
    fparser.add_argument("--method", default="matchingnet")
    fparser.add_argument(
        "--save_freq", type=int, default=10000
    )  # In CL script, by default, never save, just keep best model
    fparser.add_argument("--lr", type=float, default=1e-3)
    fparser.add_argument(
        "--optimizer", default="adam", choices=["adam", "amsgrad", "rmsprop"]
    )
    fparser.add_argument("--n_way", type=int, default=5)
    fparser.add_argument(
        "--test_n_way",
        type=int,
        default=None,
        help="Specify to change n_way eval at test",
    )
    fparser.add_argument("--n_shot", type=int, default=1)
    fparser.add_argument("--epochs", type=int, default=600)
    fparser.add_argument("--n_workers", type=int, default=4)
    fparser.add_argument("--resume", action="store_true")
    fparser.add_argument("--debug", action="store_true")
    fparser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if args.test_n_way is None:
        args.test_n_way = args.n_way

    args.cl_dir = os.path.join(args.log_dir, "checkpoints")
    args.cl_record_file = os.path.join(args.log_dir, "results_novel.json")
    args.cl_args_file = os.path.join(args.log_dir, "args.json")

    os.makedirs(args.log_dir, exist_ok=True)
    if os.path.exists(args.cl_record_file):
        os.remove(args.cl_record_file)

    # Save arg metadata to root directory
    # Only save if training a model
    print("==== RUN_CL: PARAMS ====")
    argsv = vars(args)
    print(argsv)
    if not args.no_train:
        with open(args.cl_args_file, "w") as fout:
            json.dump(argsv, fout, sort_keys=True, indent=4, separators=(",", ": "))

    # Train
    for i in range(1, args.n + 1):
              
        if not args.no_train:
            print("==== RUN_CL ({}/{}): TRAIN ====".format(i, args.n))
            train_cmd = [
                "python3",
                "fewshot/train.py",
                "--model",
                args.model,
                "--method",
                args.method,
                "--n_shot",
                args.n_shot,
                "--train_n_way",
                args.n_way,
                "--test_n_way",
                args.test_n_way,
                "--stop_epoch",
                args.epochs,
                "--stop_epoch",
                args.epochs,
                "--checkpoint_dir",
                args.cl_dir,
                "--save_freq",
                args.save_freq,
                "--n",
                i,
                "--lr",
                args.lr,
                "--optimizer",
                args.optimizer,
                "--n_workers",
                args.n_workers
            ]
            if args.seed is not None:
                train_cmd.extend(["--seed", args.seed])
            if args.resume:
                train_cmd.append("--resume")
            train_cmd = [str(x) for x in train_cmd]
            check_call(train_cmd)
            
        print("==== RUN_CL ({}/{}): TEST NOVEL ====".format(i, args.n))
        test_cmd = [
            "python3",
            "fewshot/test.py",
            "--model",
            args.model,
            "--method",
            args.method,
            "--n_shot",
            args.n_shot,
            "--test_n_way",
            args.test_n_way,
            "--checkpoint_dir",
            args.cl_dir,
            "--split",
            "novel",
            "--n_workers",
            args.n_workers,
            "--record_file",
            args.cl_record_file,
        ]
        if args.seed is not None:
            test_cmd.extend(["--seed", args.seed])
        test_cmd = [str(x) for x in test_cmd]
        check_call(test_cmd)
