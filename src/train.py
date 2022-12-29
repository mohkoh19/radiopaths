# ----------------
# train.py
# ----------------
import os
import traceback
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import wandb
from datasets.mimic_cxr import MIMIC_CXR_Loader
from models.multimodal import CXR2Classifier, Early2Fusion, Radiopaths
from models.unimodal import ClicoClassifier, ClinoClassifier, CXRClassifier


def train(args):
    ckpt_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    dm = MIMIC_CXR_Loader.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()

    if args.model_name == "cxr":
        model = CXRClassifier(num_classes=dm.num_classes, target_idx=dm.target_idx, **vars(args))
    elif args.model_name == "cxr2":
        model = CXR2Classifier(num_classes=dm.num_classes, target_idx=dm.target_idx, **vars(args))
    elif args.model_name == "radiopaths":
        model = Radiopaths(num_classes=dm.num_classes, target_idx=dm.target_idx, **vars(args))
    elif args.model_name == "clino":
        model = ClinoClassifier(num_classes=dm.num_classes, target_idx=dm.target_idx, **vars(args))
    elif args.model_name == "clico":
        model = ClicoClassifier(num_classes=dm.num_classes, target_idx=dm.target_idx, **vars(args))
    elif args.model_name == "early2":
        model = Early2Fusion(num_classes=dm.num_classes, target_idx=dm.target_idx, **vars(args))

    wandb_logger = WandbLogger(
        project="radiopaths",
        log_model="all",
        name=args.exp_name,
        save_dir=args.save_dir,
        id=None if args.run_id is None else args.run_id,
        resume=False if args.run_id is None else "must",
    )

    optim_metric = args.optim_metric
    optim_mode = args.optim_mode

    checkpoint_callback = ModelCheckpoint(
        monitor=optim_metric,
        save_top_k=5,
        mode=optim_mode,
        dirpath=ckpt_dir,
        filename=args.exp_name
        + "-epoch={epoch}-"
        + args.optim_metric.replace("/", "_")
        + "={"
        + optim_metric
        + ":.2f}",
        auto_insert_metric_name=False,
    )

    early_stop_callback = EarlyStopping(
        monitor=optim_metric,
        min_delta=1e-4,
        patience=args.early_stop_patience,
        verbose=True,
        mode=optim_mode,
    )

    trainer = Trainer.from_argparse_args(
        args, logger=wandb_logger, callbacks=[checkpoint_callback, early_stop_callback]
    )

    wandb_logger.watch(model, log="all", log_graph=False)

    try:
        trainer.fit(model, dm, ckpt_path=None if args.ckpt_path is None else args.ckpt_path)
        wandb_logger.finalize("success")
        wandb.finish(0)
    except KeyboardInterrupt as ki:
        print(ki)
        wandb_logger.finalize("abort")
        wandb.finish(-1)
    except Exception:
        traceback.print_exc()
        wandb_logger.finalize("error")
        wandb.finish(-1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument(
        "--model_name",
        type=str,
        default="cxr",
        help="cxr, clino, clico, early2, radiopaths",
    )
    parser.add_argument("--exp_name", type=str, default=None, help="Name of run")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./",
        help="Directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    parser.add_argument("--run_id", type=str, default=None, help="Wandb run id to resume run")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint of model to resume",
    )

    # This line pulls the model's name
    temp_args, _ = parser.parse_known_args()

    # let the model add hyperparameters
    if temp_args.model_name == "cxr":
        parser = CXRClassifier.add_model_specific_args(parser)
    elif temp_args.model_name == "cxr2":
        parser = CXR2Classifier.add_model_specific_args(parser)
    elif temp_args.model_name == "radiopaths":
        parser = Radiopaths.add_model_specific_args(parser)
    elif temp_args.model_name == "clino":
        parser = ClinoClassifier.add_model_specific_args(parser)
    elif temp_args.model_name == "clico":
        parser = ClicoClassifier.add_model_specific_args(parser)
    elif temp_args.model_name == "early2":
        parser = Early2Fusion.add_model_specific_args(parser)

    # let the datamodule add hyperparameters
    parser = MIMIC_CXR_Loader.add_argparse_args(parser)

    args = parser.parse_args()

    # train
    train(args)
