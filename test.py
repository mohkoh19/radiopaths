# ----------------
# test.py
# ----------------
import traceback
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
from datasets.mimic_cxr import MIMIC_CXR_Loader
from models.multimodal import CXR2Classifier, Early2Fusion, Radiopaths
from models.unimodal import ClicoClassifier, ClinoClassifier, CXRClassifier
from util.logging import WandbLightningModule


def train(args):
    dm = MIMIC_CXR_Loader.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()

    if args.model_name == "cxr":
        model = CXRClassifier.load_from_checkpoint(
            args.ckpt_path,
            num_classes=dm.num_classes,
            target_idx=dm.target_idx,
        )
    elif args.model_name == "cxr2":
        model = CXR2Classifier.load_from_checkpoint(args.ckpt_path)
    elif args.model_name == "radiopaths":
        model = Radiopaths.load_from_checkpoint(args.ckpt_path)
    elif args.model_name == "clino":
        model = ClinoClassifier.load_from_checkpoint(
            args.ckpt_path,
            num_classes=dm.num_classes,
            target_idx=dm.target_idx,
        )
    elif args.model_name == "clico":
        model = ClicoClassifier.load_from_checkpoint(
            args.ckpt_path, num_classes=dm.num_classes, target_idx=dm.target_idx
        )
    elif args.model_name == "early2":
        model = Early2Fusion.load_from_checkpoint(args.ckpt_path, num_classes=dm.num_classes, target_idx=dm.target_idx)

    if args.run_id is not None:
        wandb_logger = WandbLogger(project="radiopaths", id=args.run_id, save_dir=args.save_dir, resume=False)
    else:
        wandb_logger = WandbLogger(
            project="radiopaths",
            name=args.exp_name,
            save_dir=args.save_dir,
            resume=False,
        )

    trainer = Trainer.from_argparse_args(args, logger=wandb_logger)

    try:
        trainer.test(model=model, dataloaders=dm)
        wandb.finish(0)
    except KeyboardInterrupt as ki:
        print(ki)
        wandb.finish(-1)
    except Exception:
        traceback.print_exc()
        wandb.finish(-1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument("--model_name", type=str, default="cxr", help="cxr, clino, clico, early2, radiopaths")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./",
        help="Directory to save logs and checkpoints",
    )
    parser.add_argument("--exp_name", type=str, default="default-experiment", help="Name of run")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint of model")
    parser.add_argument("--run_id", type=str, default=None, help="Wandb run id to resume run")

    parser = WandbLightningModule.add_model_specific_args(parser)

    # let the datamodule add hyperparameters
    parser = MIMIC_CXR_Loader.add_argparse_args(parser)

    args = parser.parse_args()

    # train
    train(args)
