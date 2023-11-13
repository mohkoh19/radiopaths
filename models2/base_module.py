from math import sqrt

import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from transformers import get_cosine_schedule_with_warmup

from util.util import semi_flag


class BinaryFocalLossWithLogits(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, pos_weight=None):
        """ Focal loss as in https://arxiv.org/abs/1708.02002

        Args:
            weight (float, optional): Class weight (alpha parameter in paper). Defaults to None.
            gamma (int, optional): _description_. Defaults to 2.
            pos_weight (float, optional): Per-class skew (beta parameter in paper). Defaults to None.
        """
        super().__init__()
        self.gamma = gamma

        # weight parameter will act as the alpha parameter to balance class weights
        self.weight = weight

        self.pos_weight = pos_weight

    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy_with_logits(
            input, target, reduction="none", weight=self.weight, pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma * bce_loss).mean()

        return focal_loss


class BaseModule(LightningModule):
    def __init__(
        self,
        num_classes,
        target_idx,
        # TRAINING
        learning_rate=1e-3,
        weight_decay=1e-2,
        # OPTIMIZER
        optimizer="adamw",
        amsgrad=False,
        nesterov=False,
        momentum=0.0,
        dampening=0.0,
        # SCHEDULER
        scheduler="plateau",
        scheduler_factor=0.1,
        scheduler_patience=2,
        scheduler_warmup_steps=2000,
        optim_metric="val/loss",
        optim_mode="min",
        # LOSS
        loss_fn="focal",
        gamma=2.0,
        pos_weight=False,
        target_focus=False,
        **kwarg,
    ):
        """ Super class for all our models. 
            All parameters related to optimization can be configured here

        Args:
            num_classes (int): Number of label classes.
            target_idx (int): Index of target label.
            learning_rate (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 1e-2.
            optimizer (str, optional): Optimizer ("adam", "adamw", or "sgd"). Defaults to "adamw".
            amsgrad (bool, optional): Enables the AMSGrad extension for the Adam optimizer. Defaults to False.
            nesterov (bool, optional): Enables the Nesterov extension for the SGD optmizer. Defaults to False.
            momentum (float, optional): Momentum for the SGD optimizer. Defaults to 0.
            dampening (float, optional): Dampening for the SGD optimizer. Defaults to 0.
            scheduler (str, optional): Learning rate scheduler ("plateau", "linear", or "cosine_warmup"). Defaults to "plateau".
            scheduler_factor (float, optional): Scheduling factor for the plateau scheduler. Defaults to 0.1.
            scheduler_patience (int, optional): Scheduling patience for the plateau scheduler. Defaults to 2.
            scheduler_warmup_steps (int, optional): Number of warmup steps incase the cosine_warmup scheduler is chosed. Defaults to 2000.
            optim_metric (str, optional): Target optimization metric name for the scheduler. Defaults to "val/loss".
            optim_mode (str, optional): Optimization target ("min" or "max"). Defaults to "min".
            loss_fn (str, optional): Loss function ("focal" or "bce"). Defaults to "focal".
            gamma (float, optional): Gamma parameter for the focal loss. Defaults to 2.0.
            pos_weight (bool, optional): If enabled, weights the loss with the per-class skew (beta parameter in paper). Defaults to False.
            target_focus (bool, optional): If enabled, sets alpha parameter of the target class to 1 in the loss. Defaults to False.
        """
        super().__init__()
        self.save_hyperparameters(ignore=[*kwarg.keys()])

        # Labels and number of tasks/classes
        self.num_classes = self.hparams.num_classes

        # Index of the target
        self.target_idx = self.hparams.target_idx

        # Loss Function
        if self.hparams.loss_fn == "focal":
            self.loss = BinaryFocalLossWithLogits(gamma=self.hparams.gamma)
        elif self.hparams.loss_fn == "bce":
            self.loss = nn.BCEWithLogitsLoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseModule")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-2)
        parser.add_argument("--optimizer", type=str, default="adamw")
        parser.add_argument("--amsgrad", type=semi_flag, nargs="?", const=True, default=False)
        parser.add_argument("--nesterov", type=semi_flag, nargs="?", const=True, default=False)
        parser.add_argument("--momentum", type=float, default=0.0)
        parser.add_argument("--dampening", type=float, default=0.0)
        parser.add_argument("--scheduler", type=str, default="plateau")
        parser.add_argument("--scheduler_factor", type=float, default=0.1)
        parser.add_argument("--scheduler_patience", type=int, default=2)
        parser.add_argument("--scheduler_warmup_steps", type=int, default=3000)
        parser.add_argument("--optim_metric", type=str, default="val/loss")
        parser.add_argument("--optim_mode", type=str, default="min")
        parser.add_argument("--loss_fn", type=str, default="focal")
        parser.add_argument("--gamma", type=float, default=2.0)
        parser.add_argument("--pos_weight", type=semi_flag, nargs="?", const=True, default=False)
        parser.add_argument("--target_focus", type=semi_flag, nargs="?", const=True, default=False)
        return parent_parser

    def runtime_init(self):
        # Move all tensors to current device
        # Init/modify variables that need information from DataModule

        # Each task is balanced based on the frequency of that task in the dataset
        # https://arxiv.org/pdf/2002.02497v2.pdf
        self.class_weight = self.trainer.datamodule.get_class_weight()

        # If target focus is enabled, give the target the same weight as the sum of all other tasks
        if self.hparams.target_focus:
            self.class_weight[self.target_idx] = sqrt(sum(self.class_weight) - self.class_weight[self.target_idx])

        self.class_weight = torch.tensor(self.class_weight, device=self.device)

        # If positive weight is enabled, false positives get higher weight
        # according to the per class imbalance
        if self.hparams.pos_weight:
            self.trainer.datamodule.get_pos_weight()
            self.pos_weight = self.trainer.datamodule.get_pos_weight()
            self.pos_weight = torch.tensor(self.pos_weight, device=self.device)
        else:
            self.pos_weight = None

    def on_fit_start(self):
        self.runtime_init()

    def on_test_start(self):
        self.runtime_init()

    def get_loss_fn(self, stage):
        if stage != "train":
            self.loss.pos_weight = None
            self.loss.weight = None
        else:
            self.loss.pos_weight = self.pos_weight
            self.loss.weight = self.class_weight

        return self.loss

    def _step(self, batch, stage):
        x, y = batch

        logits = self(x)

        loss_fn = self.get_loss_fn(stage=stage)

        loss = loss_fn(logits, y)

        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "logits": logits.detach(), "y": y.detach()}

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, stage="test")

    def get_optimizer(self):
        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                amsgrad=self.hparams.amsgrad,
            )
        elif self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                amsgrad=self.hparams.amsgrad,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=self.hparams.momentum,
                dampening=self.hparams.dampening,
                nesterov=self.hparams.nesterov,
            )

        return optimizer

    @property
    def num_training_steps(self):
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def get_scheduler(self, optimizer):
        if self.hparams.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                self.hparams.optim_mode,
                patience=self.hparams.scheduler_patience,
                verbose=True,
                factor=self.hparams.scheduler_factor,
            )
        elif self.hparams.scheduler == "cosine_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, self.hparams.scheduler_warmup_steps, self.num_training_steps
            )
        elif self.hparams.scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=7)
        else:
            scheduler = None

        return scheduler

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        metric = self.hparams.optim_metric

        if scheduler is None:
            return {"optimizer": optimizer}
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": metric,
            }
