from turtle import back
from numpy import require
import torch
import torch.nn as nn
import torch.nn.functional as TF
from transformers import AutoModel

# from models.transformer_encoder import TransformerBlock
from self_attention_cv import RelPosEmb2D
from einops import rearrange
from util.vision import AugmentedConv, AttentionConv

from util.util import semi_flag
from util.logging import WandbLightningModule
from util.vision import prepare_backbone

class CXRClassifier(WandbLightningModule):
    def __init__(
        self,
        pretrained=True,
        backbone="efficientnet b0",
        layer_1=0,
        **kwarg,
    ):
        """ CXR Classifier

        Args:
            pretrained (bool, optional): Uses model pretrained on ImageNet if enabled. Defaults to True.
            backbone (str, optional): Choice between multiple variations of DenseNet and EfficientNet (see util.vision for more info). Defaults to "efficientnet b0".
            layer_1 (int, optional): If not set to 0, creates a linear layer of provided size with ReLU activation before the classification layer. Defaults to 0.
        """
        super().__init__(**kwarg)
        self.save_hyperparameters(ignore=[*kwarg.keys()])

        self.backbone_model, self.num_features = prepare_backbone(
            self.hparams.backbone,
            self.hparams.pretrained,
            self.hparams.layer_1,
            self.num_classes,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = WandbLightningModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("CXRClassifier")
        parser.add_argument("--pretrained", type=semi_flag, nargs="?", const=True, default=False)
        parser.add_argument("--backbone", type=str, default="efficientnet b0")
        parser.add_argument("--layer_1", type=int, default=0)
        return parent_parser

    def forward(self, x):
        if type(x) is dict:
            x = x["image"]

        return self.backbone_model(x)


class ClinoClassifier(WandbLightningModule):
    def __init__(
        self,
        pretrained="dmis-lab/biobert-v1.1",
        **kwarg,
    ):
        """ BioBERT classifier for clinical notes

        Args:
            pretrained (str, optional): Model name from Hugging Face. Defaults to "dmis-lab/biobert-v1.1".
        """
        super().__init__(**kwarg)
        self.save_hyperparameters(ignore=[*kwarg.keys()])

        self.backbone_model = AutoModel.from_pretrained(
            self.hparams.pretrained, output_hidden_states=True, output_attentions=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.backbone_model.pooler.dense.out_features, self.num_classes),
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = WandbLightningModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("ClinoClassifier")
        parser.add_argument("--pretrained", type=str, default="dmis-lab/biobert-v1.1")
        return parent_parser

    def forward(self, x):
        if type(x) is dict and "clino" in x.keys():
            x = x["clino"]

        x = self.backbone_model(**x).pooler_output
        return self.classifier(x)


class ClicoClassifier(WandbLightningModule):
    def __init__(
        self,
        num_features=2,
        layer_1=64,
        droprate=0.0,
        **kwarg,
    ):
        """ FFN for classifying clinical covariates

        Args:
            num_features (int, optional): Number of covariates. Defaults to 2.
            layer_1 (int, optional): If not set to 0, creates a linear layer of provided size with ReLU activation before the classification layer. Defaults to 64.
            droprate (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__(**kwarg)
        self.save_hyperparameters(ignore=[*kwarg.keys()])

        self.num_features = num_features

        if self.hparams.layer_1 != 0:
            self.features = nn.Sequential(
                nn.Linear(self.num_features, self.hparams.layer_1),
                nn.Dropout(self.hparams.droprate),
                nn.ReLU(inplace=True),
            )
            self.num_features = self.hparams.layer_1
        else:
            self.features = nn.Identity()

        self.classifier = nn.Linear(self.num_features, self.num_classes)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = WandbLightningModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("ClicoClassifier")
        parser.add_argument("--num_features", type=int, default=2)
        parser.add_argument("--layer_1", type=int, default=64)
        parser.add_argument("--droprate", type=float, default=0.0)
        return parent_parser

    def forward(self, x):
        if type(x) is dict:
            x = torch.cat((x["clico"], x["demog"]), dim=1)

        x = x.nan_to_num(0)
        x = self.features(x)
        return self.classifier(x)
