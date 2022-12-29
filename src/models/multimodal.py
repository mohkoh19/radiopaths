import torch
import torch.nn as nn
import torch.nn.functional as F

from util.util import semi_flag
from util.logging import WandbLightningModule
from self_attention_cv import RelPosEmb2D
from util.vision import prepare_backbone
from util.vision import AttentionConv
import torch.nn.functional as F

from .unimodal import ClinoClassifier, CXRClassifier, ClicoClassifier
from .transformer_encoder import CrossAttentionBlock, PositionalEncoding
from util.vision import AttentionConv

class CXR2Classifier(WandbLightningModule):
    def __init__(
        self,
        num_heads=8,
        dropout=0.1,
        dim_feedforward=2048,
        pretrained=True,
        backbone="efficientnet b0",
        cxr_pretrained=None,
        layer_1=0,
        **kwarg,
    ):
        """ Multi-View CXR classifier.
            Feature extractor is an EfficientNet B0 CNN pre-trained on ImageNet
            Either fine-tuned on same dataset & frozen weights or no fine-tuning and no frozen weights.
            Feature output are local features of shape 1280 x 7 x 7
            The high dimensionality might be prhibitive to apply attention
            Instead: Use the penultimate convolutional layer that outputs 320 X 7 X 7 features
            The last convolutional layer in EfficientNet is a 1x1 Conv2D layer, thus we just append it
            In this way we "inject" attention between the CNN feature maps.

        Args:
            num_heads (int, optional): Number of heads for MHA. Defaults to 8.
            dropout (float, optional): Dropout rate in Encoder. Defaults to 0.1.
            dim_feedforward (int, optional): Size of hidden layer in Encoder FFN. Defaults to 2048.
            pretrained (bool, optional): Uses model pre-trained on ImageNet if enabled. Defaults to True.
            backbone (str, optional): _description_. Defaults to "efficientnet b0".
            cxr_pretrained (str, optional): Path to other pre-trained model. Defaults to None.
            layer_1 (int, optional): If not 0, adds a linear layer with the provided size and ReLU activation before the final classification layer. Defaults to 0.
        """
        super().__init__(**kwarg)
        self.save_hyperparameters(ignore=[*kwarg.keys()])

        # Initialize feature extractor
        if self.hparams.cxr_pretrained is not None:
            cxrc = CXRClassifier.load_from_checkpoint(self.hparams.cxr_pretrained)
            cxrc.freeze()
            backbone = cxrc.backbone_model
        else:
            backbone, _ = prepare_backbone(
                self.hparams.backbone,
                self.hparams.pretrained,
                0,
                self.num_classes,
            )

        # Replace Conv before and after cross attention with AttConv
        # Fix number of intermediate features and output features
        conv2 = backbone.features[-2][0].block[1][0]
        self.num_features = backbone.features[-2][0].block[3][0].out_channels
        backbone.features[-2][0].block[1][0] = AttentionConv(
            in_channels=conv2.in_channels,
            out_channels=conv2.out_channels,
            kernel_size=conv2.kernel_size[0],
            stride=conv2.stride[0],
            padding=conv2.padding[0],
            groups=self.hparams.num_heads,
        )

        conv2 = backbone.features[-1][0]
        self.out_features = conv2.out_channels

        self.cxr_xtr = backbone.features[:-1]
        self.upsample = nn.Sequential(backbone.features[-1], nn.AdaptiveAvgPool2d(1), nn.Flatten(1))

        # Unfreeze parameters of last feature layer and upsampling layer if pre-trained
        if self.hparams.cxr_pretrained is not None:
            for param in self.cxr_xtr[-1].parameters():
                param.requires_grad = True
            for param in self.upsample.parameters():
                param.requires_grad = True

        self.seq_length = 49
        self.fmap_size = (7, 7)

        # Cross-Attention is applied in 2 stages like in Liu et al., 2021
        ## We need to mask out all instances in the batch without lateral images
        self.cross_attention = CrossAttentionBlock(
            dim=self.num_features,
            heads=self.hparams.num_heads,
            dim_linear_block=self.hparams.dim_feedforward,
            dropout=self.hparams.dropout,
            activation=nn.SiLU,
            pe=RelPosEmb2D(self.fmap_size, dim_head=(self.num_features // self.hparams.num_heads)),
        )

        self.num_features = self.out_features

        # Classification of global representation
        layers = []

        ## optional intermediate layer
        if self.hparams.layer_1 != 0:
            layers.extend(
                [
                    nn.Linear(self.out_features, self.hparams.layer_1),
                    nn.ReLU(),
                ]
            )
            self.out_features = self.hparams.layer_1

        ## classification layer
        layers.append(nn.Linear(self.out_features, self.num_classes))

        self.classifier = nn.Sequential(*layers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = WandbLightningModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("CXR2Classifier")
        parser.add_argument("--pretrained", type=semi_flag, nargs="?", const=True, default=False)
        parser.add_argument("--backbone", type=str, default="efficientnet b0")
        parser.add_argument("--cxr_pretrained", type=str, default=None)
        parser.add_argument("--layer_1", type=int, default=0)
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=256)
        parser.add_argument("--dropout", type=float, default=0.1)
        return parent_parser

    # Attention map is 1 where lateral image is missing
    def _build_attention_mask(self, x):
        lateral = x["image"]["lateral"]

        mask = torch.zeros(
            (
                lateral.shape[0],
                self.hparams.num_heads,
                self.seq_length,
                self.seq_length,
            ),
            device=self.device,
        )

        missing = (lateral.isnan().sum((1, 2, 3)) != 0).nonzero().flatten().tolist()

        mask[missing] += 1

        return mask.bool()

    def _apply_on_non_missing(self, laterals, non_missing, func):

        lateral_features = func(laterals[non_missing])

        result = torch.zeros(
            (
                laterals.shape[0],
                lateral_features.shape[1],
                lateral_features.shape[2],
                lateral_features.shape[3],
            ),
            device=self.device,
            dtype=lateral_features.dtype,
        )

        result[non_missing] = lateral_features

        return result

    def _extract_features(self, x):
        # Extract frontal features
        frontal_features = self.cxr_xtr(x["image"]["frontal"])

        # Extract lateral features if at least one lateral exists
        laterals = x["image"]["lateral"]
        non_missing = (laterals.isnan().sum((1, 2, 3)) == 0).nonzero().flatten()

        if not non_missing.numel():
            lateral_features = torch.zeros_like(
                frontal_features,
                device=self.device,
                dtype=torch.half,
            )
        else:
            lateral_features = self._apply_on_non_missing(laterals, non_missing, self.cxr_xtr)

        # Sequentialize feature maps
        frontal_features = frontal_features.flatten(2, 3).transpose(1, 2)
        lateral_features = lateral_features.flatten(2, 3).transpose(1, 2)

        return frontal_features, lateral_features

    def forward(self, x):
        # Build attention mask based on missing laterals
        attn_mask = self._build_attention_mask(x)

        # Extract feature maps from CNN and sequentialize them
        frontal_features, lateral_features = self._extract_features(x)

        # Apply cross-attention
        frontal_features = self.cross_attention(X=frontal_features, Y=lateral_features, mask1=attn_mask)

        # Upsample & Pool
        frontal_features = self.upsample(frontal_features.transpose(1, 2).unflatten(2, self.fmap_size))

        # Classify global representation
        out = self.classifier(frontal_features)
        return out


class Early2Fusion(WandbLightningModule):
    def __init__(
        self,
        cxr_pretrained,
        clino_pretrained,
        clico_pretrained,
        cxr_type="single",
        layer_reduce_img=0,
        layer_reduce_clino=0,
        layer_1=1024,
        layer_2=512,
        dropout=0.1,
        **kwarg,
    ):
        """  Our Early Fusion Model using covariates and notes

        Args:
            cxr_pretrained (str): Path to pre-trained CXR-classifier.
            clino_pretrained (str): Path to pre-trained BioBERT model for clinical notes.
            clico_pretrained (str): Path to pre-trained FFN for clinical covariates.
            cxr_type (str, optional): Set to "multi" for multi-view and otherwise to "single". Defaults to "single".
            layer_reduce_img (int, optional): Optional linear layer of provided size with ReLU activation that reduced the CXR feature size. Defaults to 0.
            layer_reduce_clino (int, optional): Optional linear layer of provided size with ReLU activation that reduced the clinical notes feature size. Defaults to 0.
            layer_1 (int, optional): If not set to 0, creates a linear layer of provided size with ReLU activation before the classification layer. Defaults to 1024.
            layer_2 (int, optional): If not set to 0, creates another linear layer of provided size with ReLU activation before the classification layer. Defaults to 512.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        self.save_hyperparameters(ignore=[*kwarg.keys()])
        super().__init__(**kwarg)

        # Load pretrained CXRClassifier and "drop" last layer
        if self.hparams.cxr_type == "single":
            self.cxr_backbone = CXRClassifier.load_from_checkpoint(self.hparams.cxr_pretrained)
            self.cxr_backbone.backbone_model.classifier = nn.Identity()
        elif self.hparams.cxr_type == "multi":
            self.cxr_backbone = CXR2Classifier.load_from_checkpoint(self.hparams.cxr_pretrained)
            self.cxr_backbone.classifier = nn.Identity()

        self.num_img_features = self.cxr_backbone.num_features
        self.cxr_backbone.freeze()

        # Determine if image features should be reduced before fusion
        if self.hparams.layer_reduce_img != 0:
            self.layer_reduce_img = nn.Sequential(
                nn.Linear(self.num_img_features, self.hparams.layer_reduce_img),
                nn.Dropout(self.hparams.dropout),
                nn.ReLU(inplace=True),
            )
            self.num_img_features = self.hparams.layer_reduce_img
        else:
            self.layer_reduce_img = nn.Identity()

        # Load pretrained ClinoClassifier and "drop" last layer
        # TODO: retrain clino for new implementation with num_classes and target_idx...
        self.clino_backbone = ClinoClassifier.load_from_checkpoint(
            self.hparams.clino_pretrained, num_classes=14, target_idx=0
        )
        self.clino_backbone.classifier = nn.Identity()
        self.clino_backbone.freeze()

        # Determine if pooler output features should be reduced before fusion
        self.num_clino_features = self.clino_backbone.backbone_model.pooler.dense.out_features
        if self.hparams.layer_reduce_clino != 0:
            self.layer_reduce_clino = nn.Sequential(
                nn.Linear(self.num_clino_features, self.hparams.layer_reduce_clino),
                nn.Dropout(self.hparams.dropout),
                nn.ReLU(inplace=True),
            )
            self.num_clino_features = self.hparams.layer_reduce_clino
        else:
            self.layer_reduce_clino = nn.Identity()

        # Use pretrained Clico Classifier
        self.clico_backbone = ClicoClassifier.load_from_checkpoint(self.hparams.clico_pretrained)
        self.clico_backbone.classifier = nn.Identity()
        self.clico_backbone.freeze()
        self.num_clico_features = self.clico_backbone.num_features

        self.num_fusion_features = self.num_img_features + self.num_clico_features + self.num_clino_features

        # Determine how fusion features should be reduced before classification
        if self.hparams.layer_1 != 0:
            self.layer_1 = nn.Sequential(
                nn.Linear(self.num_fusion_features, self.hparams.layer_1),
                nn.Dropout(self.hparams.dropout),
                nn.ReLU(inplace=True),
            )
            self.num_fusion_features = self.hparams.layer_1
        else:
            self.layer_1 = nn.Identity()

        if self.hparams.layer_2 != 0:
            self.layer_2 = nn.Sequential(
                nn.Linear(self.num_fusion_features, self.hparams.layer_2),
                nn.Dropout(self.hparams.dropout),
                nn.ReLU(inplace=True),
            )
            self.num_fusion_features = self.hparams.layer_2
        else:
            self.layer_2 = nn.Identity()

        self.classifier = nn.Linear(self.num_fusion_features, self.num_classes)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = WandbLightningModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Early2Fusion")
        parser.add_argument("--cxr_pretrained", type=str, required=True)
        parser.add_argument("--clino_pretrained", type=str, required=True)
        parser.add_argument("--clico_pretrained", type=str, required=True)
        parser.add_argument("--cxr_type", type=str, default="single")
        parser.add_argument("--layer_reduce_img", type=int, default=0)
        parser.add_argument("--layer_reduce_clino", type=int, default=0)
        parser.add_argument("--layer_1", type=int, default=1024)
        parser.add_argument("--layer_2", type=int, default=512)
        parser.add_argument("--dropout", type=float, default=0.1)
        return parent_parser

    def forward(self, x):
        img_features = self.cxr_backbone(x)
        clico_features = self.clico_backbone(x)
        clino_features = self.clino_backbone(x)

        # Reduce image features
        img_features = self.layer_reduce_img(img_features)

        # Reduce clino features
        clino_features = self.layer_reduce_clino(clino_features)

        # Concatenate reduced image features with clinical covariates and notes
        x_fusion = torch.cat((img_features, clico_features, clino_features), dim=1).float()

        # Further reduce through layer_1 and layer_2
        x_fusion = self.layer_1(x_fusion)
        x_fusion = self.layer_2(x_fusion)

        # Classify multimodal features
        y_hat_logits = self.classifier(x_fusion)

        return y_hat_logits


class Radiopaths(WandbLightningModule):
    def __init__(
        self,
        cxr_pretrained,
        clino_pretrained,
        clico_pretrained,
        cxr_type="single",
        dim=512,
        num_heads=4,
        dropout=0.1,
        layer_1=0,
        **kwarg,
    ):
        """ Our Radiopaths model with attention-based fusion.

        Args:
            cxr_pretrained (str): Path to pre-trained CXR-classifier.
            clino_pretrained (str): Path to pre-trained BioBERT model for clinical notes.
            clico_pretrained (str): Path to pre-trained FFN for clinical covariates.
            cxr_type (str, optional): Set to "multi" for multi-view and otherwise to "single". Defaults to "single".
            dim (int, optional): Hidden dimension. Defaults to 512.
            num_heads (int, optional): Number of heads for MHA. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            layer_1 (int, optional): If not set to 0, creates a linear layer of provided size with ReLU activation before the classification layer. Defaults to 0.
        """
        self.save_hyperparameters(ignore=[*kwarg.keys()])
        super().__init__(**kwarg)

        # Load pretrained CXRClassifier and "drop" last layer
        if self.hparams.cxr_type == "single":
            self.cxr_backbone = CXRClassifier.load_from_checkpoint(self.hparams.cxr_pretrained)
            self.cxr_backbone.backbone_model.classifier = nn.Identity()
        elif self.hparams.cxr_type == "multi":
            self.cxr_backbone = CXR2Classifier.load_from_checkpoint(self.hparams.cxr_pretrained)
            self.cxr_backbone.classifier = nn.Identity()

        self.num_img_features = self.cxr_backbone.num_features

        self.cxr_backbone.freeze()
        self.cxr_proj = nn.Linear(self.num_img_features, self.hparams.dim, bias=False)

        # Load pretrained ClinoClassifier and "drop" last layer
        # TODO: retrain clino for new implementation with num_classes and target_idx...
        self.clino_backbone = ClinoClassifier.load_from_checkpoint(
            self.hparams.clino_pretrained, num_classes=14, target_idx=0
        )
        self.clino_backbone.classifier = nn.Identity()
        self.num_clino_features = self.clino_backbone.backbone_model.pooler.dense.out_features
        self.clino_backbone.freeze()
        self.clino_proj = nn.Linear(self.num_clino_features, self.hparams.dim, bias=False)

        # Use pretrained Clico Classifier
        self.clico_backbone = ClicoClassifier.load_from_checkpoint(self.hparams.clico_pretrained)
        self.clico_backbone.classifier = nn.Identity()
        self.clico_backbone.freeze()
        self.num_clico_features = self.clico_backbone.num_features
        self.clico_proj = nn.Linear(self.num_clico_features, self.hparams.dim, bias=False)

        self.pe = PositionalEncoding(self.hparams.dim, 3)

        self.mha = nn.MultiheadAttention(
            embed_dim=self.hparams.dim, dropout=self.hparams.dropout, num_heads=self.hparams.num_heads, batch_first=True
        )

        # Classification of global representation
        layers = []
        self.out_features = self.hparams.dim  # * 3

        ## optional intermediate layer
        if self.hparams.layer_1 != 0:
            layers.extend(
                [
                    nn.Linear(self.out_features, self.hparams.layer_1),
                    nn.ReLU(),
                ]
            )
            self.out_features = self.hparams.layer_1

        ## classification layer
        layers.append(nn.Linear(self.out_features, self.num_classes))

        self.classifier = nn.Sequential(*layers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = WandbLightningModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Early2Fusion")
        parser.add_argument("--cxr_pretrained", type=str, required=True)
        parser.add_argument("--clino_pretrained", type=str, required=True)
        parser.add_argument("--clico_pretrained", type=str, required=True)
        parser.add_argument("--cxr_type", type=str, default="single")
        parser.add_argument("--dim", type=int, default=512)
        parser.add_argument("--num_heads", type=int, default=512)
        parser.add_argument("--layer_1", type=int, default=1024)
        parser.add_argument("--dropout", type=float, default=0.1)
        return parent_parser

    def _build_attention_mask(self, x):
        clico = x["clico"]
        clino = x["clino"]

        l1 = (clico.sum(1) == 0).nonzero().flatten().tolist()
        l2 = (clino["input_ids"].sum(1) == 0).nonzero().flatten().tolist()

        m0 = torch.zeros(
            (clico.shape[0], 3, 3),
            dtype=torch.int16,
            device=self.device,
        )

        m0[l1, 1, :] += 1
        m0[l2, 2, :] += 1

        m0 = m0.transpose(1, 2).unsqueeze(1).repeat(1, self.hparams.num_heads, 1, 1).bool()

        return m0

    def forward(self, x):
        # Extract and project features
        img_features = self.cxr_proj(self.cxr_backbone(x)).unsqueeze(1)
        clico_features = self.clico_proj(self.clico_backbone(x)).unsqueeze(1)
        clino_features = self.clino_proj(self.clino_backbone(x)).unsqueeze(1)

        # Tokenize
        tokens = self.pe(torch.cat((img_features, clico_features, clino_features), dim=1))

        # Masking, Self-Attention, and Encoding
        tokens = self.mha(query=tokens, key=tokens, value=tokens)[0] + tokens

        # Cat tokens to fused representation
        fused = F.adaptive_avg_pool1d(tokens.transpose(1, 2), 1).squeeze()

        # Classification
        out = self.classifier(fused)

        return out
