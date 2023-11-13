import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms.functional as TF
from torchvision import models, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)



def prepare_backbone(backbone, pretrained, layer_1, num_classes, dropout=0.0):
    """ Creates a CNN based on the given configuration.

    Args:
        backbone (str): Choice between "efficientnet" and "densenet" variations (see below). 
                        Keep in mind that all the muldimodal modals are based on EfficientNet
        pretrained (bool): If enabled, uses model pre-trained on ImageNet.
        layer_1 (int): If not set to 0, creates a linear layer of provided size with ReLU activation before the classification layer.
        num_classes (int): Number of output classes.
        dropout (float, optional): Dropout rate. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    backbone = backbone.split()
    if backbone[0] == "efficientnet":
        if backbone[1] == "b0":  ##because of a current hash error use this
          #WeightsEnum.get_state_dict = get_state_dict
          #model=efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
          #efficientnet_b0(weights="DEFAULT")
          model = models.efficientnet_b0(pretrained)
        elif backbone[1] == "b1":
            model = models.efficientnet_b1(pretrained)
        elif backbone[1] == "b2":
            model = models.efficientnet_b2(pretrained)
        elif backbone[1] == "b3":
            model = models.efficientnet_b3(pretrained)
        elif backbone[1] == "b4":
            model = models.efficientnet_b4(pretrained)

        num_features = model.classifier[1].in_features
        if layer_1 == 0:
            model.classifier[1] = nn.Linear(num_features, num_classes)
        else:
            model.classifier[1] = nn.Sequential(
                nn.Linear(num_features, layer_1),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.Linear(layer_1, num_classes),
            )

    # All the muldimodal modals are based on EfficientNet!!! 
    # Using densent might break the multimodal models.
    elif backbone[0] == "densenet":
        if backbone[1] == "121":
            model = models.densenet121(pretrained)
        elif backbone[1] == "161":
            model = models.densenet161(pretrained)
        elif backbone[1] == "169":
            model = models.densenet169(pretrained)
        elif backbone[1] == "201":
            model = models.densenet201(pretrained)

        num_features = model.classifier.in_features
        if layer_1 == 0:
            model.classifier = nn.Linear(num_features, num_classes)
        else:
            model.classifier = nn.Sequential(
                nn.Linear(num_features, layer_1),
                nn.Dropout(dropout),
                nn.ReLU(inplace=False),
                nn.Linear(layer_1, num_classes),
            )

    return model, num_features


def transform_images(batch, train, mean, std, gamma):
    """ Pre-processing for images. Includes augmentation and other transformations.

    Args:
        batch (Tensor): Batch of images to apply transformations on.
        train (bool): If enabled, applies augmentations.
        mean (list): Per-channel means to normalize with.
        std (list): Per-channel standard deviations to normalze with.
        gamma (float): Gamma value for gamma compression.

    Returns:
        Tensor: Batch of transformed images.
    """
    missing = (batch.sum((1, 2, 3)) == 0).nonzero().flatten().tolist()
    batch = TF.adjust_gamma(batch, gamma=gamma)

    transform_list = []

    if train:
        transform_list += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=(-45, +45), translate=(0.15, 0.15), scale=(0.9, 1.1)),
        ]

    transform_list += [transforms.Normalize(mean, std)]

    transform = transforms.Compose(transform_list)

    batch = transform(batch)

    if missing:
        batch[missing] = torch.zeros(batch.shape[1:], device=batch.device)

    return batch


# Source: https://github.com/leaderj1001/Attention-Augmented-Conv2d
class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(
            self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride, padding=self.padding
        )

        self.qkv_conv = nn.Conv2d(
            self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride, padding=self.padding
        )

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = self.softmax(logits)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        scaled_dot_product = dkh**-0.5
        q = scaled_dot_product * q
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum("bhxyd,md->bhxym", q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1 :]
        return final_x

# Source: https://github.com/leaderj1001/Attention-Augmented-Conv2d
class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert (
            self.out_channels % self.groups == 0
        ), "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum("bnchwk,bnchwk -> bnchw", out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.value_conv.weight, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.query_conv.weight, mode="fan_out", nonlinearity="relu")

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

# Source: https://github.com/leaderj1001/Attention-Augmented-Conv2d
class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert (
            self.out_channels % self.groups == 0
        ), "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)]
        )

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum("mc,ca->ma", self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum("mc,cb->mb", self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum("bnchwk,bnchwk->bnchw", out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.query_conv.weight, mode="fan_out", nonlinearity="relu")
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode="fan_out", nonlinearity="relu")

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)
