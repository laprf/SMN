import math

import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import Conv2d

from HSI_dataset import img_size
from models.NAT2D import NeighborhoodAttention2D as NAT
from models.PVTV2 import pvt_v2_b1
from models.resnet_list import resnet18
from models.swin_transformer import swin_t, Swin_T_Weights

image_size = img_size()


class EdgeDetectionModule(nn.Module):
    def __init__(self, in_dim=256, out_dim=256, size=88):
        super(EdgeDetectionModule, self).__init__()
        self.size = size
        self.edge_conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.edge_out = nn.Conv2d(out_dim, 1, 3, 1, 1)
        self.edge_mask = nn.Sequential(
            nn.AdaptiveAvgPool2d(size),
            nn.Conv2d(out_dim, out_dim, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_dim, out_dim, 1),
            nn.Sigmoid(),
        )
        self.conv_edge_enhance = nn.Conv2d(
            out_dim, out_dim, kernel_size=3, stride=1, padding=1
        )
        self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        edge_feat = self.edge_conv(x)
        edge_feat_mask = self.edge_mask(edge_feat)
        edge_feat = torch.mul(edge_feat, edge_feat_mask) + self.conv_edge_enhance(edge_feat)

        edge_feat = self.bn(edge_feat)
        edge_feat = F.relu(edge_feat, inplace=False)

        edge_out = self.edge_out(edge_feat)
        return edge_feat, edge_out


class MixedFrequencyAttention(nn.Module):
    def __init__(self, config):
        super(MixedFrequencyAttention, self).__init__()
        hidden_size = config.hidden_size
        sa_kernel = config.NAT.kernel_size_sa
        kernel = config.NAT.kernel_size_ca
        heads = config.NAT.num_heads

        # Self Attention (SA) and Cross Attention (CA) using NAT (Neighborhood Attention)
        self.SA = NAT(hidden_size // 2, kernel_size=sa_kernel, num_heads=heads)
        self.CA = NAT(hidden_size // 2, kernel_size=kernel, num_heads=heads)
        # Convolutional layer for frequency convergence
        self.converge = nn.Sequential(Conv2d(hidden_size, hidden_size, 1, 1, 0), nn.BatchNorm2d(hidden_size), nn.ReLU())

    def forward(self, sal_feat, edge_feat):
        """
        Args:
            sal_feat (Tensor): Low-frequency spectral saliency features.
            edge_feat (Tensor): High-frequency edge features.

        Returns:
            Tensor: Refined feature map after applying mixed frequency attention.
        """
        h = int(math.sqrt(sal_feat.shape[1]))
        c = sal_feat.shape[-1]

        # NAT requires the input to be in the shape of [b, h, w, c]
        sal_feat = rearrange(sal_feat, "b (h w) c -> b h w c", h=h)
        edge_feat_ref = rearrange(edge_feat, 'b c h w -> b h w c')

        # Apply mixed-frequncy attenion
        attn_sa = rearrange(self.SA(sal_feat[:, :, :, :c // 2], sal_feat[:, :, :, :c // 2]), 'b h w c -> b c h w', h=h)
        attn_ca = rearrange(self.CA(edge_feat_ref, sal_feat[:, :, :, c // 2:]), 'b h w c -> b c h w', h=h)

        # Frequency convergence
        refine_feat = torch.cat((attn_sa, attn_ca), dim=1)
        refine_feat = self.converge(refine_feat)
        return refine_feat


class SpecEmbedding(nn.Module):
    def __init__(self, config, in_channels=3, backbone="resnet18"):
        super(SpecEmbedding, self).__init__()
        if backbone == "resnet18":
            self.hybrid_model = resnet18(in_channels=in_channels, pretrained=True)
            in_channels = 256
        elif backbone == "swin_t":
            self.hybrid_model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
            in_channels = 384
        elif backbone == "pvt_v2_b1":
            self.hybrid_model = pvt_v2_b1(pre_trained_path="models/pretrained_models/pvt_v2_b1.pth")
            in_channels = 320
        else:
            raise ValueError("Unsupported backbone model")

        # Convolutional layer for patch embeddings
        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,  # 768, 1024
            kernel_size=(1, 1),  # (1,1)
            stride=(1, 1),
        )
        # Learnable position embeddings
        self.position_embeddings = nn.Parameter(
            torch.randn(1, (config.image_size // 16) * (config.image_size // 16), config.hidden_size)
        )

    def forward(self, x):
        res_feat, x = self.hybrid_model(x)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = self.position_embeddings + x
        return embeddings, res_feat


class SalEdgeAwareDecoder(nn.Module):
    """
    Saliency-edge-aware Decoder.

    Args:
        embed_dim (int): Embedding dimension.
        hidden_size (int): Hidden size for convolutional layers.
        size (int): Size of the input image.
        backbone (str): Name of the backbone model to be used.
    """

    def __init__(self, embed_dim=1024, hidden_size=256, size=352, backbone='resnet18'):
        super(SalEdgeAwareDecoder, self).__init__()
        self.norm = nn.LayerNorm([embed_dim, size // 16, size // 16], eps=1e-6)

        self.conv_0 = nn.Conv2d(embed_dim, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(hidden_size + 256, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(hidden_size, 1, kernel_size=3, stride=1, padding=1)

        self.syncbn_fc_0 = nn.BatchNorm2d(hidden_size)
        self.syncbn_fc_1 = nn.BatchNorm2d(hidden_size)
        self.syncbn_fc_2 = nn.BatchNorm2d(hidden_size)
        self.syncbn_fc_3 = nn.BatchNorm2d(hidden_size)

        if backbone == 'resnet18' or backbone == 'pvt_v2_b1':
            self.up_conv_1 = nn.ConvTranspose2d(128, 128, stride=4, kernel_size=4, padding=0)
            self.up_conv_2 = nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1)
        elif backbone == 'swin_t':
            self.up_conv_1 = nn.ConvTranspose2d(192, 128, stride=4, kernel_size=4, padding=0)
            self.up_conv_2 = nn.ConvTranspose2d(96, 128, kernel_size=4, stride=2, padding=1)
        else:
            raise NotImplementedError("Backbone only support resnet18, pvt_v2_b1, and swin_t")

    def forward(self, x, edge_feat, spec_feat_list):
        """
        Args:
            x (Tensor): Input tensor.
            edge_feat (Tensor): Shallow edge feature.
            spec_feat_list (list): List of spectral features.

        Returns:
            Tensor: Output tensor after decoding.
        """
        x = self.norm(x)
        x = self.conv_0(x)
        x = self.syncbn_fc_0(x)
        x = F.relu(x, inplace=False)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode="bilinear", align_corners=False)

        x = self.conv_1(x)
        x = self.syncbn_fc_1(x)
        x = F.relu(x, inplace=False)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode="bilinear", align_corners=False)

        x = torch.cat([x, edge_feat], dim=1)
        x = self.conv_2(x)
        x = self.syncbn_fc_2(x)
        x = F.relu(x, inplace=False)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode="bilinear", align_corners=False)

        spec_feat_1 = self.up_conv_1(spec_feat_list[1])
        spec_feat_2 = self.up_conv_2(spec_feat_list[0])

        x = self.conv_3(x + torch.cat([spec_feat_1, spec_feat_2], dim=1))
        x = self.syncbn_fc_3(x)
        x = F.relu(x, inplace=False)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode="bilinear", align_corners=False)
        x = self.conv_4(x)
        return x


class SMN(nn.Module):
    def __init__(self, config):
        super(SMN, self).__init__()
        self.config = config
        hidden_size = config.hidden_size

        # Edge embedding layers
        self.edge_embedding_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, hidden_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )
        self.edge_embedding_2 = EdgeDetectionModule(in_dim=hidden_size, out_dim=hidden_size // 2,
                                                    size=config.image_size // 16)

        # Spectral saliency embbeding
        self.spec_embedding = SpecEmbedding(config, in_channels=3, backbone=config.backbone)

        self.attn = MixedFrequencyAttention(config)

        self.decoder = SalEdgeAwareDecoder(embed_dim=hidden_size, size=config.image_size, backbone=config.backbone)

    def forward(self, spec_sals, edges):
        """
        Args:
            spec_sals (Tensor): Spectral saliency maps.
            edges (Tensor): Spectral edge maps.

        Returns:
            Tuple[Tensor, Tensor]: Output of the network and edge output.
        """
        # edge_feat: used in decoder 
        edge_feat = self.edge_embedding_1(edges)
        # edge_feat_1: used in mixed-frequency attention, edge_out: supervision
        edge_feat_1, edge_out = self.edge_embedding_2(edge_feat)

        # s_feat: used in decoder and mixed-frequency attention, s_feat_list: used in decoder
        s_feat, s_feat_list = self.spec_embedding(spec_sals)

        # mixed-frequency attenion
        decoder_input = self.attn(s_feat, edge_feat_1)

        # saliency-edge-aware decoder
        out = self.decoder(decoder_input, edge_feat, s_feat_list)
        return out, edge_out


def get_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"grid": (14, 14)})
    config.hidden_size = 256
    config.image_size = image_size
    config.backbone = "pvt_v2_b1"

    config.NAT = ml_collections.ConfigDict()
    config.NAT.kernel_size_ca = 13
    config.NAT.kernel_size_sa = 9
    config.NAT.num_heads = 1
    return config
