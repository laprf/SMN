import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def generate_kernel(size):
    assert size in [5, 15, 25], "kernel size must be 5, 15 or 25!"
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)

    for i in range(size):
        if i < center + 1:
            for j in range(size):
                if j < center:
                    kernel[i, j] = j - center - i
                elif j > center:
                    kernel[i, j] = 0 - kernel[i, size - j - 1]
        elif i > center:
            kernel[i, :] = kernel[size - i - 1, :]

    return kernel


class GenEdge(nn.Module):
    def __init__(self, kernel_size=25):
        super(GenEdge, self).__init__()
        self.kernel_size = kernel_size

        kernel_x = generate_kernel(kernel_size)
        kernel_x = torch.from_numpy(kernel_x).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).cuda()
        kernel_y = rearrange(kernel_x, 'b c h w -> b c w h')
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).cuda()

    def forward(self, cube):
        B, H, W, C = cube.shape
        k = self.kernel_size
        pad_len = k // 2
        edge_map = torch.zeros([B, H, W]).cuda()
        edge_map_out = torch.zeros([B, H, W]).cuda()

        # divide the cube into 10*10 blocks to avoid OOM
        num_blocks = 10
        block_H = H // num_blocks
        block_W = W // num_blocks

        # [B, H+2*pad_len, W+2*pad_len, C]
        padded_cube = F.pad(cube.permute(0, 3, 1, 2), (pad_len, pad_len, pad_len, pad_len), mode='constant',
                            value=0).permute(0, 2, 3, 1)

        for i in range(num_blocks):
            for j in range(num_blocks):
                start_H = i * block_H
                end_H = (i + 1) * block_H + pad_len * 2 if i < num_blocks - 1 else H + pad_len
                start_W = j * block_W
                end_W = (j + 1) * block_W + pad_len * 2 if j < num_blocks - 1 else W + pad_len

                # each block: [B, H/num_blocks+2*pad_len, W/num_blocks+2*pad_len, C]
                cube_block = padded_cube[:, start_H:end_H, start_W:end_W, :]
                edge_block = self.compute_block(cube_block)
                edge_map[:, start_H:end_H - 2 * pad_len, start_W:end_W - 2 * pad_len] = edge_block

        # remove boundary error
        edge_map_out[:, pad_len:-pad_len, pad_len:-pad_len] = edge_map[:, pad_len:-pad_len, pad_len:-pad_len]
        return edge_map_out / torch.max(edge_map_out)

    def compute_block(self, cube):
        B, H, W, C = cube.shape
        k = self.kernel_size
        H = H - 2 * k // 2 + 1
        W = W - 2 * k // 2 + 1

        # [B, H, W, C, k, k]
        cube_unfold = cube.unfold(1, k, 1).unfold(2, k, 1)
        cube_center = cube[:, k // 2:-k // 2 + 1, k // 2:-k // 2 + 1].unsqueeze(3).unsqueeze(3).repeat(1, 1, 1, k, k,
                                                                                                       1).permute(0, 1,
                                                                                                                  2, 5,
                                                                                                                  3, 4)

        # compute Spectral Angular Distance (SAD) and gradient
        region_SAD = self.SAD(cube_center, cube_unfold)
        grad_x = F.conv2d(region_SAD.view(-1, 1, k, k), self.weight_x).view(B, H, W)
        grad_y = F.conv2d(region_SAD.view(-1, 1, k, k), self.weight_y).view(B, H, W)
        edge_map = torch.abs(grad_x) + torch.abs(grad_y)

        return edge_map

    def SAD(self, M_c, M_s):
        numer = torch.sum(M_c * M_s, dim=3)
        denomin = M_c.norm(p=2, dim=3) * M_s.norm(p=2, dim=3) + 1e-4
        SAD = torch.arccos(numer / denomin)
        return SAD
