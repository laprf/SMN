# This code is based on https://github.com/Samaretas/ITTI-saliency-detection.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GenSaliencyFeats(nn.Module):
    def __init__(self):
        super(GenSaliencyFeats, self).__init__()
        kernel = torch.tensor([[1, 4, 6, 4, 1],
                               [4, 16, 24, 16, 4],
                               [6, 24, 36, 24, 6],
                               [4, 16, 24, 16, 4],
                               [1, 4, 6, 4, 1]]) / 256
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def forward(self, image):
        '''

        Args:
            image: [B,H,W,C]

        Returns:
            saliency_feats: []

        '''
        shape = image.shape[1:3]
        # generate Gaussian pyramid
        image_gaussian_pyr = self.get_gaussian_pyramid(image.cuda())
        # compute 6 feature maps at different scales
        maps = self.center_surround_SAD(image_gaussian_pyr)
        # normalize feature maps using map normalization
        # get conspicuity map from normalized maps
        # NOTE: in 1998 paper they use scale 4 for the conspicuity maps, here I use scale 0
        maps = [
            F.interpolate(
                m.unsqueeze(1), shape, mode="bilinear", align_corners=True
            )
            .squeeze(1)
            for m in maps
        ]
        saliency_feats = []
        saliency_conspicuitys = torch.zeros_like(maps[0])

        for cs_index in range(len(maps)):
            batch = saliency_conspicuitys.shape[0]
            saliency_conspicuitys = maps[cs_index]
            min = torch.min(saliency_conspicuitys.view(batch, -1), 1)[0].unsqueeze(1).unsqueeze(2).repeat(1, shape[0],
                                                                                                          shape[1])
            max = torch.max(saliency_conspicuitys.view(batch, -1), 1)[0].unsqueeze(1).unsqueeze(2).repeat(1, shape[0],
                                                                                                          shape[1])
            saliency_conspicuitys = (saliency_conspicuitys - min) / (max - min)
            # saliency_conspicuitys = saliency_conspicuitys[:, :,
            #                         torch.arange(saliency_conspicuitys.shape[2] - 1, -1, -1)]
            saliency_feats.append(saliency_conspicuitys.cpu())
        return rearrange(saliency_feats, 'm b h w -> b m h w')

    def SAD(self, M_c, M_s):
        fenzi = torch.sum(M_c * M_s, axis=3)
        fenmu = torch.sqrt(torch.sum(M_c * M_c, axis=3)) * torch.sqrt(torch.sum(M_s * M_s, axis=3))
        SAD = torch.arccos(fenzi / fenmu)
        return SAD

    def pyr_down(self, input_layer):
        input_layer = rearrange(input_layer, 'b h w c -> b c h w')
        new_arr = torch.zeros_like(input_layer)
        channels = input_layer.shape[1]
        new_arr = F.conv2d(input_layer, self.weight.repeat(channels, 1, 1, 1), padding=2,
                           groups=channels)
        new_arr = rearrange(new_arr, 'b c h w -> b h w c')
        return new_arr[:, ::2, ::2, :]

    def get_gaussian_pyramid(self, image):
        """
            Get gaussian pyramid with 8 levels of downsampling.
        """
        pyr = list()
        pyr.append(image)
        for i in range(1, 9):
            next_layer = self.pyr_down(pyr[i - 1])
            pyr.append(next_layer)
        return pyr

    def center_surround_SAD(self, gauss_pyr):
        maps = list()
        for c in range(2, 5):
            center = gauss_pyr[c]
            size = (center.shape[2], center.shape[1])
            for s in range(3, 4):
                # Transpose
                surround = F.interpolate(rearrange(gauss_pyr[c + s], 'b h w c -> b c h w'), size, mode='bilinear',
                                         align_corners=True)
                surround = rearrange(surround, 'b c h w -> b w h c')
                cs_SAD_map = self.SAD(center, surround)
                maps.append(cs_SAD_map)
        return maps

    def simple_normalization(self, image, M=1):
        img_min, img_max = image.min(), image.max()
        if img_min != img_max:
            normalized = image / (img_max - img_min) - img_min / (img_max - img_min)
            if M != 1:
                normalized = normalized * M
        else:
            normalized = image - img_min
        return normalized

    def compute_average_local_maxima(self, feature_map, stepsize=-1):
        # NOTE: I compute local maxima taking into account last slices of the matrix
        # 30 corresponds to ~1 degree of visual angle
        local_avg_size = 20
        width = feature_map.shape[1]
        height = feature_map.shape[0]
        avg_size = local_avg_size if (stepsize < 1) else stepsize
        if (avg_size > height - 1):
            avg_size = height - 1
        if (avg_size > width - 1):
            avg_size = width - 1
        # find local maxima
        num_maxima = 0
        sum_all_maxima = 0

        for y in range(0, height - avg_size, avg_size):
            for x in range(0, width - avg_size, avg_size):
                local_img = feature_map[y:y + avg_size, x:x + avg_size]
                loc_max = local_img.max()
                sum_all_maxima += loc_max
                num_maxima += 1
                last_x = x + avg_size
            local_img = feature_map[y:y + avg_size, last_x:(width)]
            loc_max = local_img.max()
            sum_all_maxima += loc_max
            num_maxima += 1
            last_y = y + avg_size

        for x in range(0, width - avg_size, avg_size):
            local_img = feature_map[last_y:height, x:x + avg_size]
            loc_max = local_img.max()
            sum_all_maxima += loc_max
            num_maxima += 1
            last_x = x + avg_size
        local_img = feature_map[last_y:height, last_x:(width)]
        loc_max = local_img.max()
        sum_all_maxima += loc_max
        num_maxima += 1

        # averaging over all the local regions
        return sum_all_maxima / num_maxima

    def normalize_map(self, feature_maps):
        """
            This function implements the particular normalization operator N
            described in Itti 1998.
        """
        # normalize in range [0...M], choice M=1
        itti_normalized = torch.zeros_like(feature_maps)
        for i in range(feature_maps.shape[0]):
            feature_map = feature_maps[i, :, :]
            M = 1
            simply_normalized = self.simple_normalization(feature_map, M)
            # get average local maximum
            avg_local_maximum = self.compute_average_local_maxima(
                simply_normalized)
            # normalize feature map as from paper
            coeff_normalization = (M - avg_local_maximum) ** 2
            itti_normalized[i, :, :] = simply_normalized * coeff_normalization
        return itti_normalized
