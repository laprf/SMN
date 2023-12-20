import os

import cv2
import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from models.SpectralEdgeOperator import GenEdge
from models.SpectralSaliencyGenerator import GenSaliencyFeats


def img_size():
    return 224


class Compose(object):
    def __init__(self, input_transforms):
        self.transforms = input_transforms

    def __call__(self, edge, spec_sal, gt, edge_gt):
        for t in self.transforms:
            edge, spec_sal, gt, edge_gt = t(
                edge, spec_sal, gt, edge_gt
            )
        return edge, spec_sal, gt, edge_gt


class RandomHorizontallyFlip(object):
    def __call__(self, edge, spec, gt, edge_gt):
        if np.random.random() < 0.5:
            return (
                edge[:, :, torch.arange(spec.shape[2] - 1, -1, -1)],
                spec[:, :, torch.arange(spec.shape[2] - 1, -1, -1)],
                gt.transpose(Image.FLIP_LEFT_RIGHT),
                edge_gt.transpose(Image.FLIP_LEFT_RIGHT)
            )
        return edge, spec, gt, edge_gt


class RandomCrop(object):
    def __call__(self, edge, spec, gt, edge_gt):
        gt = np.array(gt)
        edge_gt = np.array(edge_gt)
        H, W = gt.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        edge = edge[:, p0:p1, p2:p3]
        spec = spec[:, p0:p1, p2:p3]
        gt = Image.fromarray(gt[p0:p1, p2:p3].astype("uint8"))
        edge_gt = Image.fromarray(edge_gt[p0:p1, p2:p3].astype("uint8"))
        return edge, spec, gt, edge_gt


class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        GSF = GenSaliencyFeats()
        GE5 = GenEdge(5)
        GE15 = GenEdge(15)
        GE25 = GenEdge(25)

        # train
        self.joint_transform_train = Compose(
            [
                RandomHorizontallyFlip(),
                RandomCrop(),
            ]
        )
        self.mask_transform_train = transforms.ToTensor()

        # test
        self.gt_transform_test = transforms.ToTensor()  # ->(C,H,W),(0~1)
        self.spec_transform_test = transforms.Compose(
            [transforms.Resize((img_size(), img_size())), transforms.ToTensor()]
        )
        with open(cfg.datapath + "/" + cfg.mode + ".txt", "r") as lines:
            self.samples = [line.strip() for line in lines]

        # Initialize lists for storing processed data
        self.gts, self.edge_gts, self.edges, self.specs = [], [], [], []

        print("Loading data...")

        for name in tqdm(self.samples):
            name = name.split(".")[0]
            os.makedirs(self.cfg.datapath + '/input_maps', exist_ok=True)
            mat_name = self.cfg.datapath + '/input_maps/' + name + ".mat"
            if not os.path.exists(mat_name):
                mat = h5py.File(self.cfg.datapath + "/hyperspectral/" + cfg.mode + "/" + name + ".mat",
                                "r")
                hypercube = np.float32(np.array(mat["hypercube"]))  # (C,H,W)
                hypercube = torch.from_numpy(hypercube / np.max(hypercube)).cuda()  # (C,H,W)

                # ------------ Shape transform, optional, feel free to comment out or modify ------------
                hypercube = hypercube[:, :, torch.arange(hypercube.shape[2] - 1, -1, -1)]
                if "HSOD-BIT" in cfg.datapath:
                    hypercube = rearrange(hypercube.unsqueeze(0), 'b c h w -> b h w c')  # [1,H,W,C]
                else:
                    hypercube = rearrange(hypercube.unsqueeze(0), 'b c h w -> b w h c')  # [1,H,W,C]
                    hypercube = hypercube[:, torch.arange(hypercube.shape[1] - 1, -1, -1), :]
                # --------------------------------------------------------------------------------------

                # generate and save edge map
                edge_5, edge_15, edge_25 = GE5(hypercube), GE15(hypercube), GE25(hypercube)
                edge = torch.concat((edge_5, edge_15, edge_25), dim=0).cpu()  # [3, H, W]
                edge_save = edge.numpy()

                # generate and save spec_sal map
                spec = GSF(hypercube).squeeze(0).cpu()  # [C, H, W]
                spec = spec / torch.max(spec)
                spec_save = spec.numpy()  # [C, H, W]
                sio.savemat(self.cfg.datapath + '/input_maps/' + name + '.mat', {'spec': spec_save, 'edge': edge_save})
            else:
                mat = sio.loadmat(mat_name)

                edge = np.float32(np.array(mat["edge"]))
                edge = torch.from_numpy(edge / np.max(edge))  # [C, H, W]

                spec = np.float32(np.array(mat["spec"]))
                spec = torch.from_numpy(spec / np.max(spec))  # [C, H, W]

            self.specs.append(spec)
            self.edges.append(edge)

            if self.cfg.mode == "train":
                gt = Image.open(
                    self.cfg.datapath + "/GT/train/" + name + ".jpg"
                ).convert("L")
                edge_gt = Image.open(
                    self.cfg.datapath + "/edge_GT/" + name + ".jpg"
                ).convert("L")

                self.gts.append(gt)
                self.edge_gts.append(edge_gt)
            else:
                gt = Image.open(
                    self.cfg.datapath + "/GT/test/" + name + ".jpg"
                ).convert("L")
                shape = gt.size[::-1]
                self.shape = shape
                self.gts.append(gt)

        print(f"{len(self.samples)} data loaded!")

    def __getitem__(self, idx):
        name = self.samples[idx]
        if self.cfg.mode == "train":
            edge, spec, gt, edge_gt = self.joint_transform_train(
                self.edges[idx],
                self.specs[idx],
                self.gts[idx],
                self.edge_gts[idx],
            )
            gt = self.mask_transform_train(gt)
            edge_gt = self.mask_transform_train(edge_gt)
            return edge, spec, gt, edge_gt, name
        else:
            edge = F.interpolate(
                self.edges[idx].unsqueeze(0), (img_size(), img_size()), mode="bilinear", align_corners=False
            ).squeeze(0)
            spec = F.interpolate(
                self.specs[idx].unsqueeze(0), (img_size(), img_size()), mode="bilinear", align_corners=False
            ).squeeze(0)
            gt = self.gt_transform_test(self.gts[idx])
            return edge, gt, spec, self.shape, name

    def __len__(self):
        return len(self.samples)

    def collate(self, batch):
        size = img_size()
        edge, spec, gt, edge_gt, name = [
            list(item) for item in zip(*batch)
        ]
        for i in range(len(batch)):
            gt[i] = np.array(gt[i]).transpose((1, 2, 0))
            gt[i] = cv2.resize(
                gt[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR
            )

            edge[i] = F.interpolate(edge[i].unsqueeze(0), size=(size, size), mode="bilinear",
                                    align_corners=False).squeeze(0)
            spec[i] = F.interpolate(spec[i].unsqueeze(0), size=(size, size), mode="bilinear",
                                    align_corners=False).squeeze(0)

            edge_gt[i] = np.array(edge_gt[i]).transpose((1, 2, 0))
            edge_gt[i] = cv2.resize(
                edge_gt[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR
            )
        edge = torch.stack(edge)
        spec = torch.stack(spec)
        gt = torch.from_numpy(np.stack(gt, axis=0)).unsqueeze(dim=1)
        edge_gt = torch.from_numpy(np.stack(edge_gt, axis=0)).unsqueeze(dim=1)
        return edge, spec, gt, edge_gt, name
