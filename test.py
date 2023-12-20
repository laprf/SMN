import argparse
import os

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import HSI_dataset
from models.SMN_modeling import SMN, get_config


class Test(object):
    def __init__(self, network, path, snapshot):
        """
        Initializes the testing environment.

        Args:
            network: Neural network model class.
            path: Path to the dataset.
            snapshot: Path to the pre-trained model.
        """
        # Configuration for the dataset
        self.cfg = HSI_dataset.Config(datapath=path, snapshot=snapshot, mode="test")

        # Initialize model
        config = get_config()
        config.backbone = args.backbone
        self.net = network(config).cuda()

        # Load the pre-trained model
        model_dict = self.net.state_dict()
        pretrained_dict = torch.load(self.cfg.snapshot, map_location=torch.device("cpu"))

        # Check for any missing keys in the pretrained model
        for k, v in model_dict.items():
            if k not in pretrained_dict.keys():
                print("miss keys in pretrained_dict: {}".format(k))

        model_dict.update(pretrained_dict)
        print("load pretrained model from {}".format(self.cfg.snapshot))
        self.net.load_state_dict(model_dict)

        self.net.train(False)

        self.data = HSI_dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=8, shuffle=False, num_workers=0)

    def save(self):
        with torch.no_grad():
            for edge, gt, spec, (H, W), name in tqdm(self.loader):
                spec, gt, edge = spec.cuda().float(), gt.cuda().float(), edge.cuda().float()

                out, _ = self.net(spec, edge)

                pred = torch.sigmoid(out)
                pred = F.interpolate(pred, (H[0], W[0]), mode="bilinear", align_corners=True)

                # Save the output images
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for i in range(pred.shape[0]):
                    cv2.imwrite(save_path + "/" + name[i].split(".")[0] + ".jpg", pred[i, 0].cpu().numpy() * 255)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="pvt_v2_b1")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str, default="HSOD-BIT")
    args = parser.parse_args()

    save_path = "DataStorage/" + args.dataset + "/exp_results/" + args.backbone + "/test_result"

    t = Test(SMN, args.data_path, args.model_path)
    t.save()
