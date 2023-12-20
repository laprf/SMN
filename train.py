import argparse
import os
import warnings

import torch
import torch.nn.functional as F
from natten.flops import get_flops
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import HSI_dataset
import pytorch_iou
from models.SMN_modeling import SMN, get_config
from utils import AverageMeter, count_parameters, mean_square_error, clip_gradient, set_seed


def setup(config):
    """
    Initializes and sets up the SMN model.
    Calculates and prints the number of parameters and GFLOPs.

    Returns:
        Initialized model object.
    """
    # Initialize SMN model with provided arguments
    model = SMN(config)

    # Calculate number of parameters and GFLOPs for the model
    num_params = count_parameters(model)
    flops = get_flops(model,
                      (torch.rand(1, 3, config.image_size, config.image_size),
                       torch.rand(1, 3, config.image_size, config.image_size)),
                      True)
    print(f'Number of parameters: {num_params:.3f}M, FLOPs: {flops / 1e9:.3f}G')
    return model


def valid_one_epoch(test_loader, net):
    """
    Validates the model for one epoch, calculates and returns the average mean absolute error.

    Args:
        test_loader: DataLoader for the test dataset

    Returns:
        Average MAE for the epoch.
    """
    maes = AverageMeter()
    net.train(False)
    with torch.no_grad():
        for edge, gt, spec, (H, W), name in test_loader:
            spec = spec.cuda().float()
            gt = gt.cuda().float()
            edge = edge.cuda().float()

            out, _ = net(spec, edge)
            pred = torch.sigmoid(out)
            pred = F.interpolate(pred, (H[0], W[0]), mode="bilinear", align_corners=False)

            # (Optional) Save predicted and ground truth images
            # head = "./DataStorage/"  + args.dataset + "/exp_results/"+ args.backbone + "/valid_result"
            # os.makedirs(head, exist_ok=True)
            # for i in range(pred.shape[0]):
            #     cv2.imwrite(
            #         head + "/" + name[i].split(".")[0] + "_out.jpg", pred[i, 0].cpu().numpy() * 255
            #     )
            #     cv2.imwrite(head + "/" + name[i].split(".")[0] + "_gt.jpg", gt[i, 0].cpu().numpy() * 255)

            # Calculate and update the mean absolute error
            mae = mean_square_error(gt, pred)
            maes.update(mae)
    return maes.avg


def train_all_epoches(train_cfg, train_loader, test_loader, net):
    """
    Trains the model for all the epochs, saves the best model and logs the training and validation losses.

    Args:
        train_cfg: Training configuration
        train_loader: DataLoader for the training dataset
        test_loader: DataLoader for the test dataset

    """
    optimizer = torch.optim.SGD(net.parameters(), lr=train_cfg.lr, momentum=train_cfg.momen,
                                weight_decay=train_cfg.decay, nesterov=True)
    sw = SummaryWriter()
    mae_loss_record = 0.07

    CE = torch.nn.BCELoss().cuda()
    iou_loss = pytorch_iou.IOU(size_average=True)

    for epoch in trange(train_cfg.epoch):
        losses = AverageMeter()
        net.train(True)

        # lr: warm-up and linear decay
        optimizer.param_groups[0]["lr"] = (1 - abs((epoch + 1) / (train_cfg.epoch + 1) * 2 - 1)) * train_cfg.lr

        for step, (edge, spec, gt, edge_gt, name) in enumerate(train_loader):
            edge, spec, gt, edge_gt = (
                edge.type(torch.FloatTensor).cuda(),
                spec.type(torch.FloatTensor).cuda(),
                gt.type(torch.FloatTensor).cuda(),
                edge_gt.type(torch.FloatTensor).cuda(),
            )
            out_final, edge = net(spec, edge)

            edge = F.interpolate(edge, (gt.shape[2], gt.shape[3]), mode="bilinear", align_corners=False)
            edge_loss = CE(torch.sigmoid(edge), edge_gt)

            out_final_prob = torch.sigmoid(out_final)
            sal_loss = iou_loss(out_final_prob, gt) + CE(out_final_prob, gt)

            optimizer.zero_grad()

            loss = edge_loss + sal_loss
            loss.backward()
            clip_gradient(optimizer, train_cfg.lr)
            optimizer.step()
            losses.update(loss)

        mae_loss = valid_one_epoch(test_loader, net)

        # tensorboard visualization
        sw.add_scalar("loss/valid", mae_loss, epoch + 1)
        sw.add_scalar("loss/train", losses.avg, epoch + 1)
        sw.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)

        # save model
        if mae_loss < mae_loss_record:
            if not os.path.exists(train_cfg.savepath):
                os.makedirs(train_cfg.savepath)
            torch.save(net.state_dict(), train_cfg.savepath + "/model-best")
            mae_loss_record = mae_loss


def main():
    # model config
    config = get_config()
    config.backbone = args.backbone
    net = setup(config).cuda()

    train_cfg = HSI_dataset.Config(datapath=args.data_path, savepath=save_path, mode="train", batch=5,
                                   lr=args.learning_rate, momen=0.9, decay=5e-4, epoch=100)
    train_data = HSI_dataset.Data(train_cfg)
    train_loader = DataLoader(train_data, collate_fn=train_data.collate, batch_size=train_cfg.batch, shuffle=True,
                              pin_memory=True, num_workers=0, drop_last=True)

    test_cfg = HSI_dataset.Config(datapath=args.data_path, mode="test")
    test_data = HSI_dataset.Data(test_cfg)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)

    train_all_epoches(train_cfg, train_loader, test_loader, net)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    set_seed(7)
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--backbone", type=str, default='pvt_v2_b1')  # resnet18, swin_t, pvt_v2_b1
    # training args
    parser.add_argument("--learning_rate", type=float,
                        default=7e-3)  # for resnet18: 2e-2, for swin_t and pvt_v2_b1: 7e-3
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str, default='HSOD-BIT')
    args = parser.parse_args()

    save_path = "DataStorage/trained_models/" + args.dataset + "/" + args.backbone
    os.makedirs(save_path, exist_ok=True)

    main()
