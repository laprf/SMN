import argparse

from dataloader import EvalDataset
from evaluator import Eval_thread


def main(cfg):
    if cfg.save_dir is not None:
        output_dir = cfg.save_dir
    else:
        output_dir = cfg.sm_dir
    gt_dir = cfg.gt_dir
    method = "SMN"
    dataset = cfg.datasets

    threads = []

    loader = EvalDataset(cfg.sm_dir, gt_dir)
    thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
    threads.append(thread)
    for thread in threads:
        print(thread.run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default="SMN")
    parser.add_argument('--datasets', type=str, default='HSOD-BIT')
    parser.add_argument('--sm_dir', type=str, default='')
    parser.add_argument('--gt_dir', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--cuda', type=bool, default=True)
    config = parser.parse_args()
    main(config)
