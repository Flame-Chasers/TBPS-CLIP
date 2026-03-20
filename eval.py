import os
import random
import time
from pathlib import Path

import torch

from misc.build import load_checkpoint, cosine_scheduler, build_optimizer
from misc.data import build_pedes_data
from misc.eval import test
from misc.utils import parse_config, init_distributed_mode, set_seed, is_master, is_using_distributed, \
    AverageMeter
from model.tbps_model import clip_vitb
from options import get_args


def eval(config):
    print(config)

    # data
    dataloader = build_pedes_data(config)
    train_loader = dataloader['train_loader']
    num_classes = len(train_loader.dataset.person2text)

    # model
    model = clip_vitb(config, num_classes)
    model.to(config.device)

    model, load_result = load_checkpoint(model, config)

    if is_using_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.device],
                                                          find_unused_parameters=True)

    if is_master():
        eval_result = test(model.module, dataloader['test_loader'], config.experiment.text_length, config.device)
        rank_1, rank_5, rank_10, map = eval_result['r1'], eval_result['r5'], eval_result['r10'], eval_result['mAP']
        print('Acc@1 {top1:.5f} Acc@5 {top5:.5f} Acc@10 {top10:.5f} mAP {mAP:.5f}'.format(top1=rank_1, top5=rank_5,
                                                                                          top10=rank_10, mAP=map))


if __name__ == '__main__':
    config_path = 'config/config.yaml'

    args = get_args()
    if args.simplified:
        config_path = 'config/s.config.yaml'
    config = parse_config(config_path)

    init_distributed_mode(config)

    set_seed(config)

    eval(config)
