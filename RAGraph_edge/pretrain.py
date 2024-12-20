import sys
sys.path.append('./')

from os import path
from utils.parse_args import args
from utils.dataloader import EdgeListData

import random
import numpy as np
import torch
from utils.logger import Logger, log_exceptions
from utils.trainer import Trainer
import importlib
import setproctitle

setproctitle.setproctitle('RAGraph')

modules_class = 'modules.'
if args.plugin:
    modules_class = 'modules.plugins.'

def import_model():
    module = importlib.import_module(modules_class + args.model)
    return getattr(module, args.model)

def import_pretrained_model():
    module = importlib.import_module(modules_class + args.pre_model)
    return getattr(module, args.pre_model)

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


init_seed(args.seed)
logger = Logger(args)

pretrain_data = path.join(args.data_path, "pretrain.txt")
pretrain_val_data = path.join(args.data_path, "pretrain_val.txt")
finetune_data = path.join(args.data_path, "fine_tune.txt")
test_data_num = 8 if args.data_path.split('/')[-1] == 'amazon' else 4
logger.log(f"test_data_num: {test_data_num}")
test_datas = [
    path.join(args.data_path, f"test_{i}.txt") for i in range(1, test_data_num+1)]
all_data = [pretrain_data, finetune_data, *test_datas]


if args.phase == "pretrain":
    @log_exceptions
    def run():
        edgelist_dataset = EdgeListData(pretrain_data, pretrain_val_data)

        model = import_pretrained_model()(edgelist_dataset, phase='pretrain').to(args.device)

        trainer = Trainer(edgelist_dataset, logger)
        trainer.train(model)
    run()

if args.phase == "pretrain_vanilla":
    @log_exceptions
    def run():
        edgelist_dataset = EdgeListData(pretrain_data, pretrain_val_data)

        model = import_model()(edgelist_dataset, phase='vanilla').to(args.device)

        trainer = Trainer(edgelist_dataset, logger)
        trainer.train(model)
    run()