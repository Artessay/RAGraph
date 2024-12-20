import sys
sys.path.append('./')

import torch
import importlib
import numpy as np
import pandas as pd
import setproctitle

from os import path
from utils.parse_args import args
from utils.trainer import Trainer
from utils.dataloader import EdgeListData
from utils.logger import Logger, log_exceptions
from utility import init_seed, merge_pd

setproctitle.setproctitle('RAGraph')

modules_class = 'modules.'
if args.plugin:
    modules_class = 'modules.plugins.'

module_name = 'RAGraph'
args.exp_name = 'RAGraph-vanilla'
args.pre_model = module_name
args.f_model = module_name

def import_pretrained_model():
    module = importlib.import_module(modules_class + args.pre_model)
    return getattr(module, args.pre_model)

def import_vanilla_model():
    module = importlib.import_module(modules_class + args.pre_model)
    return getattr(module, args.pre_model)

init_seed(args.seed)
logger = Logger(args)

pretrain_data = path.join(args.data_path, "pretrain.txt")
pretrain_val_data = path.join(args.data_path, "pretrain_val.txt")
finetune_data = path.join(args.data_path, "fine_tune.txt")
test_data_num = 8 if args.data_path.split('/')[-1] == 'amazon' else 4
logger.log(f"test_data_num: {test_data_num}")
test_datas = [path.join(args.data_path, f"test_{i}.txt") for i in range(1, test_data_num+1)]
all_data = [pretrain_data, finetune_data, *test_datas]

recalls, ndcgs = [], []

@log_exceptions
def run():
    pretrain_dataset = EdgeListData(pretrain_data, pretrain_val_data)

    recalls, ndcgs = [], []
    for num_stage in range(1, test_data_num+1):
        test_data_idx = num_stage + 1
        ft_data_idx = test_data_idx - 1
    

        ##########################################################################################
        # structural prompt construction
        pretrain_df = pd.read_csv(pretrain_data, sep="\t", names=["user", "item", "time"])
        all_data_pd = [
            pretrain_df,
            pd.read_csv(finetune_data, sep="\t", names=["user", "item", "time"]),
            *[pd.read_csv(test_data, sep="\t", names=["user", "item", "time"]) for test_data in test_datas],
        ]
        data_to_merge = all_data_pd[:ft_data_idx+1]
        merged_pre_pd = merge_pd(data_to_merge)

        # test file here is useless
        pre_dataset = EdgeListData(
            train_file=merged_pre_pd,
            test_file=all_data_pd[ft_data_idx],
            has_time=True,
            pre_dataset=pretrain_dataset,
        )

        # pretrain model
        pretrain_model = import_pretrained_model()(pre_dataset, phase='pretrain').to(args.device)
        pretrain_model.load_state_dict(torch.load(args.pre_model_path), strict=False)
        pretrain_model.eval()
        
        model = import_vanilla_model()(pre_dataset, pretrain_model, phase='vanilla').to(args.device)
        model.load_state_dict(torch.load(args.pre_model_path), strict=False)
        model.eval()

        logger.info(f"Learning Stage {num_stage}, test data: {all_data[test_data_idx]}, incremental train data: {all_data[ft_data_idx]}")

        finetune_dataset = EdgeListData(train_file=all_data[ft_data_idx], test_file=all_data[test_data_idx], phase='finetune', pre_dataset=pretrain_dataset, has_time=True, user_hist_files=all_data[:ft_data_idx])

        trainer = Trainer(finetune_dataset, logger)
        perform = trainer.test(model, finetune_dataset)

        recalls.append(perform["recall"][0])
        ndcgs.append(perform["ndcg"][0])

    logger.info(
        f"\n recalls: {recalls} \n ndcgs: {ndcgs} \n"
        f" avg. recall: {np.round(np.mean(recalls), 5)} std. {np.round(np.std(recalls), 5)}, \n"
        f" avg. ndcg:   {np.round(np.mean(ndcgs), 5)} std. {np.round(np.std(ndcgs), 5)}"
    )
    
 
if __name__ == "__main__":
    run()
