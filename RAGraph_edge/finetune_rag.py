import sys
sys.path.append('./')

import torch
import datetime
import importlib
import numpy as np
import pandas as pd
import setproctitle
import torch.nn.functional as F

from os import path
from utils.parse_args import args
from utils.trainer import Trainer
from utils.dataloader import EdgeListData
from utils.logger import Logger, log_exceptions
from utility import init_seed, merge_pd

args.exp_name = 'RAGraph-finetune'
args.pre_model = 'RAGraph'
args.f_model = 'RAGraph'

setproctitle.setproctitle('RAGraph')

modules_class = 'modules.'
if args.plugin:
    modules_class = 'modules.plugins.'

def import_pretrained_model():
    module = importlib.import_module(modules_class + args.pre_model)
    return getattr(module, args.pre_model)

def import_finetune_model():
    module = importlib.import_module(modules_class + args.f_model)
    return getattr(module, args.f_model)


init_seed(args.seed)
logger = Logger(args)

pretrain_data = path.join(args.data_path, "pretrain.txt")
pretrain_val_data = path.join(args.data_path, "pretrain_val.txt")
finetune_data = path.join(args.data_path, "fine_tune.txt")
test_data_num = (
    8 if args.data_path.split("/")[-1] == "amazon" else 4
)
logger.log(f"test_data_num: {test_data_num}")
test_datas = [path.join(args.data_path, f"test_{i}.txt") for i in range(1, test_data_num + 1)]

all_data = [pretrain_data, finetune_data, *test_datas]



@log_exceptions
def run():
    saved_model_paths = []

    pretrain_dataset = EdgeListData(pretrain_data, pretrain_val_data)

    recalls, ndcgs = [], []
    for num_stage in range(1, len(test_datas) + 1):
        interval = args.updt_inter
        if len(saved_model_paths) >= interval:
            all_state_dict = [torch.load(args.pre_model_path, map_location=args.device)]
            for i in range(interval):
                all_state_dict.append(torch.load(saved_model_paths[-i - 1], map_location=args.device))

            ##########################################################################################
            # interpolative update
            pretrain_weight = 0.5
            interpolative_weight = (1 - pretrain_weight) * F.normalize(
                torch.arange(1, interval + 1).float(), dim=0, p=1
            ).flip(dims=[0])
            interpolative_weight = (
                torch.cat([torch.tensor([pretrain_weight]), interpolative_weight])
                .unsqueeze(1)
                .unsqueeze(1)
                .to(args.device)
            )
            state_dict = {}
            for k in all_state_dict[0].keys():
                if any(k.startswith(s) for s in ["user_embedding", "item_embedding"]):
                    state_dict[k] = torch.sum(
                        torch.stack([sd[k] for sd in all_state_dict]) * interpolative_weight, dim=0
                    )
                    state_dict[k] = F.normalize(state_dict[k], dim=1)

        else:
            state_dict = torch.load(args.pre_model_path)

        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(s) for s in ["user_embedding", "item_embedding"]):
                new_state_dict[k] = v

        # starts with test_data, which equals 2
        test_data_idx = num_stage + 1
        ft_data_idx = test_data_idx - 1
        logger.info(
            f"Finetune Stage {num_stage}, test data: {all_data[test_data_idx]}, finetune data {all_data[ft_data_idx]}"
        )

        # use all data to propogate as prompt
        logger.info(f"use {all_data[:ft_data_idx]} to propogate as prompt")

        pretrain_df = pd.read_csv(pretrain_data, sep="\t", names=["user", "item", "time"])

        ##########################################################################################
        # structural prompt construction
        all_data_pd = [
            pretrain_df,
            pd.read_csv(finetune_data, sep="\t", names=["user", "item", "time"]),
            *[pd.read_csv(test_data, sep="\t", names=["user", "item", "time"]) for test_data in test_datas],
        ]
        data_to_merge = all_data_pd[:ft_data_idx+1]
        merged_pre_pd = merge_pd(data_to_merge)
        # data_to_merge = all_data_pd[:ft_data_idx+1]
        # if num_stage == 1:
        #     merged_pre_pd = merge_pd(data_to_merge)
        # else:
        #     sub_dfs = data_to_merge[1:]
        #     sample_decay = abs(args.samp_decay)
        #     samp_rates = [1 + sample_decay - sample_decay * i for i in range(1, len(sub_dfs) + 1)]
        #     if args.samp_decay < 0:
        #         # more recent data has higher sampling rate
        #         samp_rates = samp_rates[::-1]
        #     for i, df in enumerate(sub_dfs):
        #         sub_dfs[i] = df.sample(frac=samp_rates[i], random_state=args.seed)
        #     sub_dfs.insert(0, data_to_merge[0])
        #     merged_pre_pd = merge_pd(sub_dfs)

        # test file here is useless
        pre_dataset = EdgeListData(
            train_file=merged_pre_pd,
            test_file=all_data_pd[ft_data_idx],
            has_time=True,
            pre_dataset=pretrain_dataset,
        )

        # no gating weights to load
        pretrained_model = import_pretrained_model()(pre_dataset, phase="for_tune").to(args.device)

        pretrained_model.load_state_dict(new_state_dict, strict=True)
        logger.info(f"Successfully loaded: {new_state_dict.keys()}")
        pretrained_model.eval()

        finetune_dataset = EdgeListData(
            train_file=all_data[ft_data_idx],
            test_file=path.join(args.data_path, f"test_{num_stage}.txt"),
            phase="finetune",
            pre_dataset=pre_dataset,
            has_time=True,
            user_hist_files=all_data[:ft_data_idx],
        )
        model = import_finetune_model()(finetune_dataset, pretrained_model, phase="finetune").to(args.device)

        trainer = Trainer(finetune_dataset, logger, pre_dataset=pretrain_dataset)
        best_perform = trainer.train_finetune(model, pretrained_model)

        recalls.append(best_perform["recall"][0])
        ndcgs.append(best_perform["ndcg"][0])

        saved_model_paths.append(trainer.save_path)
        # update exp time for saving new model
        args.exp_time = datetime.datetime.now().strftime("%b-%d-%Y_%H-%M-%S")

    logger.info(
        f"\n recalls: {recalls} \n ndcgs: {ndcgs} \n"
        f" avg. recall: {np.round(np.mean(recalls), 5)} std. {np.round(np.std(recalls), 5)}, \n"
        f" avg. ndcg:   {np.round(np.mean(ndcgs), 5)} std. {np.round(np.std(ndcgs), 5)}"
    )

if __name__ == "__main__":
    run()
