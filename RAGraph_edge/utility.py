import torch
import random
import numpy as np
import pandas as pd

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def merge_pd(pds):
    if not isinstance(pds, list):
        return pds
    for i in range(len(pds)):
        if i == 0:
            merged = pds[i]
        else:
            merged = pd.merge(merged, pds[i], on=["user"], how="left")

            merged.loc[merged["item_y"].notna(), "item_x"] = (
                merged.loc[merged["item_y"].notna(), "item_x"] + " " + merged.loc[merged["item_y"].notna(), "item_y"]
            )
            merged.drop(columns=["item_y"], inplace=True)

            merged.loc[merged["time_y"].notna(), "time_x"] = (
                merged.loc[merged["time_y"].notna(), "time_x"] + " " + merged.loc[merged["time_y"].notna(), "time_y"]
            )
            merged.drop(columns=["time_y"], inplace=True)
            merged.rename(columns={"item_x": "item", "time_x": "time"}, inplace=True)
    return merged
