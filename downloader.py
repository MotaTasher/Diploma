import random
import string
import torch
import wandb

import numpy as np
import pandas as pd

from Code import Dataloader
from Code import Dataset
from Code import ModelBertV1
from Code import Train


start_block = 22000000
end_block = 22000000 + 500 * 330
# end_block = 21100000
batch_size = 500

address_limit = 100
use_async = False

step = batch_size * 10

for i in range(start_block, end_block, step):
    Dataloader.GetEthereumDataset(
        start_block=i,
        end_block=i + step,
        address_limit=address_limit,
        use_async=use_async,
        batch_size=batch_size,
    )