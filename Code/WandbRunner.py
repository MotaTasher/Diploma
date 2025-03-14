import random
import string
import torch
import wandb

import numpy as np
import pandas as pd

from . import Dataloader
from . import Dataset
from . import ModelBertV1
from . import Train

from sklearn.model_selection import train_test_split

def CreateRun(config: dict):
    id = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    run_name = config.get('name', 'model training')

    df_config = config.get('df_config', dict())
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    df_raw_fabtic = getattr(Dataloader, config.get('dataset_fabric', 'DatasetTimeBoomAuthorDependence'))
    df_raw = df_raw_fabtic(**df_config)

    cnt_known_address = config.get('cnt_known_address', 100)

    train_batch_size = config.get('train_batch_size', 64)
    test_batch_size = config.get('test_batch_size', 64)
    sample_len = config.get('sample_len', 100)
    
    model_fabric = getattr(ModelBertV1,
                        config.get('model', 'BertForTransactionRegressionV1'))
    model_params = config.get('model_params')

    train_params = dict(
        time_coef_loss=config.get('time_coef_loss', 1),

        cnt_last_for_show=config.get('cnt_last_for_show', 50),
        seconds_betwen_image_show=config.get('seconds_betwen_image_show', 10),

        start_epoch=0,
        num_epochs=config.get('num_epochs', 30),
        learning_rate=config.get('learning_rate', 3e-4),
        gamma=config.get('gamma', 0.8),

        step_size=config.get('step_size', 10),
        device=device,
    )
    model_predictor = getattr(Train, config.get('model_predictor', 'time_cross_predictior'))
    result_loss = getattr(Train, config.get('result_loss', 'result_loss_slower_change'))

    with wandb.init(config=config,
                    name=f"{id} run: {run_name}") as run:

        std, std_sq = (df_raw.value - df_raw.value.mean()).std(), (df_raw.value - df_raw.value.mean()).std() ** 2
        run.log(dict(std=std, std_sq=std_sq))

        df_to_for_predict = df_raw
        train_dataset, val_dataset = train_test_split(
            df_raw, test_size=1/4, shuffle=False)

        known_address = pd.concat(
            [df_to_for_predict['to'], df_to_for_predict['from']]).value_counts()[:cnt_known_address].keys().sort_values()

        # run.log(dict(known_address=known_address))

        dataset_params = dict(
            known_address=list(known_address),
            sample_len=sample_len
        )

        train_dataloader = torch.utils.data.DataLoader(
            Dataset.TransactionDataset(train_dataset, **dataset_params),
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0)

        val_dataloader = torch.utils.data.DataLoader(
            Dataset.TransactionDataset(val_dataset, **dataset_params),
            batch_size=test_batch_size,
            shuffle=True,
            num_workers=0)

        train_params['train_loader']=train_dataloader
        train_params['val_loader']=val_dataloader

        model = model_fabric(known_address_len=len(known_address), **model_params).to(device)

        Train.train_model(
            model=model,
            model_predictor=model_predictor,
            result_loss=result_loss,
            run=run,
            **train_params
        )