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
from omegaconf import DictConfig, OmegaConf


def CreateRun(cfg: DictConfig, device: str):
    run_id = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))

    # === 1. Загружаем данные
    df_raw = getattr(Dataloader, cfg.dataset_fabric)(**cfg.df_config)

    known_address = pd.concat([df_raw['to'], df_raw['from']])\
                     .value_counts()[:cfg.cnt_known_address]\
                     .keys().sort_values()

    dataset_params = {
        'known_address': list(known_address),
        'sample_len': cfg.sample_len
    }

    train_data, val_data = train_test_split(df_raw, test_size=1/4, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        Dataset.TransactionDataset(train_data, **dataset_params),
        batch_size=cfg.train_batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        Dataset.TransactionDataset(val_data, **dataset_params),
        batch_size=cfg.test_batch_size,
        shuffle=False
    )

    # === 2. Создаём модель
    model = getattr(ModelBertV1, cfg.model)(
        known_address_len=len(known_address),
        **cfg.model_params
    ).to(device)

    # === 3. Собираем конфигурацию для Trainer
    config = Train.Config(
        # num_epochs=cfg.num_epochs,
        # learning_rate=cfg.learning_rate,
        # gamma=cfg.gamma,
        # step_size=cfg.step_size,
        # warmup_epochs=cfg.warmup_epochs,
        # p_change=cfg.p_change,
        # p_msk=cfg.p_msk,
        # p_save=cfg.p_save,
        # change_strategy=cfg.change_strategy,
        # loss_fn=cfg.loss_fn,
        # result_loss=cfg.result_loss,
        # time_coef_loss=cfg.time_coef_loss,
        # seconds_betwen_image_show=cfg.seconds_betwen_image_show,
        # cnt_last_for_show=cfg.cnt_last_for_show,
        # loggin_each=cfg.loggin_each,
        # show_img=cfg.show_img,
        # show_batch_size=cfg.show_batch_size,
        # predictor=cfg.model_predictor,
        # start_epoch=cfg.start_epoch,
        **cfg
    )

    # === 4. Запуск через wandb
    try:
        with wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                        name=f"{run_id} run: {cfg.name}") as run:

            trainer = Train.Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                run=run,
                config=config,
                device=device
            )

            trainer.train()

    except KeyboardInterrupt:
        print("Training interrupted. Cleaning up.")
        torch.cuda.empty_cache()
