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
    for _ in range(10):
        run_id = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
        print(cfg.df_config)

        try:
            df_raw = getattr(Dataloader, cfg.dataset_fabric)(
                
                    **cfg.df_config,
                    **{'address_limit': cfg.cnt_known_address}
                )
            print(cfg.cnt_known_address)
            known_address = pd.concat([df_raw['to'], df_raw['from']])\
                            .value_counts()[:cfg.cnt_known_address]\
                            .keys().sort_values()

            dataset_params = {
                'known_address': list(known_address),
                'sample_len': cfg.sample_len
            }

            train_data, val_data = train_test_split(df_raw, test_size=1/4, shuffle=False)
            print(f"square std of all: {df_raw['value'].std() ** 2}\nTrain: {train_data['value'].std() ** 2}\nVal: {val_data['value'].std() ** 2}")

            train_loader = torch.utils.data.DataLoader(
                Dataset.TransactionDataset(train_data, **dataset_params, apply_log=cfg.use_log),
                batch_size=cfg.train_batch_size,
                shuffle=True,
                num_workers=12,
            )

            val_loader = torch.utils.data.DataLoader(
                Dataset.TransactionDataset(val_data, **dataset_params, apply_log=cfg.use_log),
                batch_size=cfg.test_batch_size,
                shuffle=False,
                num_workers=12,
            )

            model = getattr(ModelBertV1, cfg.model)(
                known_address_len=len(known_address),
                **cfg.model_params
            ).to(device)

            config = Train.Config(
                **cfg,
                run_id=run_id,
            )

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
                break

        except ValueError as e:
                if "NaN" in str(e) or "Inf" in str(e):
                    print("⚠️ NaN detected in training, restarting experiment...")
                    torch.cuda.empty_cache()
                    continue  # перезапуск
                else:
                    raise
        except KeyboardInterrupt:
            print("Training interrupted. Exiting.")
            torch.cuda.empty_cache()
            break

        else:
            print("❌ Too many NaN failures, aborting.")