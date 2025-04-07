import hydra
import torch

from omegaconf import DictConfig, OmegaConf

from Code import WandbRunner


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    device = f'cuda:{cfg.cuda_id}'
    WandbRunner.CreateRun(cfg, device)

if __name__ == '__main__':
    main()
