import hydra
import os

from omegaconf import DictConfig
from transformers import AutoTokenizer
from importlib import import_module
from torch.utils.data import DataLoader

from fache.model import FacheModel
from fache.work import train

@hydra.main(config_path='../conf', config_name='train_config', version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.model.dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.arch)
    tokenizer.save_pretrained(cfg.model.dir)

    train_data = import_module(f"fache.tasks.{cfg.run.task}.dataset").FacheData(
        tokenizer=tokenizer,
        max_len=cfg.data.max_len,
        split='train',
        data_dir=cfg.data.dir
    )
    train_data = DataLoader(
        train_data, 
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=train_data.collate_fn
    )

    dev_data = import_module(f"fache.tasks.{cfg.run.task}.dataset").FacheData(
        tokenizer=tokenizer,
        max_len=cfg.data.max_len,
        split='dev',
        data_dir=cfg.data.dir
    )
    dev_data = DataLoader(
        dev_data,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=dev_data.collate_fn
    )

    model = FacheModel(
        cfg.model.arch,
        cfg.model.classifier_hidden_size
    )

    train(model, train_data, dev_data, cfg.model.dir, cfg.train.n_workers, cfg.train.epochs)

if __name__ == '__main__':
    main()