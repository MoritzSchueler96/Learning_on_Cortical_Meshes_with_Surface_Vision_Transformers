import argparse
import os

import pytorch_lightning as pl
import torch
import wandb
import yaml
from cv_utils.callbacks import MyPrintingCallback, LogPredictionsCallback, LogLossStatsCallback
from cv_utils.datamodule import SViTDataModule
from cv_utils.models import SViT
from cv_utils.utils.utils import set_seeds
from dotenv import load_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

BASE_REPO_FOLDER = os.getenv("BASE_REPO_FOLDER")
LOG_DIR = os.path.join(BASE_REPO_FOLDER, "logs/")
PROJECT = os.getenv("PROJECT")
USER = os.getenv("USER")
wandb.tensorboard.patch(root_logdir=LOG_DIR)

def train(config):

    # set seed to get consistent results, deactivate if random results are wanted
    set_seeds(config["setup"]["seed"])

    task = config["data"]["task"]
    run = wandb.init(project=PROJECT, entity=USER, dir=LOG_DIR, group=task, job_type="training", sync_tensorboard=True, anonymous="allow") # anonymous is to get temp acc to see results
    run.save()
    logger = WandbLogger(name=run.name, log_model='all', sync_tensorboard=True)
    # logger = TensorBoardLogger(save_dir=LOG_DIR, name="sit_tb")
    run.log_code(os.path.join(BASE_REPO_FOLDER, "code/cv_utils/"))

    # Set num classes
    config["transformer"]["num_classes"] = config["data"]["num_classes"]

    data_module = SViTDataModule(config=config)
    data_module.prepare_data()
    data_module.setup()

    config["transformer"]["num_channels"] = data_module.trainset.features.shape[1]
    config["setup"]["num_train_samples"] = data_module.trainset.features.shape[0]
    
    model = SViT(config=config)
    model.encoder = data_module.encoder

    use_gpu = config['setup']['use_gpu']

    callbacks = [MyPrintingCallback()]
    if config["setup"]["save_ckpt"]:
        metric = config["setup"]["monitor"]
        checkpoint_callback = ModelCheckpoint(
            monitor=metric,
            filename="{task}-{epoch:02d}-{metric:.2f}",
            save_top_k=3,
            mode=config["setup"]["monitor_mode"],
            dirpath=os.path.join(run.dir, "checkpoints/")
        )
        callbacks += [checkpoint_callback]

    if config["data"]["num_classes"] == 1:
        callbacks += [LogLossStatsCallback()]

    if config["setup"]["early_stopping"]:
        metric = config["setup"]["monitor"]
        early_stopping_callback = EarlyStopping(
            monitor=metric, 
            mode=config["setup"]["monitor_mode"], 
            **config["early_stopping"])
        callbacks += [early_stopping_callback]

    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch', log_momentum=True)
    callbacks += [LogPredictionsCallback(), lr_monitor_callback]

    trainer = pl.Trainer(
        accelerator="auto" if use_gpu else "cpu", 
        devices=-1 if torch.cuda.is_available() and use_gpu else None,
        log_every_n_steps=10,
        callbacks=callbacks,
        deterministic=True,
        default_root_dir=LOG_DIR,
        max_epochs=config["setup"]["epochs"],
        check_val_every_n_epoch=config["setup"]["val_epoch"],
        logger=logger,
    )

    trainer.validate(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)

    if config["setup"]["testing"]:
        trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SViT')

    parser.add_argument(
                        '-config',
                        type=str,
                        default='./code/cv_utils/configs/training.yaml',
                        help='path where the config is stored')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call training
    train(config)

    wandb.finish()
