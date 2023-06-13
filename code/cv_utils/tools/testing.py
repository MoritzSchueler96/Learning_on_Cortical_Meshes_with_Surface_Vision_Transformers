import argparse

import pytorch_lightning as pl
import yaml
import wandb
import os
import torch
from dotenv import load_dotenv
from cv_utils.datamodule import SViTDataModule
from cv_utils.models import SViT
from cv_utils.utils.utils import set_seeds
from pytorch_lightning.loggers import WandbLogger
from cv_utils.utils.utils import load_model_weights


load_dotenv()

BASE_REPO_FOLDER = os.getenv("BASE_REPO_FOLDER")
LOG_DIR = os.path.join(BASE_REPO_FOLDER, "logs/test/")
PROJECT = os.getenv("PROJECT")
USER = os.getenv("USER")
wandb.tensorboard.patch(root_logdir=LOG_DIR)

def test(config):

    # set seed to get consistent results, deactivate if random results are wanted
    set_seeds(config["setup"]["seed"])

    run = wandb.init(project=PROJECT, entity=USER, dir=LOG_DIR, group=config["data"]["task"], job_type="testing", sync_tensorboard=True, anonymous="allow")
    run.save()
    wandb_logger = WandbLogger(name=run.name, log_model="all", sync_tensorboard=True)
    
    # Set num classes
    config["transformer"]["num_classes"] = config["data"]["num_classes"]

    data_module = SViTDataModule(config=config)
    data_module.prepare_data()
    data_module.setup()

    config["transformer"]["num_channels"] = data_module.testset.features.shape[1]

    model = SViT(config=config)

    use_gpu = config['setup']['use_gpu']

    # load model
    new_state_dict = load_model_weights(model, config)
    if new_state_dict:
        model.load_state_dict(new_state_dict)

    trainer = pl.Trainer(
        accelerator="auto" if use_gpu else "cpu", 
        devices=-1 if torch.cuda.is_available() and use_gpu else None,
        log_every_n_steps=1,
        default_root_dir=LOG_DIR,
        logger=wandb_logger,
    )

    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SViT')

    parser.add_argument(
                        '-config',
                        type=str,
                        default='./code/cv_utils/configs/testing.yaml',
                        help='path where the config is stored')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call testing
    test(config)

    wandb.finish()
