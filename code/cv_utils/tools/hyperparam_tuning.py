# training loop that tests out different hyperparameters and saves the results into the log folder
import argparse
import os
import random

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
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback, TuneReportCheckpointCallback)
from ray.tune.integration.wandb import wandb_mixin
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

os.environ["WANDB_CONSOLE"] = "off"

# ray.init(local_mode=True) # use for debugging

load_dotenv()

BASE_REPO_FOLDER = os.getenv("BASE_REPO_FOLDER")
LOG_DIR = os.path.join(BASE_REPO_FOLDER, "logs/tune/")
PROJECT = "svit"
USER = "moritzschueler96"
wandb.tensorboard.patch(root_logdir=LOG_DIR)

@wandb_mixin
def train_tune(config, checkpoint_dir=None, callbacks=[], epochs=1000, use_gpu=True):

    # set seed to get consistent results, deactivate if random results are wanted
    set_seeds(config["setup"]["seed"])

    task = config["data"]["task"]
    run = wandb.init(project=PROJECT, entity=USER, dir=LOG_DIR, group=task, job_type="tuning", sync_tensorboard=True, anonymous="allow")
    run.save()
    logger = WandbLogger(name=run.name, log_model='all')
    wandb.run.log_code(os.path.join(BASE_REPO_FOLDER, "code/cv_utils/"))

    # Set num classes
    config["transformer"]["num_classes"] = config["data"]["num_classes"]
    
    data_module = SViTDataModule(config=config)
    data_module.prepare_data()
    data_module.setup()

    config["transformer"]["num_channels"] = data_module.trainset.features.shape[1]
    config["setup"]["num_train_samples"] = data_module.trainset.features.shape[0]

    config["transformer"]["dropout"] = config["setup"]["dropout"]
    config["transformer"]["emb_dropout"] = config["setup"]["emb_dropout"]

    model = SViT(config=config)
    model.encoder = data_module.encoder

    config["learning_rate"] = config["setup"]["learning_rate"]
    config["batch_size"] = config["setup"]["batch_size"]

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
    callbacks += [lr_monitor_callback]

    trainer = pl.Trainer(
        accelerator="auto" if use_gpu else "cpu", 
        devices=-1 if torch.cuda.is_available() and use_gpu else None,
        log_every_n_steps=10,
        callbacks=callbacks,
        deterministic=True,
        default_root_dir=LOG_DIR,
        max_epochs=epochs,
        check_val_every_n_epoch=config["setup"]["val_epoch"],
        # enable_checkpointing=False,
        logger=logger,
    )  # gradient_clip_val=0.5, stochastic_weight_avg=True, check_val_every_n_epoch=10, num_sanity_val_steps=2, overfit_batches=0.01)
    
    trainer.validate(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)

    # might not be called due to scheduler and reporter which cancel training early if results don't look promising
    if config["setup"]["testing"]:
        trainer.test(model, datamodule=data_module)

    wandb.finish()

def hyperparameter_optimization(config):
    
    seeds = list(range(0, 1000000))
    config["setup"] = {**config["setup"], 
        "learning_rate": tune.sample_from(lambda: 10 ** random.uniform(**config["tune"]["lr"])),
        "batch_size": tune.choice(config["tune"]["batch_sizes"]),
        "epochs": config["tune"]["epochs"],
        "seed": tune.choice(seeds),
        "architecture": tune.choice(config["tune"]["architecture"]),
        "load_weights": tune.choice(config["tune"]["load_weights"]),
        "finetuning": tune.choice(config["tune"]["finetuning"]),
        "dropout": tune.sample_from(lambda: random.uniform(**config["tune"]["dropout"])),
        "emb_dropout": tune.sample_from(lambda: random.uniform(**config["tune"]["emb_dropout"])),
        "add_noise": tune.choice(config["tune"]["add_noise"]),
        "noise_scale": tune.sample_from(lambda: 10 ** random.uniform(**config["tune"]["noise_scale"])),
        "temperature": tune.sample_from(lambda: 10 ** random.uniform(**config["tune"]["temperature"])),
        "early_stopping": tune.choice(config["tune"]["early_stopping"])
    }
    use_scheduler = random.choice([True, False])
    config["optimisation"]["use_scheduler"] = tune.choice(config["tune"]["use_scheduler"])
    config["optimisation"]["weight_decay"] = tune.sample_from(lambda: random.uniform(**config["tune"]["weight_decay"]))
    config["optimisation"]["optimiser"] = tune.choice(config["tune"]["optimisers"])
    config["optimisation"]["scheduler"] = tune.choice(config["tune"]["schedulers"])
    config["optimisation"]["warmup"] = tune.choice(config["tune"]["warmup"])

    config["data"]["task"] = tune.choice(config["tune"]["task"]) # use different datasets (different split, oversampled etc)

    callback = TuneReportCallback(
        {"loss": "loss/val"}, on="validation_end"
    )
    callbacks = [callback]

    if use_scheduler:
        scheduler = ASHAScheduler(max_t=config["tune"]["epochs"], grace_period=1, reduction_factor=2)
    else:
        scheduler = None

    # start hyperparameter optimization
    analysis = tune.run(
        tune.with_parameters(
            train_tune, callbacks=callbacks, epochs=config["tune"]["epochs"], use_gpu=config["tune"]["use_gpu"]
        ),
        config={**config, 
        "wandb": 
            {"project": PROJECT}},
        num_samples=config["tune"]["num_trials"],
        local_dir=LOG_DIR,
        scheduler=scheduler,
        resources_per_trial={"gpu": 1},
        metric="loss",
        mode="min"
    )

    return analysis


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SViT')

    parser.add_argument(
                        '-config',
                        type=str,
                        default='./code/cv_utils/configs/tuning.yaml',
                        help='path where the config is stored')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call tuning
    analysis = hyperparameter_optimization(config)

    # get some information from the optimization
    best_trial = analysis.best_trial  # Get best trial
    best_config = analysis.best_config  # Get best trial's hyperparameters
    best_logdir = analysis.best_logdir  # Get best trial's logdir
    best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
    best_result = analysis.best_result  # Get best trial's last results
    best_result_df = analysis.best_result_df  # Get best result as pandas dataframe

    # Get a dataframe with the last results for each trial
    df_results = analysis.results_df

    # Get a dataframe of results for a specific score or mode
    df = analysis.dataframe(metric="loss", mode="min")

    df.to_csv("Results.csv")

    print("Best hyperparameters found were: ", analysis.best_config)
