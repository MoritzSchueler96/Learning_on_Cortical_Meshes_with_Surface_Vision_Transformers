# -*- coding: utf-8 -*-
# @Author: Moritz Schüler
# @Date:   2022-05-10 14:00:00
# @Last Modified by:   Moritz Schüler
# @Last Modified time: 2022-05-17 17:35:57
#
# Created on Fri Oct 01 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#

import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchmetrics
import yaml
from dotenv import load_dotenv
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.optim.lr_scheduler import (CosineAnnealingLR, ReduceLROnPlateau,
                                      StepLR)
from vit_pytorch.vit import Transformer
from warmup_scheduler import GradualWarmupScheduler
import wandb

from cv_utils import metrics
from cv_utils.utils.utils import load_model_weights

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

BASE_REPO_FOLDER = os.getenv("BASE_REPO_FOLDER")
CONFIG_DIR = os.path.join(BASE_REPO_FOLDER, "code/cv_utils/configs/")


class SViT(pl.LightningModule):
    """
    This is the Surface Vision Transformer for Alzheimer Prediction
    """

    def __init__(self, *, config=None):

        super().__init__()

        if config is None:
            config_path = os.path.join(CONFIG_DIR, "training.yaml")

            with open(config_path) as f:
                config = yaml.safe_load(f)

            config["use_default"] = True

        #### Model params ####
        dim=config['transformer']['dim']
        depth=config['transformer']['depth']
        heads=config['transformer']['heads']
        mlp_dim=config['transformer']['mlp_dim']
        pool=config['transformer']['pool']
        sub_ico = config['resolution']['sub_ico']
        num_patches=config['sub_ico_{}'.format(sub_ico)]['num_patches']
        num_vertices=config['sub_ico_{}'.format(sub_ico)]['num_vertices']
        num_classes=config['transformer']['num_classes']
        num_channels=config['transformer']['num_channels']
        dim_head=config['transformer']['dim_head']
        dropout=config['transformer']['dropout']
        emb_dropout=config['transformer']['emb_dropout']

        #### config stuff #####
        self.save_hyperparameters(config)
        self.config = config
        if "training" in config["setup"]["mode"]:
            self.learning_rate = config["setup"]["learning_rate"]
        else:
            self.learning_rate = 0.00001

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        patch_dim = num_channels * num_vertices

        # inputs has size = b * c * n * v
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # metrics
        # needs to be done here, s.t. device can be infered
        # if moved into own function like optimizer, the device will be wrong
        if self.config["transformer"]["num_classes"] == 1:
            metric_collection = torchmetrics.MetricCollection([
                torchmetrics.MeanAbsoluteError(),
                torchmetrics.MeanAbsolutePercentageError(),
                torchmetrics.MeanSquaredError(),
                metrics.RMSE(),
                torchmetrics.SymmetricMeanAbsolutePercentageError(),
                torchmetrics.WeightedMeanAbsolutePercentageError(),
                torchmetrics.ExplainedVariance(),
                torchmetrics.PearsonCorrCoef(),
                torchmetrics.R2Score(),
                torchmetrics.SpearmanCorrCoef(),
            ], prefix="metrics/")
            # loss
            self.criterion = nn.MSELoss(reduction="none")
            self.add_train_metrics = None
            self.add_val_metrics = None
            self.add_test_metrics = None
        else:
            avg = "weighted"
            metric_collection = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(num_classes=num_classes, average=avg),
                metrics.BalancedAccuracy(),
                torchmetrics.Recall(num_classes=num_classes, average=avg),
                torchmetrics.Precision(num_classes=num_classes, average=avg),
                torchmetrics.CalibrationError(),
                torchmetrics.CohenKappa(num_classes=num_classes),
                torchmetrics.F1Score(num_classes=num_classes, average=avg),
                # torchmetrics.FBetaScore(beta=1.0, num_classes=num_classes, average=avg),
                torchmetrics.MatthewsCorrCoef(num_classes=num_classes),
                torchmetrics.Specificity(num_classes=num_classes, average=avg),
            ], prefix="metrics/")
            additional_metrics = torchmetrics.MetricCollection({
                "MacroAccuracy": torchmetrics.Accuracy(num_classes=3, average="macro"),
            }, prefix="add_metrics/")
            # loss
            self.criterion = nn.CrossEntropyLoss(reduction="none")
            self.add_train_metrics = additional_metrics.clone(postfix="/train")
            self.add_val_metrics = additional_metrics.clone(postfix="/val")
            self.add_test_metrics = additional_metrics.clone(postfix="/test")

        self.train_metrics = metric_collection.clone(postfix="/train")
        self.val_metrics = metric_collection.clone(postfix="/val")
        self.test_metrics = metric_collection.clone(postfix="/test")

        if "testing" in config["setup"]["mode"]:
            self.optimizer = None
            self.scheduler = None
        else:
            # load pretrained model parameters
            new_state_dict = load_model_weights(self, config)
            if new_state_dict:
                self.load_state_dict(new_state_dict)

            # set optimizer
            self.optimizer = self._set_optimizer()
            self.scheduler = self._set_scheduler()


    def _add_weight_decay(self, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]


    def _set_optimizer(self):
        config = self.config

        # weight decay
        weight_decay = config["optimisation"]["weight_decay"]
        assert 0 <= weight_decay <= 1, "Weight Decay needs to be between 0 and 1!"

        if weight_decay != 0:
            if config["optimisation"]["decouple_weight_decay"]:
                # https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8#:~:text=where%20weight_decay%20is%20a%20hyperparameter,all%20the%20updates%20for%20you.
                # https://arxiv.org/abs/1711.05101
                # lambda = lambda_norm * sqrt(batch_size / (num_samples * epochs))
                batch_size = config["setup"]["batch_size"]
                epochs = config["setup"]["epochs"]
                num_samples = config["setup"]["num_train_samples"]
                weight_decay = weight_decay * np.sqrt(batch_size / (num_samples * epochs))

            parameters = self._add_weight_decay(weight_decay)
            weight_decay = 0.
        else:
            parameters = self.parameters()

        if config['setup']['finetuning']:
            print('freezing all layers except mlp head')
            for param in self.parameters():
                param.requires_grad = False
            for param in self.mlp_head.parameters():
                param.requires_grad = True
            parameters = filter(lambda p: p.requires_grad, parameters)

        if config['optimisation']['optimiser']=='Adam':
            print('using Adam optimiser')
            return optim.Adam(parameters, 
                lr=self.learning_rate, 
                weight_decay=weight_decay)
        elif config['optimisation']['optimiser']=='SGD':
            print('using SGD optimiser')
            return optim.SGD(parameters, 
                lr=self.learning_rate, 
                weight_decay=weight_decay,
                momentum=config['SGD']['momentum'], nesterov=config['SGD']['nesterov'])
        elif config['optimisation']['optimiser']=='AdamW':
            print('using AdamW optimiser')
            return optim.AdamW(parameters,
                lr=self.learning_rate,
                weight_decay=weight_decay)
        else:
            raise('not implemented yet')


    def _set_scheduler(self):
        config = self.config
        optimizer = self.optimizer
        scheduler = None
        epochs = config["setup"]["epochs"]

        if config['optimisation']['use_scheduler']:
            print('Using learning rate scheduler')
            if config['optimisation']['scheduler'] == 'StepLR':
                scheduler = StepLR(optimizer=optimizer,
                                    step_size= config['StepLR']['stepsize'],
                                    gamma=config['StepLR']['decay'])

            elif config['optimisation']['scheduler'] == 'CosineDecay':
                scheduler = CosineAnnealingLR(optimizer,
                                            T_max = config['CosineDecay']['T_max'],
                                            eta_min=self.learning_rate/10,
                                            )

            elif config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
                scheduler = ReduceLROnPlateau(optimizer,
                                                factor=0.5,
                                                patience=250,
                                                cooldown=0,
                                                min_lr=0.000001
                                            )

            if config['optimisation']['warmup']:
                scheduler = GradualWarmupScheduler(optimizer,
                multiplier=1, 
                total_epoch=config['optimisation']['nbr_step_warmup'], 
                after_scheduler=scheduler)
        else:
            # to use warmup without fancy scheduler
            if config['optimisation']['warmup']:
                scheduler = StepLR(optimizer,
                                    step_size=epochs)
                scheduler = GradualWarmupScheduler(optimizer,
                                                multiplier=1, 
                                                total_epoch=config['optimisation']['nbr_step_warmup'], 
                                                after_scheduler=scheduler)
        
        return scheduler

           
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        return self.mlp_head(x)


    def _step(self, batch, add_noise=False):
        y, img = batch

        if add_noise:
            img = self._add_noise(img)

        y_hat = self(img)

        if self.config["transformer"]["num_classes"] != 1:
            y = y.reshape(-1)
        else:
            y_hat = y_hat.reshape(-1)
            y = y.float()

        loss = self.criterion(y_hat, y)
        return loss, y_hat, y


    def _add_noise(self, img):
        noise = torch.empty(img.shape, device=img.device).normal_(mean=0, std=self.config["setup"]["noise_scale"])
        data = img + noise
        means = torch.mean(torch.mean(torch.mean(data, axis=3), axis=2), axis=0)
        stds = torch.std(torch.std(torch.std(data, axis=3), axis=2), axis=0)
        num_channels = img.shape[1]
        img = (data - means.reshape(1,num_channels,1,1))/stds.reshape(1,num_channels,1,1)

        return img


    def training_step(self, batch, batch_idx):       
        loss, y_hat, y = self._step(batch, self.config["setup"]["add_noise"])

        return {"loss": loss.mean(), "preds": y_hat, "target": y}


    def training_step_end(self, step_output):
        self.log("loss/train", step_output["loss"].item())
        output = self.train_metrics(step_output["preds"], step_output["target"])

        if self.add_train_metrics:
            output1 = self.add_train_metrics(step_output["preds"], step_output["target"])
            self.log_dict(output1)

        self.log_dict(output)

        return super().training_step_end(step_output)


    def validation_step(self, batch, batch_idx):
        _, y, img = batch
        loss, y_hat, y = self._step((y, img))

        return {"val_loss": loss.mean(), "loss_list": loss, "preds": y_hat, "target": y}
        

    def validation_step_end(self, step_output):
        if self.trainer.sanity_checking:
            return super().validation_step_end(step_output)
        self.log("loss/val", step_output["val_loss"].item())
        output = self.val_metrics(step_output["preds"], step_output["target"])

        if self.add_val_metrics:
            output1 = self.add_val_metrics(step_output["preds"], step_output["target"])
            self.log_dict(output1)

            # log confusion matrix
            preds = torch.argmax(step_output["preds"], axis=1)
            cm = wandb.plot.confusion_matrix(
                y_true=step_output["target"].cpu().numpy(),
                preds=preds.cpu().numpy(),
                class_names=self.encoder.classes_)
            wandb.log({"conf_mat": cm})

        self.log_dict(output)

        return super().validation_step_end(step_output)


    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)

        return {"test_loss": loss.mean(), "preds": y_hat, "target": y}


    def test_step_end(self, step_output):
        self.log("loss/test", step_output["test_loss"].item())
        output = self.test_metrics(step_output["preds"], step_output["target"])

        if self.add_test_metrics:
            output1 = self.add_test_metrics(step_output["preds"], step_output["target"])
            self.log_dict(output1)

        self.log_dict(output)

        return super().test_step_end(step_output)


    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer
        else:
            return dict(
                optimizer=self.optimizer,
                lr_scheduler=dict(
                    scheduler=self.scheduler,
                    interval='step',
                    monitor="loss/train",
                )
            )

