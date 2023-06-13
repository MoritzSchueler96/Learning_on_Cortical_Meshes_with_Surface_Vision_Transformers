from pytorch_lightning.callbacks import Callback
import wandb
import torch
import torch.nn as nn
import pandas as pd
import os

class MyPrintingCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_fit_end(self, trainer, pl_module):
        print("Finished training")

    def on_test_start(self, trainer, pl_module):
        print("Start to test")

    def on_test_end(self, trainer, pl_module):
        print("Finished testing")


class LogPredictionsCallback(Callback):
    
    def __init__(self):
        # init df and table to store values
        self._columns = ["id", "target", "prediction", "loss", "diff"]
        self._table = wandb.Table(columns=self._columns)
        self._log_df = pd.DataFrame(columns=self._columns)
        # log softmax logits in results df

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        if trainer.sanity_checking:
            return

        id, _, _ = batch
        y_hat = outputs["preds"]
        y = outputs["target"]
        loss = outputs["loss_list"]
        preds = torch.argmax(y_hat, axis=1)
        self.log_test_predictions(id, preds, y, loss)

    def log_test_predictions(self, id, predictions, labels, loss):
        # obtain confidence scores for all classes
        log_id = id
        log_preds = predictions.detach().cpu().numpy()
        log_labels = labels.detach().cpu().numpy()
        log_loss = loss.detach().cpu().numpy()
        log_diff = (labels-predictions).abs().detach().reshape(-1).cpu().numpy()

        for i, y, p, l, d in zip(log_id, log_labels, log_preds, log_loss, log_diff):
            # add required info to data table:
            # id, label, model's guess, loss, diff
            self._table.add_data(i, y, p, l, d)
            self._log_df.loc[len(self._log_df.index)] = [i, y, p, l, d]

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            return
        # log tables
        self._log_tables(trainer)

        self._table = wandb.Table(columns=self._columns)
        self._log_df = pd.DataFrame(columns=self._columns)


    def _log_tables(self, trainer):

        current_epoch = trainer.current_epoch
        artifact = wandb.Artifact("validation_samples_" + str(wandb.run.id), type="predictions")
        artifact.add(self._table, "predictions")
        artifact.description = str(current_epoch)
        wandb.run.log_artifact(artifact)

        # statistics about predictions
        wandb.log({"Statistics/model": self._log_df.describe(percentiles=[.25, .5, .75, .9, .95, .99]).reset_index()})

        wandb.log({"predictions_df": self._log_df})

        df_sort = self._log_df.sort_values("loss", ascending=False)

        wandb.log({"worst samples": df_sort[0:5]})
        wandb.log({"best samples": df_sort[-5:]})

        # wandb.log({"loss": 0.314, "epoch": 5,
        #    "inputs": wandb.Image(inputs),
        #    "logits": wandb.Histogram(ouputs),
        #    "captions": wandb.Html(captions)})

        # Guess
        # validation_df['guess'] = predictions
        # validation_df = validation_df.replace({"guess": id_to_labels})

        # # Logits
        # score_columns = [f'{label}_score' for label in labels_dict.keys()]
        # validation_df[score_columns] = raw_preds

        # # Initialize new artifact
        # prediction_at = wandb.Artifact('val_predictions_effnetb4', type="val_prediction")

        # columns=["id", "image", "truth", "guess"]
        # columns.extend(score_columns)
        # preview_dt = wandb.Table(columns=columns)

        # for i in tqdm(range(len(validation_df))):
        #     image_name = validation_df.loc[i]['image']
        #     label = validation_df.loc[i]['labels']
        #     guess = validation_df.loc[i]['guess']
            
        #     full_path = TRAIN_PATH+image_name
            
        #     row = [image_name, wandb.Image(full_path), id_to_labels[label], guess]
        #     for score in validation_df.loc[i][3:]:
        #         row.append(score)
                
        #     preview_dt.add_data(*row)
        # save artifact to W&B
        # prediction_at.add(preview_dt, "val_prediction")
        # run.log_artifact(prediction_at)
        

class LogLossStatsCallback(Callback):
    
    def __init__(self):
        self._num_diff1 = 0
        self._num_diff3 = 0
        self._num_diff5 = 0
        self._num_total = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
 
        if trainer.sanity_checking:
            return

        loss = outputs["loss_list"]
        self._log_loss_percentages(loss, pl_module)

    def _log_loss_percentages(self, loss, pl_module):
        if type(nn.MSELoss()) == type(pl_module.criterion):
            loss = torch.sqrt(loss)
        self._num_diff1 += sum(loss<=1).detach().cpu().numpy()
        self._num_diff3 += sum(loss<=3).detach().cpu().numpy()
        self._num_diff5 += sum(loss<=5).detach().cpu().numpy()
        self._num_total += loss.numel()

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            return
        # log tables
        self._log_table(trainer)

        self._num_diff1 = 0
        self._num_diff3 = 0
        self._num_diff5 = 0
        self._num_total = 0

    def _log_table(self, trainer):
        cols = ["count_diff1", "percent_diff1", "count_diff3", "percent_diff3", "count_diff5", "percent_diff5", "count_total"]
        df = pd.DataFrame(columns=cols)
        num_total = self._num_total
        num_diff1 = self._num_diff1
        percent_diff1 = num_diff1 / num_total
        num_diff3 = self._num_diff3
        percent_diff3 = num_diff3 / num_total
        num_diff5 = self._num_diff5
        percent_diff5 = num_diff5 / num_total
        df.loc[len(df.index)] = [num_diff1, percent_diff1, num_diff3, percent_diff3, num_diff5, percent_diff5, num_total]
        
        wandb.log({"Statistics/loss_percentages": df})
