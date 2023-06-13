import os

import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader, Dataset
from cv_utils.utils.preprocessing_struct_dHCP import preprocess
from cv_utils.utils.utils import seed_worker

import torch
import numpy as np
import wandb
from dotenv import load_dotenv
from fitter import Fitter, get_common_distributions
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from PIL import Image
import io

load_dotenv()

BASE_REPO_FOLDER = os.getenv("BASE_REPO_FOLDER")
CONFIG_DIR = os.path.join(BASE_REPO_FOLDER, "code/cv_utils/configs/")


class SViTDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return self.labels[index], self.features[index]

    def __len__(self):
        return len(self.features)


class SViTDataModule(pl.LightningDataModule):
    """
    Data module for the Surface Vision Transformer model
    """
    def __init__(self, config=None, num_workers=4):
        super().__init__()

        if config is None:
            config_path = os.path.join(CONFIG_DIR, "training.yml")
            
            with open(config_path) as f:
                config = yaml.safe_load(f)

            config["use_default"] = True
            
        num_classes = config["transformer"]["num_classes"]
        architecture = config["setup"]["architecture"]
        config["transformer"] = config["transformer"][architecture]
        config["transformer"]["num_classes"] = num_classes

        if architecture in config["weights"]["imagenet"].keys():
            config["weights"]["imagenet"] = config["weights"]["imagenet"][architecture]
        else:
            config["weights"]["imagenet"] = None

        if architecture in config["weights"]["wandb"].keys():
            config["weights"]["wandb"] = config["weights"]["wandb"][architecture]
        else:
            config["weights"]["wandb"] = None
            
        self.batch_size = config["setup"]["batch_size"]
        if "batch_size_val" in config["setup"].keys():
            self.batch_size_val = config["setup"]["batch_size_val"]
        else:
            self.batch_size_val = config["setup"]["batch_size"]
        
        config["data"]["logged"] = False
        self.num_workers = num_workers
        self.config = config

        self.trainset = None
        self.valset = None
        self.testset = None

        if num_classes > 1:
            self.encoder = LabelEncoder()
        else:
            self.encoder = None

    def prepare_data(self):        
        config = self.config
        task = config["data"]["task"]
        configuration = config['data']['configuration']
        data_path = os.path.join(BASE_REPO_FOLDER, config["data"]["data_path"].format(task,configuration))

        config_path = os.path.join(CONFIG_DIR, "preprocessing.yml")
        with open(config_path) as f:
            preprocess_config = yaml.safe_load(f)

        if not os.path.exists(os.path.join(data_path, "train_data.npy")):
            preprocess_config["data"]["task"] = config["data"]["task"]
            preprocess_config["data"]["configuration"] = config["data"]["configuration"]
            preprocess_config["data"]["split"] = "train"
            preprocess(preprocess_config)
            
        if not os.path.exists(os.path.join(data_path, "validation_data.npy")):
            preprocess_config["data"]["task"] = config["data"]["task"]
            preprocess_config["data"]["configuration"] = config["data"]["configuration"]
            preprocess_config["data"]["split"] = "validation"
            preprocess(preprocess_config)
            
        if not os.path.exists(os.path.join(data_path, "test_data.npy")):
            preprocess_config["data"]["task"] = config["data"]["task"]
            preprocess_config["data"]["configuration"] = config["data"]["configuration"]
            preprocess_config["data"]["split"] = "test"
            preprocess(preprocess_config)

    def setup(self, stage=None):
        if not (self.trainset and self.valset and self.testset): 
            self.trainset, self.valset, self.testset = self._create_datasets()

    def _create_datasets(self):
        """
        creates a train, validation and test dataset.
        The trainset will be shuffeled.
        """
        config = self.config
        task = config["data"]["task"]
        configuration = config['data']['configuration']
        data_path = os.path.join(BASE_REPO_FOLDER, config["data"]["data_path"].format(task,configuration))
        
        train_data = np.load(os.path.join(data_path,'train_data.npy'))
        train_label = np.load(os.path.join(data_path,'train_labels.npy'), allow_pickle=True)

        val_data = np.load(os.path.join(data_path,'validation_data.npy'))
        val_label = np.load(os.path.join(data_path,'validation_labels.npy'), allow_pickle=True)

        test_data = np.load(os.path.join(data_path,'test_data.npy'))
        test_label = np.load(os.path.join(data_path,'test_labels.npy'), allow_pickle=True)

        print('training data: {}'.format(train_data.shape))
        print('validation data: {}'.format(val_data.shape))
        print('testing data: {}'.format(test_data.shape))

        # initialize encoder using training data
        if self.encoder:
            self.encoder.fit(train_label.reshape(-1,1))
            train_label = self.encoder.transform(train_label.reshape(-1,1))
            val_label[:,1] = self.encoder.transform(val_label[:,1].reshape(-1,1)).reshape(-1)
            test_label = self.encoder.transform(test_label.reshape(-1,1))
            
        if not config["data"]["logged"]:
            if config["data"]["log_statistics"]:
                self._log([train_label, val_label[:,1], test_label])

            if config["data"]["log_data"] and True:
                artifact = wandb.Artifact(name=task+'_dataset', type='dataset')
                artifact.add_dir(local_path=data_path)
                wandb.run.log_artifact(artifact)
            elif True:
                artifact = wandb.Artifact(name=task+'_datasplit', type='dataset')
                file_path = os.path.join(data_path, 'train_ids.npy')
                artifact.add_file(local_path=file_path)
                file_path = os.path.join(data_path, 'validation_ids.npy')
                artifact.add_file(local_path=file_path)
                file_path = os.path.join(data_path, 'test_ids.npy')
                artifact.add_file(local_path=file_path)
                wandb.run.log_artifact(artifact)

        config["data"]["logged"] = True

        return (
            SViTDataset(train_data,
            train_label),
            SViTDataset(val_data,
                        val_label),
            SViTDataset(test_data,
                        test_label),
        )

    def collate_ids(self, batch):
        labels, features = zip(*batch)

        ids, labels = zip(*labels)
        ids = np.asarray(ids) # not needed necessarily
        labels = np.asarray(labels) # not needed necessarily
        features = np.asarray(features) # needed for speed up

        return (
            ids,
            torch.as_tensor(labels),
            torch.FloatTensor(features),
        )

    def collate(self, batch):
        labels, features = zip(*batch)

        labels = np.asarray(labels) # not needed necessarily
        features = np.asarray(features) # needed for speed up

        return (
            torch.as_tensor(labels),
            torch.FloatTensor(features),
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            shuffle=False,
            batch_size=self.batch_size_val,
            collate_fn=self.collate_ids,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            shuffle=False,
            batch_size=self.batch_size_val,
            collate_fn=self.collate,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

    def _fig2img(self, fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def _create_statistics(self, data, population):
        z = np.array([1.28, 1.44, 1.65, 1.96, 2.58])
        conf = [80, 85, 90, 95, 99]

        mean = data.mean()
        std = data.std()

        std_error = std / np.sqrt(population)
        margin_of_error = z * std_error
        conf_plus = mean + margin_of_error
        conf_minus = mean - margin_of_error
        error_rate = margin_of_error / mean * 100

        columns = ["Confidence", "Mean", "LowerBound", "UpperBound", "Margin_of_Error", "Error_Prc", "Std_Error", "Std"]
        df = pd.DataFrame(columns=columns)

        for c, cm, cf, moe, e in zip(conf, conf_minus, conf_plus, margin_of_error, error_rate):
            df.loc[len(df.index)] = [c, mean, cm, cf, moe, e, std_error, std]

        return df

    def _create_plots(self, data, population, split):
        fig = plt.figure()
        binwidth = 1

        mean = data.mean()
        std = data.std()
        std_error = std / np.sqrt(population)
        
        freq, bins, patches = plt.hist(data, edgecolor='white', label='d', bins=np.arange(min(data)-0.5, max(data) + 0.5 + binwidth, binwidth))

        # x coordinate for labels
        bin_centers = np.diff(bins)*0.5 + bins[:-1]

        n = 0
        for fr, x in zip(freq, bin_centers):
            height = int(fr)
            plt.annotate("{}".format(height),
                        xy = (x, height),             # top left corner of the histogram bar
                        xytext = (0,0.2),             # offsetting label position above its bar
                        textcoords = "offset points", # Offset (in points) from the *xy* value
                        ha = 'center', va = 'bottom'
                        )
            n = n+1

        # plt.legend()
        y_pos = int(max(freq))
        x_pos = int(max(bin_centers)) - 7
        plt.annotate(f"Mean: {mean:.2f}", xy=(x_pos, y_pos))
        plt.annotate(f"Std: {std:.2f}", xy=(x_pos, int(0.95*y_pos)))
        plt.annotate(f"Std Error: {std_error:.2f}", xy=(x_pos, int(0.9*y_pos)-1))
        plt.title(split.title() + " Data distribution")
        img = self._fig2img(fig)
        plt.clf()

        return img

    def _create_fitted_dists(self, data, split):
        fig = plt.figure()
        distributions_set = get_common_distributions()
        distributions_set.extend(['alpha', 'arcsine', 'cosine', 'invgauss', 'invgamma', 'f', 't', 'pareto', 'exponnorm', "exponweib", "pareto", "genextreme"]) 
        f = Fitter(data, distributions=distributions_set) 
        f.fit()
        df = f.summary(plot=False)
        df = df.reset_index().rename(columns={"index": "distribution"})

        f.hist()
        f.plot_pdf()
        plt.title(split.title() + " Data distribution")
        img = self._fig2img(fig)
        plt.clf()

        return df, img

    def _log(self, data):
        images = []
        data = [d[int(len(d)/2):] for d in data]
        pop = sum([d.shape[0] for d in data])
        data.append(np.concatenate(data))
        splits = ["train", "val", "test", "all"]

        for d, split in zip(data, splits):
            print(split.title())
            d = d.astype('float64')
            if not self.encoder:
                df = self._create_statistics(d, pop)
                wandb.log({"Statistics/" + split: wandb.Table(dataframe=df)})
            img = self._create_plots(d, pop, split)
            df2, img2 = self._create_fitted_dists(d, split)

            images.append(img)
            images.append(img2)
            wandb.log({"Statistics/Distribution Parameters/" + split: wandb.Table(dataframe=df2)})

        wandb.log({"Statistics/Distributions": [wandb.Image(image) for image in images]})
