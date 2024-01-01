import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import lightning as L
from torch import  optim
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import *
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from lightning.pytorch.tuner import Tuner
from pathlib import Path
import requests
import gzip
import pickle
import torchmetrics
import subprocess
import webbrowser
from pytorch_lightning.loggers import MLFlowLogger
import time
from mlflow import MlflowClient
import mlflow
import boto3
import botocore
import mlflow
import pandas as pd
import plotly.express as px

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import random


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]


class MNISTDataModule(L.LightningDataModule):
    url = "https://github.com/pytorch/tutorials/raw/main/_static/"
    filename = "mnist.pkl.gz"

    def __init__(self, path:Path|str, batch_size:int=65536):
        super().__init__()
        self.path = Path(path)/self.filename
        self.path.parent.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        self.save_hyperparameters()
    
    def prepare_data(self):
        if not self.path.exists():
            content = requests.get(self.url + self.filename).content
            self.path.open("wb").write(content)

    def setup(self, stage):
        with gzip.open(self.path, "rb") as f:
            ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")
        x_train, y_train, x_valid, y_valid, x_test, y_test = map(torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test)
        )
        if stage == 'fit':
            self.ds_train = MyDataset(x_train, y_train)
            self.ds_valid = MyDataset(x_valid, y_valid)
        if stage == 'test':
            self.ds_test = MyDataset(x_test, y_test)
        if stage == 'predict':
            self.ds_predict = MyDataset(torch.cat([x_train, x_valid, x_test]), torch.cat([y_train, y_valid, y_test]))

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.ds_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.ds_predict, batch_size=self.batch_size)


def is_running_in_jupyter_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except (ImportError, AttributeError):
        return False
    return True


def rtype(x):
    if isinstance(x, list):
        return [rtype(o) for o in x]
    elif isinstance(x, tuple):
        return tuple(rtype(o) for o in x)
    elif isinstance(x, dict):
        return {k: rtype(v) for k,v in x.items()}
    else:
        type_str = str(type(x)).split("\'")[1]
        if isinstance(x, (Tensor, np.ndarray)):
            return f'{type_str}={x.shape}'
        else: 
            return type_str

            
def get_most_recently_modified_file(path: Path | str, glob: str) -> Path:
    most_recent_time = 0
    most_recent_modified_file = None
    for p in Path(path).rglob(glob):
        if p.stat().st_mtime > most_recent_time:
            most_recent_time = p.stat().st_mtime
            most_recent_modified_file = p
    return most_recent_modified_file


def launch_mlflow_ui(uri: str, run) -> None:
    subprocess.call("ps aux | grep 'mlflow.server' | awk '{print $2}' | xargs kill", shell=True)
    command = f'mlflow ui --host 0.0.0.0 --port 5050 --backend-store-uri {uri}'
    process = subprocess.Popen(command.split())
    url = 'http://localhost:5050/'
    if run:
        url += f'#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}'
    time.sleep(2)
    webbrowser.open(url)
    return process
    
root = Path('.').absolute().parent if is_running_in_jupyter_notebook else Path(__file__).absolute().parent.parent
dir_tmp = root/'tmp'
dir_mlruns = dir_tmp/'mlruns'
(dir_mlruns/'.trash').mkdir(exist_ok=True, parents=True)
uri_mlruns = "file://"+(dir_mlruns).as_posix()
dir_artifacts = dir_tmp/'artifacts'
dir_artifacts.mkdir(exist_ok=True)
dir_mnist = dir_tmp/'vector-mnist'


