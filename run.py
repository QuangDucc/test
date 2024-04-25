import os

import comet_ml
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

PATH_DATASETS = "//Users/vinbrain/Desktop/test/data"
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64


class Model(pl.LightningModule):
    def __init__(self, layer_size=784):
        super().__init__()
        
        self.l1 = torch.nn.Linear(layer_size, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.logger.log_metrics({"train_loss": loss}, step=batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.logger.log_metrics({"val_loss": loss}, step=batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# Init our model
model = Model()

# Arguments made to CometLogger are passed on to the comet_ml.Experiment class
comet_logger = CometLogger(project_name="comet-examples-lightning",api_key="aYFiDzmnZgAaAo2cTVDC1omD8")

# Log parameters
comet_logger.log_hyperparams({"batch_size": BATCH_SIZE})

# Init DataLoader from MNIST Dataset
train_ds = MNIST(
    PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

eval_ds = MNIST(
    PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor()
)
eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# Initialize a trainer
trainer = Trainer(max_epochs=3, logger=comet_logger)

# Train the model ⚡
trainer.fit(model, train_loader, eval_loader)