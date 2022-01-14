from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import auroc


class FullyConnected(pl.LightningModule):
    def __init__(self, hparams, biases):
        super(FullyConnected, self).__init__()
        self.hparams = hparams
        self.save_hyperparameters()

        self.input = nn.Linear(
            in_features=hparams.input_size, out_features=hparams.hidden1_size
        )
        self.hidden1 = nn.Linear(
            in_features=hparams.hidden1_size, out_features=hparams.hidden2_size
        )
        self.output = nn.Linear(
            in_features=hparams.hidden2_size, out_features=hparams.output_size
        )
        self.output.bias = torch.nn.Parameter(torch.tensor(biases))

    def forward(self, x):
        x = self.input(x)
        x = F.leaky_relu(x)
        x = self.hidden1(x)
        x = F.leaky_relu(x)
        x = self.output(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y, y_pred)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y, y_pred)
        self.log("val_loss", loss)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--hidden1_size", type=int, default=250)
        parser.add_argument("--hidden2_size", type=int, default=100)
        return parser


class NNClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super(NNClassifier, self).__init__()
        self.save_hyperparameters(hparams)

        self.input_layer = nn.Linear(
            in_features=self.hparams["input_size"],
            out_features=self.hparams["hidden_size"],
        )
        self.feature_layer = nn.Linear(
            in_features=self.hparams["hidden_size"],
            out_features=self.hparams["n_learned_observables"],
        )
        self.observables_generator = nn.Sequential(
            self.input_layer,
            nn.Tanh(),
            nn.Linear(self.hparams["hidden_size"], self.hparams["hidden_size"]),
            nn.Tanh(),
            nn.Linear(self.hparams["hidden_size"], self.hparams["hidden_size"]),
            nn.Tanh(),
            self.feature_layer,
        )

        self.output_layer = nn.Linear(
            in_features=self.hparams["n_learned_observables"],
            out_features=1,
            bias=False,
        )

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input_x):
        x = self.observables_generator(input_x)
        x = self.output_layer(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y.float())
        y_pred_proba = torch.sigmoid(y_pred)
        roc_auc_score = auroc(y_pred_proba, y.int(), pos_label=1)
        self.log("val_loss", loss)
        self.log("val_roc_auc", roc_auc_score)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--hidden_size", type=int, default=128)
        parser.add_argument("--n_learned_observables", type=int, default=6)
        return parser
