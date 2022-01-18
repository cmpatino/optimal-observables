from argparse import ArgumentParser

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary
from torch.utils.data import DataLoader, random_split

from optimal_observables import classifier_config
from optimal_observables.data import ClassifierDataset
from optimal_observables.models import NNClassifier

full_dataset = ClassifierDataset(**classifier_config.dataset_config)
n_train = int(len(full_dataset) * 0.7)
n_val = len(full_dataset) - n_train
train_dataset, val_dataset = random_split(
    dataset=full_dataset, lengths=[n_train, n_val]
)

batch_size = 128
n_epochs = 100
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

parser = ArgumentParser()
parser = NNClassifier.add_model_specific_args(parser)
hparams = parser.parse_args()
hparams.input_size = full_dataset.input_features
hparams.n_epochs = n_epochs
hparams.batch_size = batch_size
model = NNClassifier(hparams=hparams)

mlf_logger = MLFlowLogger(
    experiment_name="ON-OFF-pl",
    tracking_uri="file:../data/mlruns/",
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_roc_auc",
    dirpath="../data/model_ckpts/ON-OFF",
    filename=f"model_{mlf_logger.run_id}",
    save_top_k=1,
    mode="min",
)
early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
    monitor="val_roc_auc", min_delta=0.00, patience=20, verbose=False, mode="max"
)
trainer = pl.Trainer(
    logger=mlf_logger,
    max_epochs=n_epochs,
    callbacks=[checkpoint_callback, early_stop_callback, RichModelSummary(), RichProgressBar()],
)

mlflow.set_experiment("ON-OFF")
mlflow.pytorch.autolog()
with mlflow.start_run(tags={"run-id": mlf_logger.run_id}) as run:
    mlflow.log_params(classifier_config.dataset_config)
    mlflow.log_params(vars(hparams))
    trainer.fit(model, train_dataloader, val_dataloader)
