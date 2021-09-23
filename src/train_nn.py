from argparse import ArgumentParser

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, random_split

from optimal_observables import config
from optimal_observables.data import ConditionedObservablesFC
from optimal_observables.models import FullyConnected

full_dataset = ConditionedObservablesFC(**config.dataset_config)
n_train = int(len(full_dataset) * 0.7)
n_val = len(full_dataset) - n_train
train_dataset, val_dataset = random_split(
    dataset=full_dataset, lengths=[n_train, n_val]
)

train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4)

parser = ArgumentParser()
parser = FullyConnected.add_model_specific_args(parser)
hparams = parser.parse_args()
hparams.input_size = full_dataset.n_observables
hparams.output_size = config.dataset_config["n_out_samples"]
biases = full_dataset.biases
model = FullyConnected(hparams=hparams, biases=biases)

mlf_logger = MLFlowLogger(
    experiment_name=f"{config.process}-pl",
    tracking_uri="file:../data/mlruns/",
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    dirpath=f"../data/model_ckpts/{config.process}",
    filename="model",
    save_top_k=1,
    mode="min",
)
early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
    monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min"
)
trainer = pl.Trainer(
    logger=mlf_logger,
    max_epochs=10000,
    callbacks=[checkpoint_callback, early_stop_callback],
)

mlflow.set_experiment(config.process)
mlflow.pytorch.autolog()
with mlflow.start_run(tags={"pl-source": mlf_logger.run_id}) as run:
    mlflow.log_params(config.dataset_config)
    mlflow.log_params(vars(hparams))
    trainer.fit(model, train_dataloader, val_dataloader)
