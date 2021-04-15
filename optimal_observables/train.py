import pytorch_lightning as pl

from argparse import ArgumentParser
from torch.utils.data import DataLoader

from data import ConditionedObservablesFC
from models import FullyConnected


train_dataset = ConditionedObservablesFC(
    reconstructions_path="../reconstructed_events/SM_spin-ON_100k_0",
    low_exp=-2,
    high_exp=2,
    delta_exp=1,
)
train_dataloader = DataLoader(train_dataset, batch_size=1)

parser = ArgumentParser()
parser = FullyConnected.add_model_specific_args(parser)
hparams = parser.parse_args()
hparams.input_size = train_dataset.n_observables
hparams.output_size = train_dataset.n_samples
model = FullyConnected(hparams)
trainer = pl.Trainer()

trainer.fit(model, train_dataloader)
