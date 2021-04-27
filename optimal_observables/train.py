import pytorch_lightning as pl

from argparse import ArgumentParser
from torch.utils.data import DataLoader

from data import ConditionedObservablesFC
from models import FullyConnected


train_dataset = ConditionedObservablesFC(
    reconstructions_path="../reconstructed_events/SM_spin-ON_100k_0",
    low_exp=-5,
    high_exp=5,
    n_exp=1000,
)
train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0)

parser = ArgumentParser()
parser = FullyConnected.add_model_specific_args(parser)
hparams = parser.parse_args()
hparams.input_size = train_dataset.n_observables
hparams.output_size = train_dataset.n_samples
biases = train_dataset.biases
model = FullyConnected(hparams=hparams, biases=biases)
trainer = pl.Trainer(max_epochs=10000)

trainer.fit(model, train_dataloader)
