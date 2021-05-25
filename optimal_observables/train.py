import pytorch_lightning as pl

from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split

from data import ConditionedObservablesFC
from models import FullyConnected

reco_paths = [
    "../reconstructed_events/SM_spin-ON_100k_0",
    "../reconstructed_events/SM_spin-ON_100k_20210425"
]
n_out_samples = 10000

full_dataset = ConditionedObservablesFC(
    reconstructions_paths=reco_paths,
    n_out_samples=n_out_samples,
    low_exp=-5,
    high_exp=5,
    n_exp=1000,
)
n_train = int(len(full_dataset) * 0.7)
n_val = len(full_dataset) - n_train
train_dataset, val_dataset = random_split(
    dataset=full_dataset,
    lengths=[n_train, n_val]
)

train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4)

parser = ArgumentParser()
parser = FullyConnected.add_model_specific_args(parser)
hparams = parser.parse_args()
hparams.input_size = full_dataset.n_observables
hparams.output_size = n_out_samples
biases = full_dataset.biases
model = FullyConnected(hparams=hparams, biases=biases)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    dirpath='model_ckpts/',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
)
early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
   monitor='val_loss',
   min_delta=0.00,
   patience=3,
   verbose=False,
   mode='min'
)
trainer = pl.Trainer(max_epochs=10000, callbacks=[checkpoint_callback, early_stop_callback])

trainer.fit(model, train_dataloader, val_dataloader)
