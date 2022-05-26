import os

import numpy as np
import pandas as pd
import torch
from pysr import best, pysr
from rich.console import Console
from rich.progress import track
from torch.utils.data import DataLoader

from config import dataset_config
from optimal_observables.optimization.data import ClassifierDataset
from optimal_observables.optimization.models import NNClassifier

if __name__ == "__main__":
    ckpts_base_path = "../data/model_ckpts"
    process_name = "ON-OFF"
    run_id = "1c1ba212e7d044f5b5d7c7b82c7fdcda"
    ckpt_path = os.path.join(ckpts_base_path, process_name, f"model_{run_id}.ckpt")

    console = Console()

    model = NNClassifier.load_from_checkpoint(ckpt_path)
    exponents = model.exponents_layer.weight.flatten().tolist()

    dataset = ClassifierDataset(**dataset_config)
    loader = DataLoader(dataset, batch_size=32)

    all_learned_observables = list()
    all_input_observables = list()
    with torch.no_grad():
        for X_batch, y_batch in track(loader):
            learned_observables_batch = model.observables_generator(X_batch)
            all_input_observables.append(X_batch)
            all_learned_observables.append(learned_observables_batch)

    log_input_observables = torch.cat(all_input_observables, dim=0).numpy()
    log_learned_observables = torch.cat(all_learned_observables, dim=0).numpy()

    input_observables = np.exp(log_input_observables) - 2
    learned_observables = np.exp(log_learned_observables)
    random_sample = np.random.choice(len(input_observables), size=1000, replace=False)

    best_equations = list()
    for obs_idx in track(range(learned_observables.shape[1])):
        equations = pysr(
            input_observables[random_sample],
            learned_observables[random_sample, obs_idx],
            niterations=5,
            binary_operators=["+", "*"],
            variable_names=[
                "cos_k1",
                "cos_k2",
                "cos_r1",
                "cos_r2",
                "cos_n1",
                "cos_n2",
            ],
        )
        learned_equation = best(equations)
        console.print(learned_equation)
        best_equations.append(learned_equation)

    results_df = pd.DataFrame()
    results_df["observable"] = best_equations
    results_df["exponent"] = exponents

    output_dir = os.path.join(
        os.path.dirname(ckpts_base_path),
        "learned_observables",
    )
    os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(
        os.path.join(output_dir, f"learned_observable_{run_id}.csv"), index=False
    )
