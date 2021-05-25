import os
import numpy as np

from typing import List

from torch.utils.data import Dataset

import observables


class ConditionedObservablesFC(Dataset):
    def __init__(
        self,
        reconstructions_paths: List[str],
        n_out_samples: int,
        low_exp: float,
        high_exp: float,
        n_exp: int,
        rnd_seed=202094
    ):
        reco_names = [
            "p_top",
            "p_l_t",
            "p_b_t",
            "p_nu_t",
            "p_tbar",
            "p_l_tbar",
            "p_b_tbar",
            "p_nu_tbar",
            "idx",
            "weight",
        ]

        recos = dict()

        batches = {name: [] for name in reco_names}
        for reconstructions_path in reconstructions_paths:
            for batch_idx in range(10):
                for name in reco_names:
                    batches[name].append(
                        np.load(
                            os.path.join(
                                reconstructions_path, f"{name}_batch_{batch_idx}.npy"
                            )
                        )
                    )
        recos = {
            name: np.concatenate(batches, axis=0) for name, batches in batches.items()
        }
        matrix = observables.get_matrix(
            p_l_t=recos["p_l_t"],
            p_l_tbar=recos["p_l_tbar"],
            p_top=recos["p_top"],
            p_tbar=recos["p_tbar"],
        )

        n_keep = (matrix.shape[0] // n_out_samples) * n_out_samples
        trimmed_matrix = matrix[:n_keep, :]
        del matrix

        rng = np.random.default_rng(rnd_seed)
        n_observables = trimmed_matrix.shape[1]
        exps = rng.uniform(low_exp, high_exp, size=(n_exp, n_observables))

        exps_vec = np.repeat(exps, trimmed_matrix.shape[0], axis=0)
        matrix_vec = np.tile(trimmed_matrix, (exps.shape[0], 1))
        stable_matrix_vec = matrix_vec + 2
        conditioned_matrix_vec = np.prod(stable_matrix_vec ** exps_vec, axis=1)
        stable_conditioned_matrix_vec = np.log(conditioned_matrix_vec).astype(np.float32)

        self.n_observables = n_observables
        self.exps = exps.astype(np.float32)
        self.batched_conditioned_matrix = stable_conditioned_matrix_vec.reshape(
            -1, n_out_samples
        )
        self.biases = np.mean(self.batched_conditioned_matrix, axis=0)

    def __len__(self):
        return len(self.exps)

    def __getitem__(self, idx):
        return self.exps[idx], self.batched_conditioned_matrix[idx]
