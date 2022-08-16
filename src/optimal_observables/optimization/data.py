import os
from typing import Dict, List

import numpy as np
from torch.utils.data import Dataset

from optimal_observables.optimization import observables


def load_numpy_recos(reconstruction_paths: List[str]) -> Dict[str, np.ndarray]:
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
    for reconstruction_path in reconstruction_paths:
        for batch_idx in range(10):
            for name in reco_names:
                batches[name].append(
                    np.load(
                        os.path.join(
                            reconstruction_path, f"{name}_batch_{batch_idx}.npy"
                        )
                    )
                )
    recos = {name: np.concatenate(batches, axis=0) for name, batches in batches.items()}
    return recos


class ConditionedObservablesFC(Dataset):
    def __init__(
        self,
        reconstructions_paths: List[str],
        n_out_samples: int,
        low_exp: float,
        high_exp: float,
        n_exp: int,
        only_cosine_terms=False,
        rnd_seed=202094,
    ):
        recos = load_numpy_recos(reconstruction_paths=reconstructions_paths)
        matrix = observables.get_matrix(
            p_l_t=recos["p_l_t"],
            p_l_tbar=recos["p_l_tbar"],
            p_top=recos["p_top"],
            p_tbar=recos["p_tbar"],
            only_cosine_terms=only_cosine_terms,
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
        stable_conditioned_matrix_vec = np.log(conditioned_matrix_vec).astype(
            np.float32
        )

        self.n_observables = n_observables
        self.batched_exps = np.repeat(
            exps.astype(np.float32), n_keep // n_out_samples, axis=0
        )
        self.batched_conditioned_matrix = stable_conditioned_matrix_vec.reshape(
            -1, n_out_samples
        )
        self.biases = np.mean(self.batched_conditioned_matrix, axis=0)

    def __len__(self):
        return len(self.batched_exps)

    def __getitem__(self, idx):
        return self.batched_exps[idx], self.batched_conditioned_matrix[idx]


class ClassifierDataset(Dataset):
    def __init__(
        self,
        pos_reconstruction_paths: List[str],
        neg_reconstruction_paths: List[str],
        only_cosine_terms: bool = True,
    ):
        pos_recos = load_numpy_recos(reconstruction_paths=pos_reconstruction_paths)
        neg_recos = load_numpy_recos(reconstruction_paths=neg_reconstruction_paths)
        pos_matrix = observables.get_matrix(
            p_l_t=pos_recos["p_l_t"],
            p_l_tbar=pos_recos["p_l_tbar"],
            p_top=pos_recos["p_top"],
            p_tbar=pos_recos["p_tbar"],
            only_cosine_terms=only_cosine_terms,
        )
        neg_matrix = observables.get_matrix(
            p_l_t=neg_recos["p_l_t"],
            p_l_tbar=neg_recos["p_l_tbar"],
            p_top=neg_recos["p_top"],
            p_tbar=neg_recos["p_tbar"],
            only_cosine_terms=only_cosine_terms,
        )

        X_raw = np.concatenate([pos_matrix, neg_matrix], axis=0)
        self.X = np.log(X_raw + 2)
        self.y = np.concatenate(
            [np.ones(len(pos_matrix)), np.zeros(len(neg_matrix))]
        ).reshape(-1, 1)
        self.input_features = self.X.shape[1]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index].astype(np.float32), self.y[index].astype(np.int64)
