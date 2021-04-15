import os
import numpy as np


from torch.utils.data import Dataset

import observables


class ConditionedObservablesFC(Dataset):
    def __init__(
        self,
        reconstructions_path: str,
        low_exp: float,
        high_exp: float,
        delta_exp: float,
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

        n_observables = matrix.shape[1]
        n_steps = int((high_exp - low_exp) / delta_exp) + 1
        x = np.linspace(low_exp, high_exp, n_steps)
        combinations_mesh = np.array(np.meshgrid((x,) * n_observables))
        exps = combinations_mesh.T.reshape(-1, n_observables)

        exps_vec = np.repeat(exps, matrix.shape[0], axis=0)
        matrix_vec = np.tile(matrix, (exps.shape[0], 1))
        conditioned_matrix_vec = np.prod(matrix_vec ** exps_vec, axis=1).astype(np.float32)

        self.n_observables = n_observables
        self.n_samples = matrix.shape[0]
        self.exps = exps.astype(np.float32)
        self.batched_conditioned_matrix = conditioned_matrix_vec.reshape(
            exps.shape[0], matrix.shape[0]
        )

    def __len__(self):
        return len(self.exps)

    def __getitem__(self, idx):
        return self.exps[idx], self.batched_conditioned_matrix[idx]
