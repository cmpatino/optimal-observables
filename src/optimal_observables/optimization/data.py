import os
from typing import Dict, List

import numpy as np
from torch.utils.data import Dataset

from optimal_observables.optimization import observables


class DataGenerator:
    def __init__(
        self,
        pos_reconstruction_paths: List[str],
        neg_reconstruction_paths: List[str],
        only_cosine_terms: bool = True,
        include_mtt: bool = True,
    ):
        pos_recos = self._load_numpy_recos(
            reconstruction_paths=pos_reconstruction_paths
        )
        neg_recos = self._load_numpy_recos(
            reconstruction_paths=neg_reconstruction_paths
        )
        self.feature_names = [
            "cos_k1",
            "cos_k2",
            "cos_r1",
            "cos_r2",
            "cos_n1",
            "cos_n2",
        ]
        if not only_cosine_terms:
            self.feature_names.extend(
                [
                    "cos_k1 * cos_k2",
                    "cos_r1 * cos_r2",
                    "cos_n1 * cos_n2",
                    "(cos_r1 * cos_k2) + (cos_k1 * cos_r2)",
                    "(cos_r1 * cos_k2) - (cos_k1 * cos_r2)",
                    "(cos_n1 * cos_r2) + (cos_r1 * cos_n2)",
                    "(cos_n1 * cos_r2) - (cos_r1 * cos_n2)",
                    "(cos_n1 * cos_k2) + (cos_k1 * cos_n2)",
                    "(cos_n1 * cos_k2) - (cos_k1 * cos_n2)",
                ]
            )
        pos_matrix = observables.get_angles_matrix(
            p_l_t=pos_recos["p_l_t"],
            p_l_tbar=pos_recos["p_l_tbar"],
            p_top=pos_recos["p_top"],
            p_tbar=pos_recos["p_tbar"],
            only_cosine_terms=only_cosine_terms,
        )
        neg_matrix = observables.get_angles_matrix(
            p_l_t=neg_recos["p_l_t"],
            p_l_tbar=neg_recos["p_l_tbar"],
            p_top=neg_recos["p_top"],
            p_tbar=neg_recos["p_tbar"],
            only_cosine_terms=only_cosine_terms,
        )
        self.X = np.concatenate([pos_matrix, neg_matrix], axis=0)
        if include_mtt:
            pos_mtt = observables.get_mtt(
                p_top=pos_recos["p_top"],
                p_tbar=pos_recos["p_tbar"],
            )
            neg_mtt = observables.get_mtt(
                p_top=neg_recos["p_top"],
                p_tbar=neg_recos["p_tbar"],
            )
            X_mtt = np.concatenate([pos_mtt, neg_mtt], axis=0)
            self.X = np.concatenate([self.X, X_mtt], axis=1)
            self.feature_names.append("m_tt")

        self.y = np.concatenate(
            [np.ones(len(pos_matrix)), np.zeros(len(neg_matrix))]
        ).reshape(-1, 1)

    def generate_data(self):
        return self.X, self.y

    def _load_numpy_recos(
        self, reconstruction_paths: List[str]
    ) -> Dict[str, np.ndarray]:
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
        recos = {
            name: np.concatenate(batches, axis=0) for name, batches in batches.items()
        }
        return recos


class NNDataset(Dataset):
    def __init__(
        self,
        data_generator: DataGenerator,
    ):
        self.X, self.y = data_generator.generate_data()
        self.input_features = self.X.shape[1]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index].astype(np.float32), self.y[index].astype(np.int64)
