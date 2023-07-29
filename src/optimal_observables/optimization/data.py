import os
from typing import Dict, List, Tuple

import numpy as np

# from torch.utils.data import Dataset

from optimal_observables.optimization import observables
from optimal_observables.reconstruction import kinematics


class ReconstructionLoader:
    def __init__(self, reconstruction_paths: List[str]) -> None:
        self.reconstruction_paths = reconstruction_paths

    def load(self) -> Dict[str, np.ndarray]:
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
        for reconstruction_path in self.reconstruction_paths:
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


class BenchmarkDataLoader:
    def __init__(
        self,
        pos_reconstruction_paths: List[str],
        neg_reconstruction_paths: List[str],
    ):
        self.pos_recos = ReconstructionLoader(
            reconstruction_paths=pos_reconstruction_paths
        ).load()
        self.neg_recos = ReconstructionLoader(
            reconstruction_paths=neg_reconstruction_paths
        ).load()

    def load(self):
        dPhi_pos = observables.get_dPhi_ll(
            p_l_t=self.pos_recos["p_l_t"],
            p_l_tbar=self.pos_recos["p_l_tbar"],
        )
        dPhi_neg = observables.get_dPhi_ll(
            p_l_t=self.neg_recos["p_l_t"],
            p_l_tbar=self.neg_recos["p_l_tbar"],
        )
        y_bench = np.concatenate([dPhi_pos, dPhi_neg], axis=0)
        y = np.concatenate(
            [np.ones(dPhi_pos.shape[0]), np.zeros(dPhi_neg.shape[0])], axis=0
        )
        # We need to reverse the labels because the score is larger for negatives
        y = (y == 0).astype(int)
        return y_bench, y


class ClassifierDataLoader:
    def __init__(
        self,
        pos_reconstruction_paths: List[str],
        neg_reconstruction_paths: List[str],
        include_cosine_prods: bool = False,
        include_mtt: bool = False,
        include_pt: bool = False,
        include_dPhi: bool = False,
        include_top_vecs: bool = False,
        include_lepton_vecs: bool = False,
        include_bjet_vecs: bool = False,
        include_neutrino_vecs: bool = False,
        include_reco_weights: bool = False,
    ):
        pos_recos = ReconstructionLoader(
            reconstruction_paths=pos_reconstruction_paths
        ).load()
        neg_recos = ReconstructionLoader(
            reconstruction_paths=neg_reconstruction_paths
        ).load()
        self.feature_names = [
            "cos_k1",
            "cos_k2",
            "cos_r1",
            "cos_r2",
            "cos_n1",
            "cos_n2",
        ]
        if include_cosine_prods:
            self.feature_names.extend(
                [
                    "cos_k1_x_cos_k2",
                    "cos_r1_x_cos_r2",
                    "cos_n1_x_cos_n2",
                    "cos_r1_x_cos_k2_plus_cos_k1_x_cos_r2",
                    "cos_r1_x_cos_k2_minus_cos_k1_x_cos_r2",
                    "cos_n1_x_cos_r2_plus_cos_r1_x_cos_n2",
                    "cos_n1_x_cos_r2_minus_cos_r1_x_cos_n2",
                    "cos_n1_x_cos_k2_plus_cos_k1_x_cos_n2",
                    "cos_n1_x_cos_k2_minus_cos_k1_x_cos_n2",
                ]
            )

        pos_matrix = observables.get_angles_matrix(
            p_l_t=pos_recos["p_l_t"],
            p_l_tbar=pos_recos["p_l_tbar"],
            p_top=pos_recos["p_top"],
            p_tbar=pos_recos["p_tbar"],
            include_cosine_prods=include_cosine_prods,
        )
        neg_matrix = observables.get_angles_matrix(
            p_l_t=neg_recos["p_l_t"],
            p_l_tbar=neg_recos["p_l_tbar"],
            p_top=neg_recos["p_top"],
            p_tbar=neg_recos["p_tbar"],
            include_cosine_prods=include_cosine_prods,
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
        if include_pt:
            pos_pt = observables.get_pt_ttbar(
                p_top=pos_recos["p_top"],
                p_tbar=pos_recos["p_tbar"],
            )
            neg_pt = observables.get_pt_ttbar(
                p_top=neg_recos["p_top"],
                p_tbar=neg_recos["p_tbar"],
            )
            X_pt = np.concatenate([pos_pt, neg_pt], axis=0)
            self.X = np.concatenate([self.X, X_pt], axis=1)
            self.feature_names.append("pt")
        if include_dPhi:
            pos_dPhi = observables.get_dPhi_ll(
                p_l_t=pos_recos["p_l_t"],
                p_l_tbar=pos_recos["p_l_tbar"],
            )
            neg_dPhi = observables.get_dPhi_ll(
                p_l_t=neg_recos["p_l_t"],
                p_l_tbar=neg_recos["p_l_tbar"],
            )
            X_dPhi = np.concatenate([pos_dPhi, neg_dPhi], axis=0)
            self.X = np.concatenate([self.X, X_dPhi], axis=1)
            self.feature_names.append("dPhi_ll")

        if include_top_vecs:
            X_top_vecs, top_feature_names = self._get_vec_features(
                particle_names=["p_top", "p_tbar"],
                pos_recos=pos_recos,
                neg_recos=neg_recos,
            )
            self.X = np.concatenate([self.X, X_top_vecs], axis=1)
            self.feature_names.extend(top_feature_names)

        if include_lepton_vecs:
            X_lepton_vecs, lepton_feature_names = self._get_vec_features(
                particle_names=["p_l_t", "p_l_tbar"],
                pos_recos=pos_recos,
                neg_recos=neg_recos,
            )
            self.X = np.concatenate([self.X, X_lepton_vecs], axis=1)
            self.feature_names.extend(lepton_feature_names)

        if include_neutrino_vecs:
            X_neutrino_vecs, neutrino_feature_names = self._get_vec_features(
                particle_names=["p_nu_t", "p_nu_tbar"],
                pos_recos=pos_recos,
                neg_recos=neg_recos,
            )
            self.X = np.concatenate([self.X, X_neutrino_vecs], axis=1)
            self.feature_names.extend(neutrino_feature_names)
        if include_bjet_vecs:
            X_bjet_vecs, bjet_feature_names = self._get_vec_features(
                particle_names=["p_b_t", "p_b_tbar"],
                pos_recos=pos_recos,
                neg_recos=neg_recos,
            )
            self.X = np.concatenate([self.X, X_bjet_vecs], axis=1)
            self.feature_names.extend(bjet_feature_names)
        if include_reco_weights:
            X_reco_weights = np.concatenate(
                [pos_recos["weight"], neg_recos["weight"]], axis=0
            )
            self.X = np.concatenate([self.X, X_reco_weights], axis=1)
            self.feature_names.append("reco_weight")

        self.y = np.concatenate(
            [np.ones(len(pos_matrix)), np.zeros(len(neg_matrix))]
        ).reshape(-1, 1)

    def load(self):
        return self.X, self.y

    def _get_vec_features(
        self,
        particle_names: List[str],
        pos_recos: Dict[str, np.ndarray],
        neg_recos: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, List[str]]:
        pos_vecs_xyz = np.concatenate(
            [pos_recos[particle_name] for particle_name in particle_names],
            axis=1,
        )
        neg_vecs_xyz = np.concatenate(
            [neg_recos[particle_name] for particle_name in particle_names],
            axis=1,
        )
        pos_vecs_polar = np.concatenate(
            [
                self._get_polar_vecs(pos_recos[particle_name])
                for particle_name in particle_names
            ],
            axis=1,
        )
        neg_vecs_polar = np.concatenate(
            [
                self._get_polar_vecs(neg_recos[particle_name])
                for particle_name in particle_names
            ],
            axis=1,
        )
        X_vecs_xyz = np.concatenate([pos_vecs_xyz, neg_vecs_xyz], axis=0)
        X_vecs_polar = np.concatenate([pos_vecs_polar, neg_vecs_polar], axis=0)
        X_vecs = np.concatenate([X_vecs_xyz, X_vecs_polar], axis=1)
        feature_names_xyz = [
            f"{particle_name}_{component}"
            for particle_name in particle_names
            for component in ["px", "py", "pz", "E"]
        ]
        feature_names_polar = [
            f"{particle_name}_{component}"
            for particle_name in particle_names
            for component in ["pt", "eta", "phi", "mass"]
        ]
        feature_names = feature_names_xyz + feature_names_polar
        return X_vecs, feature_names

    def _get_polar_vecs(self, X_four_vecs: np.ndarray) -> np.ndarray:
        pt = kinematics.get_pt(X_four_vecs)
        eta = kinematics.eta(X_four_vecs)
        phi = kinematics.phi(X_four_vecs)
        mass = kinematics.mass(X_four_vecs)[:, 0]
        return np.stack([pt, eta, phi, mass], axis=1)
