import os

import pandas as pd

from huggingface_hub import HfApi

from optimal_observables.optimization import data


if __name__ == "__main__":
    base_path = "../data/output_200k_spin"
    pos_process = "_SPIN_"
    neg_process = "_NOSPIN_"

    pos_reco_paths = [
        os.path.join(base_path, path)
        for path in os.listdir(base_path)
        if pos_process in path
    ]
    neg_reco_paths = [
        os.path.join(base_path, path)
        for path in os.listdir(base_path)
        if neg_process in path
    ]

    data_loader = data.ClassifierDataLoader(
        pos_reconstruction_paths=pos_reco_paths,
        neg_reconstruction_paths=neg_reco_paths,
        include_cosine_prods=True,
        include_mtt=True,
        include_pt=True,
        include_dPhi=True,
        include_top_vecs=True,
        include_lepton_vecs=True,
        include_neutrino_vecs=True,
        include_bjet_vecs=True,
        include_reco_weights=True,
    )
    X, y = data_loader.load()

    features_df = pd.DataFrame(X, columns=data_loader.feature_names)
    features_df["target"] = y
    features_df.to_parquet("../artifacts/all_features.parquet", index=False)

    X_pos = X[y.reshape(-1) == 1]
    X_neg = X[y.reshape(-1) == 0]
    labels = ["spin-ON", "spin-OFF"]

    # Update HuggingFace dataset from CSV
    api = HfApi()
    api.upload_file(
        path_or_fileobj="../artifacts/all_features.parquet",
        path_in_repo="data/all_features.parquet",
        repo_id="cmpatino/optimal_observables",
        repo_type="dataset",
    )
