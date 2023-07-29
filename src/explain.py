import os
import json
import pickle

import mlflow
import numpy as np
import pandas as pd

from optimal_observables.optimization.data import ClassifierDataLoader
from optimal_observables.optimization.explainers import SymbolicRegressionExplainer
from sklearn import metrics


def exponent_product(x, *params):
    coeffs_arr = np.array(params)
    y_pred = np.sum(coeffs_arr * x, axis=1)
    return y_pred


if __name__ == "__main__":
    run_id = "47949c8950d24c379341a0fb28c0582b"
    n_features = 5

    # Get run
    mlflow_client = mlflow.tracking.MlflowClient()
    run = mlflow_client.get_run(run_id)

    base_path = run.data.params["base_path"]
    pos_process = run.data.params["pos_process"]
    neg_process = run.data.params["neg_process"]
    include_cosine_prods = (
        False if run.data.params["include_cosine_prods"] == "False" else True
    )
    include_mtt = False if run.data.params["include_mtt"] == "False" else True
    include_pt = False if run.data.params["include_pt"] == "False" else True

    mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="feature_importance_gain.json", dst_path="."
    )
    with open("feature_importance_gain.json", "r") as f:
        feature_importance_gain = pd.Series(json.load(f))
    os.remove("feature_importance_gain.json")
    features = (
        feature_importance_gain.sort_values(ascending=False)
        .index[:n_features]
        .to_list()
    )

    pos_reco_paths = [
        os.path.join(base_path, pos_process, path)
        for path in os.listdir(os.path.join(base_path, pos_process))
        if os.path.isdir(os.path.join(base_path, pos_process, path))
    ]
    neg_reco_paths = [
        os.path.join(base_path, neg_process, path)
        for path in os.listdir(os.path.join(base_path, neg_process))
        if os.path.isdir(os.path.join(base_path, neg_process, path))
    ]

    data_loader = ClassifierDataLoader(
        pos_reconstruction_paths=pos_reco_paths,
        neg_reconstruction_paths=neg_reco_paths,
        include_cosine_prods=include_cosine_prods,
        include_mtt=include_mtt,
        include_pt=include_pt,
        include_dPhi=True,
    )
    X, y = data_loader.load()
    y = y.reshape(-1)

    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")

    # Download artifact from mlflow run
    mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="data_preprocesser.pkl", dst_path="."
    )
    with open("data_preprocesser.pkl", "rb") as f:
        data_processer = pickle.load(f)
    os.remove("data_preprocesser.pkl")

    X = data_processer.transform(X)
    y_model = model.predict_proba(X)[:, 1]

    # Get X to explain
    feature_idxs = [data_loader.feature_names.index(f) for f in features]
    X_explain = X[:, feature_idxs]

    # Sample 10000 rows to explain randomly
    sampled_idxs = np.random.choice(X_explain.shape[0], size=10000, replace=False)
    X_explain = X_explain[sampled_idxs]
    y_model_explain = y_model[sampled_idxs]

    explainer = SymbolicRegressionExplainer(
        binary_operators=["+", "*", "/", "-"],
        niterations=5,
        variable_names=features,
    )
    explainer.fit(X_explain, y_model_explain)
    y_explainer = explainer.score(X_explain)

    print(f"Reconstuction MAE: {np.mean(np.abs(y_model[sampled_idxs] - y_explainer))}")

    print(f"Model Performance: {metrics.roc_auc_score(y, y_model)}")
    print(
        f"Observable Performance: {metrics.roc_auc_score(y[sampled_idxs], y_explainer)}"
    )
