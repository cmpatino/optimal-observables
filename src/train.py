import os
import json
import pickle

import mlflow
import optuna
import lightgbm as lgb
import xgboost as xgb
from optimal_observables.optimization.data import ClassifierDataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_params_trial(trial, model_type):
    if model_type == "lgbm":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        }
    elif model_type == "xgboost":
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        }
    elif model_type == "rf":
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        }
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    return params


def get_init_model(model_type, params, random_state):
    if model_type == "lgbm":
        model = lgb.LGBMClassifier(**params, random_state=random_state)
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(**params, random_state=random_state)
    elif model_type == "rf":
        model = RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    return model


def optuna_objective(trial, X_train, X_val, y_train, y_val, model_type, random_state):
    params = get_params_trial(trial, model_type)
    model = get_init_model(
        model_type=model_type, params=params, random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred_val = model.predict_proba(X_val)[:, 1]
    auc_score_val = roc_auc_score(y_true=y_val, y_score=y_pred_val)
    return auc_score_val


if __name__ == "__main__":
    with open("classifier_config.json", "r") as f:
        config = json.load(f)
    base_path = config["base_path"]
    pos_process = config["pos_process"]
    neg_process = config["neg_process"]
    random_seed = config["random_seed"]
    model_type = config["model_type"]
    model_hparams = config["model_hparams"]
    include_mtt = config["include_mtt"]
    include_cosine_prods = config["include_cosine_prods"]
    include_pt = config["include_pt"]
    n_hparam_trials = config["n_hparam_trials"]

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

    data_loader = ClassifierDataLoader(
        pos_reconstruction_paths=pos_reco_paths,
        neg_reconstruction_paths=neg_reco_paths,
        include_cosine_prods=include_cosine_prods,
        include_mtt=include_mtt,
    )
    X, y = data_loader.load()
    y = y.reshape(-1)

    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, stratify=y_val_test, random_state=random_seed
    )

    # Create numeric preprocessing pipeline
    data_processer = Pipeline([("standard_scaler", StandardScaler())])
    X_train = data_processer.fit_transform(X_train)
    X_val = data_processer.transform(X_val)
    X_test = data_processer.transform(X_test)

    if model_hparams is None:
        sampler = optuna.samplers.TPESampler(seed=random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            lambda trial: optuna_objective(
                trial,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                model_type=model_type,
                random_state=random_seed,
            ),
            n_trials=n_hparam_trials,
        )
        model_hparams = study.best_trial.params

    if model_type == "lgbm":
        mlflow.lightgbm.autolog()
    elif model_type == "xgboost":
        mlflow.xgboost.autolog()
    elif model_type == "rf":
        mlflow.sklearn.autolog()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    with mlflow.start_run() as run:
        model = get_init_model(
            model_type=model_type, params=model_hparams, random_state=random_seed
        )
        model.fit(X_train, y_train, feature_name=data_loader.feature_names)

        y_pred_train = model.predict_proba(X_train)
        y_pred_val = model.predict_proba(X_val)
        y_pred_test = model.predict_proba(X_test)

        auc_score_train = roc_auc_score(y_true=y_train, y_score=y_pred_train[:, 1])
        auc_score_val = roc_auc_score(y_true=y_val, y_score=y_pred_val[:, 1])
        auc_score_test = roc_auc_score(y_true=y_test, y_score=y_pred_test[:, 1])

        with open("data_preprocesser.pkl", "wb") as f:
            pickle.dump(data_processer, f)

        mlflow.log_artifact("data_preprocesser.pkl")
        mlflow.log_metric("train_roc-auc", auc_score_train)
        mlflow.log_metric("val_roc-auc", auc_score_val)
        mlflow.log_metric("test_roc-auc", auc_score_test)
        mlflow.log_params(config)

        os.remove("data_preprocesser.pkl")
