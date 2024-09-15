import os
import random
import sys
import numpy as np
import copy
from collections import defaultdict
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    max_error,
)
from .logging import logger


class PathManager:
    def __init__(self) -> None:
        self.PROJECT_ROOT = os.getcwd()
        self.DEFAULT_DATA_ROOT = f"{self.PROJECT_ROOT}/data"
        self.SWEEP_OUT_PATH = f"{self.PROJECT_ROOT}/sweeper_out"
        self.LOG_PATH = f"{self.SWEEP_OUT_PATH}/job_results"
        self.SWEEP_PATH = f"{self.PROJECT_ROOT}"
        self.WANDB_PROJECT_NAME = "sweeper"

        self.slurm_email = None

    def set_wandb_project_name(self, name):
        logger.info(
            f"Wandb project name updated - was ({self.WANDB_PROJECT_NAME}) now ({name})"
        )
        self.WANDB_PROJECT_NAME = name

    def set_data_path(self, path):
        self.DEFAULT_DATA_ROOT = path

    def set_output_path(self, path):
        self.SWEEP_OUT_PATH = path
        self.LOG_PATH = f"{self.SWEEP_OUT_PATH}/job_results"

    def set_slurm_email(self, email):
        self.slurm_email = email


path_manager = PathManager()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Factory:
    def __init__(self) -> None:
        self.factory_map = {}

    def create_obj(self, name: str, *args, **kwargs):
        return self.create_cls(name)(*args, **kwargs)

    def create_cls(self, name: str):
        return self.factory_map[name]


def generate_seed(upper_bound=None):
    upper = upper_bound if upper_bound else sys.maxsize
    seed = random.randint(0, upper)
    return seed


def train_and_benchmark_gbdt(experiment_data, train_data, test_data, model="GBM"):
    metrics = {}

    y_col = train_data.y
    # Real data
    y_train = train_data.df[y_col]
    X_train = train_data.df.drop([y_col], axis=1)
    y_test = test_data.df[y_col]
    X_test = test_data.df.drop([y_col], axis=1)

    for exp in experiment_data:
        exp_preds = {"train": [], "test": []}
        name, datasets, model = exp
        logger.info(f"Training GBDT model on {name} data...")
        metrics[name] = defaultdict(int)

        for i, synth_data in enumerate(datasets):
            c_model = copy.deepcopy(model)

            # Match column ordering, PrivBayes rearranges column orders...
            synth_data = synth_data[train_data.df.columns]
            y = synth_data[y_col]
            X = synth_data.drop([y_col], axis=1)

            # In cases where synthetic data is malformed, take random guess
            if len(np.unique(y)) == 1:
                trained_model = DummyClassifier(strategy="uniform").fit(
                    [None, None], [0, 1]
                )
            else:
                trained_model = c_model.fit(X, y)

            # Get metrics - Train and Test
            for task in [("train", X_train, y_train), ("test", X_test, y_test)]:
                prefix, X_tr, y_tr = task
                y_pred = trained_model.predict(X_tr)

                if train_data.task_type == "classification":
                    y_pred_proba = trained_model.predict_proba(X_tr)
                    # y_pred_proba = y_pred_proba[
                    #     :, y_pred_proba.shape[1] - 1
                    # ]  # to get it to work with DummyClassifier()...
                    exp_preds[prefix].append(y_pred_proba)
                    report = classification_report(y_tr, y_pred, output_dict=True)
                    metrics[name][f"{prefix}_acc"] += report["accuracy"]
                    metrics[name][f"{prefix}_prec"] += report["macro avg"]["precision"]
                    metrics[name][f"{prefix}_recall"] += report["macro avg"]["recall"]
                    metrics[name][f"{prefix}_f1"] += report["macro avg"]["f1-score"]
                    if not train_data.is_multiclass:
                        y_pred_proba = y_pred_proba[:, 1]
                    try:
                        metrics[name][f"{prefix}_auc"] += roc_auc_score(
                            y_tr, y_pred_proba, multi_class="ovo"
                        )
                    except ValueError:
                        metrics[name][f"{prefix}_auc"] = 0
                else:  # regression
                    metrics[name][f"{prefix}_r2"] = r2_score(y_tr, y_pred)
                    metrics[name][f"{prefix}_mse"] = mean_squared_error(y_tr, y_pred)
                    metrics[name][f"{prefix}_mae"] = mean_absolute_error(y_tr, y_pred)
                    metrics[name][f"{prefix}_max_err"] = max_error(y_tr, y_pred)

        # TODO: Rework this...
        if "ensemble" in name:
            metrics[f"{name}_weighted"] = defaultdict(int)
            for task in [("train", X_train, y_train), ("test", X_test, y_test)]:
                prefix, _, y = task
                preds = exp_preds[prefix]
                prob_avg = sum(preds) / len(preds)
                weights = np.array([d.shape[0] / X_train.shape[0] for d in datasets])
                weighted_average = sum(preds * weights.reshape(-1, 1))

                y_pred = prob_avg >= 0.5
                report = classification_report(y, y_pred, output_dict=True)
                metrics[name][f"{prefix}_acc"] = report["accuracy"]
                metrics[name][f"{prefix}_prec"] = report["macro avg"]["precision"]
                metrics[name][f"{prefix}_recall"] = report["macro avg"]["recall"]
                metrics[name][f"{prefix}_f1"] = report["macro avg"]["f1-score"]
                metrics[name][f"{prefix}_auc"] = roc_auc_score(y, prob_avg)

                y_pred = weighted_average >= 0.5
                report = classification_report(y, y_pred, output_dict=True)
                metrics[f"{name}_weighted"][f"{prefix}_acc"] = report["accuracy"]
                metrics[f"{name}_weighted"][f"{prefix}_prec"] = report["macro avg"][
                    "precision"
                ]
                metrics[f"{name}_weighted"][f"{prefix}_recall"] = report["macro avg"][
                    "recall"
                ]
                metrics[f"{name}_weighted"][f"{prefix}_f1"] = report["macro avg"][
                    "f1-score"
                ]
                metrics[f"{name}_weighted"][f"{prefix}_auc"] = roc_auc_score(
                    y, weighted_average
                )
        else:
            for k in metrics[name].keys():
                metrics[name][k] /= len(datasets)  # Average metrics over each dataset

    logger.info("GBM Train metrics:")
    for name, metric in metrics.items():
        if train_data.task_type == "classification":
            logger.info(f"{name}: acc={metric['train_acc']}, auc={metric['train_auc']}")
        else:
            logger.info(f"{name}: mse={metric['train_mse']}, r2={metric['train_r2']}")

    return metrics


__all__ = [
    "logger",
    "AttrDict",
    "Factory",
    "generate_seed",
]
