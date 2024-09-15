import json
import os
import torch

import numpy as np
import pandas as pd

# PMLB
from pmlb import fetch_data
from pmlb.update_dataset_files import get_dataset_stats

# SDV
from sdv.datasets.demo import download_demo, get_available_demos

from synth_fl.libs.private_pgm.mbi import Dataset as PGMDataset
from synth_fl.libs.private_pgm.mbi import Domain as PGMDomain
from synth_fl.utils import logger
from scipy.stats import zipfian

from collections import Counter, defaultdict
from collections.abc import Iterable

from typing import List
from torch.utils.data import Dataset

# For manual datasets (e.g. not PMLB or sdv)
CLASS_MAP = {
    "adult": "income>50K",
    "marketing": "y",
    "covtype": "cover_type",
}

SUBSAMPLE_MAP = {
    "intrusion_sdv": 0.4,
    "census_sdv": 0.3,
    "covtype": 0.2,
}

CLASS_MAP = defaultdict(lambda: "label", CLASS_MAP)


class TabularDataset:
    def __init__(
        self,
        name,
        filepath=path_manager.DEFAULT_DATA_ROOT,
        df=None,
        domain=None,
        y=None,
        task_type=None,
        is_multiclass=None,
        random_state=None,
        non_iid=None,
        p=1,
        num_bins=32,
        preprocess=True,
        torch_dataset=None,
        remove_nans=True,
    ) -> None:
        self.name = name
        self.filepath = filepath
        self.remove_nans = remove_nans

        # for discretising cts features, currently PMLB datasets or self.preprocess=True only
        self.num_bins = num_bins
        # Whether to preprocess dataset, by default will bin cts features (uniformly)
        self.preprocess = preprocess
        self.df = df
        self.domain = domain
        self.n, self.d = 0, 0
        self.y = None
        self.task_type = "classification" if not task_type else task_type
        self.is_multiclass = False if not is_multiclass else is_multiclass
        self.feature_type = {}
        self.random_state = random_state
        self.non_iid = non_iid
        self.p = p

        if torch_dataset:
            X_list, y_list = [], []
            if isinstance(torch_dataset, Dataset):
                for X_item, y_item in torch_dataset:
                    if isinstance(X_item, Iterable):
                        X_list.extend(X_item)
                        y_list.extend(y_item)
                    else:
                        X_list.append(X_item)
                        y_list.append(y_item)
            else:
                # TODO: edge case when torch_dataset is just a tuple of X,y values
                for i in range(len(torch_dataset[1])):
                    X_list.append(torch_dataset[0][i])
                    y_list.append(torch_dataset[1][i])
            self.df = pd.DataFrame({"X": X_list, "y": y_list})
            self.y = "y"
        elif self.df is None:
            self._load()
        else:
            self.n, self.d = self.df.shape

        if y:
            self.y = y

    def _generate_synthetic(self, k, n, d, a, n_zipf):
        data = []
        for i in range(0, int(k)):
            client_data = {}
            for j in range(0, int(d)):
                mean = zipfian.rvs(float(a), int(n_zipf), size=1)
                bins = np.linspace(0, int(n_zipf), num=int(n_zipf))
                feat_vals = np.digitize(
                    np.random.normal(mean, 1, size=int(n)), bins, right=False
                )
                client_data[str(j)] = feat_vals
            client_df = pd.DataFrame(client_data)
            client_df["client_id"] = i
            data.append(client_df)
        return pd.concat(data)

    def _discretise_continuous_feat(self, x, num_bins=32):
        if num_bins == -1:
            return x, -1
        pd_cut = pd.cut(x, bins=num_bins, labels=False)
        return pd.to_numeric(pd_cut), num_bins

    def _preprocess_pmlb(self):
        logger.info(f"Fetching {self.name} from PMLB...")
        dataset_name = self.name.split("_pmlb")[0]
        data = fetch_data(
            dataset_name, local_cache_dir=path_manager.DEFAULT_DATA_ROOT, dropna=True
        )
        domain = {}
        dataset_stats = get_dataset_stats(data)

        if self.name == "mushroom":
            # Feature val is constant...
            data = data.drop("veil-type", axis=1)

        logger.info(f"Processing PMLB dataset... - num_bins={self.num_bins}")
        for col in data.columns:
            if col != "target":
                feat_idx = dataset_stats["feat_names"].index(col)
                feat_type = dataset_stats["types"][feat_idx]
                domain_size = data[col].max() + 1
                self.feature_type[col] = feat_type
                if feat_type == "continuous" and self.num_bins != -1:
                    data[col], domain_size = self._discretise_continuous_feat(
                        data[col], num_bins=self.num_bins
                    )
                domain[col] = domain_size

        domain["target"] = len(data["target"].unique())
        data["target"] = pd.factorize(data["target"], sort=True)[0]

        self.task_type = (
            "regression" if dataset_stats["task"] == "regression" else "classification"
        )
        self.feature_type["target"] = (
            "discrete" if self.task_type == "classification" else "continuous"
        )
        self.filepath = (
            f"{path_manager.DEFAULT_DATA_ROOT}/{self.name}/{self.name}.tsv.gz"
        )

        return data, domain

    # todo: merge _preprocess_sdv and _preprocess_pmlb
    def _preprocess_sdv(self, dataset_name, data, metadata):
        domain = {}
        class_name = CLASS_MAP[dataset_name]
        for col in data.columns:
            if col != class_name:
                feat_type = metadata["columns"][col]["sdtype"]
                self.feature_type[col] = feat_type
                # print(col, feat_type)
                if feat_type == "numerical" and self.num_bins != -1:
                    data[col], domain_size = self._discretise_continuous_feat(
                        data[col], num_bins=self.num_bins
                    )
                if (
                    feat_type == "categorical"
                    or feat_type == "boolean"
                    or feat_type == "id"
                    or feat_type == "datetime"
                ):
                    data[col] = pd.factorize(data[col], sort=True)[0]
                    domain_size = len(data[col].unique())
                domain[col] = domain_size

        domain[class_name] = len(data[class_name].unique())
        data[class_name] = pd.factorize(data[class_name], sort=True)[0]
        self.task_type = "classification"
        self.feature_type[class_name] = "categorical"
        self.y = class_name
        return data, domain

    # TODO: Add logic based on dtype of cols...
    def _infer_domain(self, df, y):
        domain = {col: -1 for col in df.columns}
        domain[y] = len(df[y].unique())
        return domain

    def _load(self):
        if "synthetic" in self.name:
            filepath = f"{path_manager.DEFAULT_DATA_ROOT}/{self.name}.csv"
            param_dict = {}
            for arg in self.name.split("_")[1:]:
                name, val = arg.split("=")[0], arg.split("=")[1]
                param_dict[name] = val
            if os.path.exists(filepath):
                logger.info(f"Loading cached synthetic dataset - {self.name}")
                self.df = pd.read_csv(filepath)
            else:
                logger.info(
                    f"Generating synthetic dataset from param dict - {param_dict}"
                )
                self.df = self._generate_synthetic(
                    k=param_dict["k"],
                    n=param_dict["n"],
                    d=param_dict["d"],
                    a=param_dict["a"],
                    n_zipf=param_dict["n-zipf"],
                )
                logger.info(f"Saving synthetic dataset to {filepath}")
                self.df.to_csv(filepath, index=False)
            self.domain = {
                col: int(param_dict["n-zipf"]) + 1
                for col in set(self.df.columns) - set(["client_id"])
            }
            self.y = str(self.df.shape[1] - 2)
        elif "_pmlb" in self.name:
            self.df, self.domain = self._preprocess_pmlb()
            self.y = "target"
        elif "_sdv" in self.name:
            dataset_name = self.name.split("_sdv")[0]
            output_folder = f"{path_manager.DEFAULT_DATA_ROOT}/{dataset_name}"
            if not os.path.exists(output_folder):
                data, metadata = download_demo(
                    modality="single_table",
                    dataset_name=dataset_name,
                    output_folder_name=output_folder,
                )
                metadata = metadata.to_dict()
            else:
                data = pd.read_csv(f"{output_folder}/{dataset_name}.csv")
                with open(f"{output_folder}/metadata.json") as f:
                    metadata = json.load(f)
            self.df, self.domain = self._preprocess_sdv(dataset_name, data, metadata)
        else:
            sep = ";" if self.name == "marketing" else ","
            self.df = pd.read_csv(f"{self.filepath}/{self.name}.csv", sep=sep)
            self.y = CLASS_MAP[self.name]

            try:
                self.domain = json.load(
                    open(os.path.join(self.filepath, f"{self.name}-domain.json"))
                )
            except FileNotFoundError:
                logger.warning(f"No domain found for {self.name} - inferring...")
                self.domain = self._infer_domain(self.df, self.y)

            self.filepath = f"{self.filepath}/{self.name}.csv"
            if self.preprocess:
                for col, domain_size in self.domain.items():
                    if domain_size == -1:  # cts feature, discretise
                        self.df[col], domain_size = self._discretise_continuous_feat(
                            self.df[col], num_bins=self.num_bins
                        )
                        self.domain[col] = self.num_bins
                        self.feature_type[col] = "continuous"
                    elif self.df[col].dtype == "object" or col == self.y:
                        logger.debug(
                            f"Col {col} is discrete (object) or class, factorising"
                        )
                        self.df[col] = self.df[col].factorize()[0]
                        self.feature_type[col] = "discrete"

        logger.debug(f"Dataset shape before filtering (domain + nans) - {self.df.shape}")
        # Filter out attributes not defined in the domain
        for col in self.df.columns:
            if col not in self.domain:
                self.df = self.df.drop(col, axis=1)
        # Filter out nans
        if self.remove_nans:
            self.df = self.df.dropna()
        # Subsample if needed
        if self.name in SUBSAMPLE_MAP:
            self.df = self.df.sample(frac=SUBSAMPLE_MAP[self.name], random_state=1034) # Fix seed
        logger.debug(f"Dataset shape after filtering (subsample + domain + nans) - {self.df.shape}")

        # import pdb; pdb.set_trace()
            
        self.shape = self.df.shape
        self.n = self.shape[0]
        self.d = self.shape[1]
        self.is_multiclass = (
            True
            if len(self.df[self.y].unique()) > 2 and self.task_type == "classification"
            else False
        )

        logger.debug(
            f"Dataset {self.name} shape={self.n, self.d}, domain={self.domain}, y={self.y}"
        )
        if self.task_type == "classification":
            logger.debug(f"Dataset class imbalance - {Counter(self.df[self.y].values)}")

    def to_pgm_dataset(self) -> PGMDataset:
        return PGMDataset(
            self.df, PGMDomain(list(self.domain.keys()), list(self.domain.values()))
        )

    def to_ohe(self) -> np.array:
        return TabularDataset.one_hot_encode(self.df, self.domain)

    def to_X_y(self, transform=None):
        y = self.df[self.y]
        X = self.df.drop([self.y], axis=1)
        return X, y

    def to_X_y_tensor(self, transform=None):
        X, y = self.to_X_y()
        y_tensor = torch.Tensor(y.to_numpy())
        if transform:
            tensor_list = []
            for x in X.to_numpy():
                tensor_list.append(transform(x))
            X_tensor = torch.stack(tensor_list)
        else:
            X_tensor = torch.Tensor(X.to_numpy())
        return X_tensor, y_tensor

    def to_subset(self, p=1, n=None, random_state=None):
        subset1 = self.df.sample(n=n, frac=p, random_state=random_state)
        subset2 = self.df[~(self.df.index.isin(subset1.index))]
        return TabularDataset(
            name=f"{self.name}",
            df=subset1,
            domain=self.domain,
            y=self.y,
            task_type=self.task_type,
            is_multiclass=self.is_multiclass,
            random_state=random_state,
            p=p,
            filepath=self.filepath,
        ), TabularDataset(
            name=f"{self.name}",
            df=subset2,
            domain=self.domain,
            y=self.y,
            task_type=self.task_type,
            is_multiclass=self.is_multiclass,
            random_state=random_state,
            p=p,
            filepath=self.filepath,
        )

    # todo: does not currently check features/schema is the same
    def merge(self, tabular_datasets: List[object]):
        for dataset in tabular_datasets:
            assert dataset.d == self.d
            self.df = pd.concat([self.df, dataset.df], axis=0)
            self.n += dataset.n
        self.df.reset_index(inplace=True)

    @staticmethod
    def one_hot_encode(df, domain) -> np.array:
        """
        return subset of data over feats
        """
        feats_domain = {key: domain[key] for key in domain}

        # get binning of attributes
        bins_size_array = [
            (size_bin, np.digitize(df[col], range(size_bin + 1), right=True))
            for col, size_bin in feats_domain.items()
        ]

        # perform one-hot-encoding of all features and stack them into a numpy matrix
        bin_dataset = np.hstack(
            [np.eye(size_bin)[bin_array] for size_bin, bin_array in bins_size_array]
        )

        return bin_dataset
