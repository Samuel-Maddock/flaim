import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import itertools

from collections import Counter
from sklearn.cluster import KMeans
from scipy.stats import zipfian
from synth_fl.utils import DEFAULT_DATA_ROOT, logger
from synth_fl.utils.dataloaders.core import TabularDataset


class Sharder(ABC):
    @abstractmethod
    def get_partition(self, num_partitions: int) -> List[TabularDataset]:
        pass


class HorizontalSharder(Sharder):
    def __init__(
        self,
        tabular_dataset: TabularDataset,
        non_iid: bool = "",
        use_cached_split: bool = True,
        cache_split: bool = True,
    ) -> None:
        self.dataset = tabular_dataset
        self.non_iid = non_iid
        self.use_cached_split = use_cached_split
        self.cache_split = cache_split
        self.k = 3
        self.non_iid_param = None
        self.param_map = {
            "label_skew": "beta=",
            "sample_skew": "beta=",
        }
        self.row_client_ids = None

    def _generate_workload(self, dataset):
        features = dataset.df.columns.values
        workload = []
        for k in range(0, self.k):
            workload.extend(list(itertools.combinations(features, k + 1)))
        return workload

    def _measure_heterogeneity(
        self,
        client_dataset,
        global_dataset,
        workload=None,
        client_answers=None,
        global_answers=None,
        client_id=None,
    ):

        if not client_answers:
            pgm_client_dataset = client_dataset.to_pgm_dataset()
            pgm_global_dataset = global_dataset.to_pgm_dataset()

        total_workload_err = 0
        for query in workload:
            if client_answers is None and client_id is None:
                global_ans = pgm_global_dataset.project(query).datavector()
                client_ans = pgm_client_dataset.project(query).datavector()
            else:
                global_ans = global_answers[query]
                client_ans = client_answers[client_id][query].astype("float32")

            global_ans /= global_ans.sum()
            client_ans /= client_ans.sum()

            # tv distance
            total_workload_err += 0.5 * np.linalg.norm(global_ans - client_ans, ord=1)

        return total_workload_err / len(workload)

    def measure_heterogeneity(
        self,
        client_datasets,
        central_dataset=None,
        workload=None,
        client_answers=None,
    ):
        if central_dataset is None and client_answers is None:
            central_dataset = client_datasets[0].merge(
                client_datasets[1:]
            )  # take union of client datasets to be global dataset

        # If workload is none, generate from global dataset
        if workload is None:
            workload = self._generate_workload(central_dataset)

        # If global answers is none, cache answers of global dataset
        global_answers = None
        if client_answers and workload:
            pgm_central = central_dataset.to_pgm_dataset()
            global_answers = {
                query: pgm_central.project(query).datavector() for query in workload
            }

        client_heterogeneity = [
            self._measure_heterogeneity(
                client_dataset,
                central_dataset,
                workload=workload,
                client_answers=client_answers,
                global_answers=global_answers,
                client_id=i,
            )
            for i, client_dataset in enumerate(client_datasets)
        ]

        return (
            np.mean(client_heterogeneity, axis=0),
            np.std(client_heterogeneity, axis=0),
            np.min(client_heterogeneity, axis=0),
            np.max(client_heterogeneity, axis=0),
        )

    def _iid_partition(self, num_partitions: int = 2):
        # shuffled_df = self.dataset.df.loc[np.random.permutation(self.dataset.df.index)]
        # return np.array_split(shuffled_df, num_partitions)
        row_ids_to_clients = np.array_split(
            np.random.permutation(self.dataset.df.index), num_partitions
        )
        client_ids = {}
        for i, items in enumerate(row_ids_to_clients):
            for row in items:
                client_ids[row] = i
        sorted_ids = dict(sorted(client_ids.items(), key=lambda item: item[0]))
        return list(sorted_ids.values())

    def _cluster_non_iid(self, num_partitions: int = 2, n_neighbors=15, min_dist=0.1):
        import umap

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
        embedding = reducer.fit_transform(self.dataset.df)
        kmeans = KMeans(n_clusters=num_partitions)
        client_ids = kmeans.fit_predict(embedding)
        return client_ids

    # beta -> 0 => skewed labels
    def _label_skew_non_iid(self, num_partitions: int = 2, beta: float = 100):
        y_train = self.dataset.df[self.dataset.y]
        min_size = 0
        partition_all = []
        front = np.array([0])
        N = y_train.shape[0]
        K = len(y_train.unique())
        client_ids = [0] * N

        while min_size < 2:
            logger.debug(f"Min size {min_size}")
            idx_batch = [[] for _ in range(num_partitions)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, num_partitions))
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / num_partitions)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                back = np.array([idx_k.shape[0]])
                partition = np.concatenate((front, proportions, back), axis=0)
                partition = np.diff(partition)
                partition_all.append(partition)
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_partitions):
            for i in idx_batch[j]:
                client_ids[i] = j

        return client_ids

    def _label_skew_dirch_non_iid(self, num_partitions: int = 2, beta: float = 1):
        y_train = self.dataset.df[self.dataset.y]
        K = len(y_train.unique())
        sample_id_map = {}

        for k in y_train.unique():
            total = sum(y_train == k)
            sample_proportions = (
                np.random.dirichlet(np.repeat(beta, num_partitions)) * total
            )
            sample_proportions = np.array([int(x) for x in sample_proportions])
            sample_proportions[sample_proportions == 0] = 1
            rounding_err = total - sum(sample_proportions)
            if rounding_err > 0:
                # Add to smallest
                sample_proportions[np.argmin(sample_proportions)] += rounding_err
            else:
                # Remove from largest
                sample_proportions[np.argmax(sample_proportions)] += rounding_err

            samples = np.random.permutation(np.where(y_train == k)[0])
            client_id, num_assigned = 0, 0
            for i, sample_id in enumerate(samples):
                sample_id_map[sample_id] = client_id
                num_assigned += 1
                if num_assigned == sample_proportions[client_id]:
                    client_id += 1
                    num_assigned = 0
        client_ids = [sample_id_map[i] for i in range(0, self.dataset.df.shape[0])]
        return client_ids

    def _sample_skew_dirch_non_iid(self, num_partitions: int = 2, a: int = 0):
        sample_proportions = (
            np.random.dirichlet(np.repeat(a, num_partitions)) * self.dataset.df.shape[0]
        )
        sample_proportions = np.array([int(x) for x in sample_proportions])
        sample_proportions[sample_proportions == 0] = 1
        rounding_err = self.dataset.df.shape[0] - sum(sample_proportions)

        if rounding_err > 0:
            # Add to smallest
            sample_proportions[np.argmin(sample_proportions)] += rounding_err
        else:
            # Remove from largest
            sample_proportions[np.argmax(sample_proportions)] += rounding_err

        sample_id_map = {}
        samples = np.random.permutation(list(range(0, self.dataset.df.shape[0])))
        client_id, num_assigned = 0, 0
        for i, sample_id in enumerate(samples):
            sample_id_map[sample_id] = client_id
            num_assigned += 1
            if num_assigned == sample_proportions[client_id]:
                client_id += 1
                num_assigned = 0

        client_ids = [sample_id_map[i] for i in range(0, self.dataset.df.shape[0])]
        return client_ids

    # a -> 0 => uniform dist, large a => high skew with number of samples
    def _sample_skew_non_iid(self, num_partitions: int = 2, a: int = 0):
        # Sample n=num_partitions, a=a from zipifian dist and then compute pmf over [1.., n] and sample data of the size from df
        n = num_partitions
        pmf = [zipfian.pmf(k, a, n) for k in range(1, num_partitions + 1)]
        client_ids = []

        # Sample each to client with pmf
        client_ids = [
            np.random.choice(list(range(0, num_partitions)), p=pmf)
            for i in range(0, self.dataset.df.shape[0])
        ]
        return client_ids

    def _parse_non_iid_method(self, num_partitions: int = 2):
        logger.info(f"Non-IID split not cached for method={self.non_iid}, computing...")
        if "debug_label_skew" in self.non_iid:
            param_regex = self.param_map["label_skew"]
            beta = (
                0.05
                if len(self.non_iid.split(param_regex)) == 1
                else float(self.non_iid.split(param_regex)[1])
            )
            self.non_iid_param = beta
            return self._label_skew_non_iid(num_partitions, beta=beta)
        elif "zipf_sample_skew" in self.non_iid:
            param_regex = self.param_map["sample_skew"]
            a = (
                0
                if len(self.non_iid.split(param_regex)) == 1
                else float(self.non_iid.split(param_regex)[1])
            )
            self.non_iid_param = a
            return self._sample_skew_non_iid(num_partitions, a=a)
        elif "dirch_label_skew" in self.non_iid:
            param_regex = self.param_map["label_skew"]
            beta = (
                0.1
                if len(self.non_iid.split(param_regex)) == 1
                else float(self.non_iid.split(param_regex)[1])
            )
            self.non_iid_param = beta
            return self._label_skew_dirch_non_iid(num_partitions, beta=beta)
        elif "dirch_sample_skew" in self.non_iid:
            param_regex = self.param_map["sample_skew"]
            a = (
                0
                if len(self.non_iid.split(param_regex)) == 1
                else float(self.non_iid.split(param_regex)[1])
            )
            self.non_iid_param = a
            return self._sample_skew_dirch_non_iid(num_partitions, a=a)
        elif "cluster" in self.non_iid:
            n_neighbors, min_dist = 15, 0.1
            n_neighbors = (
                15
                if len(self.non_iid.split("beta=")) == 1
                else float(self.non_iid.split("beta=")[1])
            )
            # min_dist = (
            #     0.1
            #     if len(self.non_iid.split("min_dist=")) == 0
            #     else float(self.non_iid.split("min_dist=")[1])
            # )
            return self._cluster_non_iid(
                num_partitions, n_neighbors=n_neighbors, min_dist=min_dist
            )

    def get_partition(self, num_partitions: int) -> List[TabularDataset]:
        non_iid_method = self.non_iid
        SPLIT_PATH = f"{DEFAULT_DATA_ROOT}/{self.dataset.name}_p={self.dataset.p}_seed={self.dataset.random_state}_n={self.dataset.df.shape[0]}_k={num_partitions}_{non_iid_method}.npy"

        if "synthetic" in self.dataset.name:
            client_ids = self.dataset.df["client_id"]
            self.non_iid_param = self.dataset.name.split("a=")[1].split("_")[0]
            self.dataset.df.drop("client_id", axis=1, inplace=True)
        elif self.use_cached_split and os.path.exists(SPLIT_PATH):
            logger.info(f"Client split is already cached, loading from {SPLIT_PATH}")
            client_ids = np.load(SPLIT_PATH)
            if self.non_iid:
                self.non_iid_param = (
                    None
                    if len(self.non_iid.split("beta=")) == 1
                    else float(self.non_iid.split("beta=")[1])
                )
        else:
            if non_iid_method:
                client_ids = self._parse_non_iid_method(num_partitions)
            else:
                client_ids = self._iid_partition(num_partitions)
            if self.cache_split:
                logger.info(f"Client split cached to {SPLIT_PATH}")
                np.save(SPLIT_PATH, client_ids)

        self.row_client_ids = client_ids
        client_distribution = Counter(client_ids)
        logger.debug(
            f"Client sample distribution {len(client_distribution)} - {client_distribution}"
        )
        partitions = [x for _, x in self.dataset.df.groupby(client_ids)]

        return [
            TabularDataset(
                f"{self.dataset.name}",
                df=partition,
                domain=self.dataset.domain,
                y=self.dataset.y,
                task_type=self.dataset.task_type,
                is_multiclass=self.dataset.is_multiclass,
                random_state=self.dataset.random_state,
                non_iid=self.non_iid,
                p=self.dataset.p,
            )
            for i, partition in enumerate(partitions)
        ]


class VerticalSharder(Sharder):
    def __init__(
        self,
        tabular_dataset: TabularDataset,
    ) -> None:
        self.dataset = tabular_dataset

    def get_partition(self, num_partitions: int) -> List[TabularDataset]:
        num_columns = self.dataset.df.shape[1]
        assert num_partitions <= num_columns
        shuffled_columns = np.random.permutation(self.dataset.df.columns)
        partitions = np.array_split(shuffled_columns, num_partitions)

        return [
            TabularDataset(
                self.dataset.name,
                df=self.dataset.df[partition],
                domain={col: self.dataset.domain[col] for col in partition},
                y=self.dataset.y,
            )
            for partition in partitions
        ]
