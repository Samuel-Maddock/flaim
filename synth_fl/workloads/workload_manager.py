from synth_fl.utils.dataloaders import TabularDataset
from synth_fl.utils import logger

from typing import Optional
import itertools
import numpy as np


class WorkloadManager:
    def __init__(
        self,
        type: Optional[str] = "uniform",
        num_marginals: Optional[int] = 64,
        k: Optional[int] = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.type = type
        self.num_marginals = num_marginals
        self.k = k
        self.seed = seed if seed else np.random.randint(0, 2**12)

    # Helpers from AIM
    def _powerset(self, iterable):
        "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(1, len(s) + 1)
        )

    def _downward_closure(self, workload):
        ans = set()
        for proj in workload:
            ans.update(self._powerset(proj))
        return list(sorted(ans, key=len))

    def _compile_workload(self, workload):
        def score(cl):
            return sum(len(set(cl) & set(ax)) for ax in workload)

        return {cl: score(cl) for cl in self._downward_closure(workload)}

    def _parse_workload_args(self, num_marginals, k, seed):
        if not k:
            k = self.k
        if not num_marginals:
            num_marginals = self.num_marginals
        if not seed:
            seed = self.seed

        return num_marginals, k, seed

    def generate_test_workload(self, dataset: TabularDataset, workload, size=64, k=3):
        columns = dataset.df.columns.to_list()
        if "client_id" in columns:
            columns.remove("client_id")
        test_workload = [query for query in itertools.combinations(columns, k)]
        for w in workload:
            test_workload.remove(w)

        rng = np.random.default_rng(seed=self.seed)
        indexes = rng.choice(
            len(test_workload), min(len(test_workload), size), replace=False
        )
        return [test_workload[i] for i in indexes]

    def generate_workload(self, dataset: TabularDataset):
        if "target" in self.type:
            workload = self._generate_kway_target_workload(dataset)
        else:
            workload = self._generate_kway_uniform_workload(dataset)

        # Set workload seed for comparisons
        rng = np.random.default_rng(seed=self.seed)
        logger.info(f"Full workload size - {len(workload)}, seed={self.seed}")
        if self.num_marginals > 0:
            # Subsample
            indexes = rng.choice(
                len(workload), min(len(workload), self.num_marginals), replace=False
            )
            workload = [workload[i] for i in indexes]
        logger.debug(f"Final size of workload - {len(workload)}")
        return workload

    def _generate_kway_uniform_workload(
        self,
        dataset: TabularDataset,
        num_marginals: Optional[int] = None,
        k: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        num_marginals, k, seed = self._parse_workload_args(num_marginals, k, seed)

        # Generate of (k-1)-way marginals with the target attribute
        columns = dataset.df.columns.to_list()
        if "client_id" in columns:
            columns.remove("client_id")
        workload = [query for query in itertools.combinations(columns, k)]
        return workload

    def _generate_kway_target_workload(
        self,
        dataset: TabularDataset,
        num_marginals: Optional[int] = None,
        k: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        num_marginals, k, seed = self._parse_workload_args(num_marginals, k, seed)

        # Generate of (k-1)-way marginals with the target attribute
        columns = dataset.df.columns.to_list()
        columns.remove(dataset.y)
        if "client_id" in columns:
            columns.remove("client_id")
        workload = [
            (query[0], query[1], dataset.y)
            for query in itertools.combinations(columns, k - 1)
        ]
        return workload
