from typing import List, Tuple

import numpy as np

from synth_fl.generators import Generator, GeneratorConfig
from synth_fl.libs.ektelo import workload as ektelo_workload
from synth_fl.libs.private_pgm.examples.mwem import mwem
from synth_fl.utils.dataloaders import TabularDataset


class MWEM(Generator):
    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__(config=config)
        self.name="mwem"
        self.type = "central"

    def _format_workload(
        self, workload: np.array, domain: dict
    ) -> List[Tuple[np.array, int]]:
        lookup = {}
        for attr, val in domain.items():
            lookup[attr] = ektelo_workload.Prefix(val)

        workloads = []
        for proj in workload:
            W = ektelo_workload.Kronecker([lookup[a] for a in proj])
            workloads.append((proj, W))

        return workloads

    def generate(
        self, dataset: TabularDataset, workload: np.array, **kwargs
    ) -> TabularDataset:

        workload = self._format_workload(workload, dataset.domain)
        mwem_dataset = dataset.to_pgm_dataset()
        mwem_results = mwem(workload, mwem_dataset, self.config.epsilon)
        # self._update_model_size(self._aim.model.size)

        return TabularDataset(
            f"MWEM {dataset.name}", None, mwem_results[1], dataset.domain
        )
