from typing import List

import numpy as np
import pandas as pd

from synth_fl.generators import Generator, GeneratorConfig
from synth_fl.generators.central import CentralGeneratorFactory
from synth_fl.utils.dataloaders import HorizontalSharder, TabularDataset
from synth_fl.utils import logger


# Uses a central generator locally on each clients data before aggregating together - Naive FL
class NaiveHFLGenerator(Generator):
    def __init__(
        self,
        gen_name: str,
        config: GeneratorConfig,
    ) -> None:
        super().__init__(config)
        self.Generator = CentralGeneratorFactory().create_cls(gen_name)
        self.central_gen_name = gen_name
        self.type = "fl"
        self.model_size_in_bytes = []
        self.name = f"naive_hfl_{gen_name}"

    @property
    def average_client_communication(self):
        return np.mean(self.model_size_in_bytes)  # Sent

    @property
    def total_server_communication(self):
        return np.sum(self.model_size_in_bytes)  # Receieved

    def _generate(self, horizontal_datasets, domain, workload):
        synth_data_hfl = []

        for i, td in enumerate(horizontal_datasets):
            logger.newline()
            logger.debug(
                f"Training local generator ({self.central_gen_name}) for Client {i}..."
            )
            client_generator = self.Generator(self.config)
            synthetic_dataset = client_generator.generate(td, workload=workload)
            self.model_size_in_bytes.append(client_generator.model_size_in_bytes)
            synth_data_hfl.append(synthetic_dataset.df)

        return pd.concat(synth_data_hfl).reset_index(drop=True)

    def generate(
        self,
        dataset: TabularDataset,
        workload=None,
        num_clients: int = 2,
        non_iid: bool = False,
        **kwargs,
    ) -> TabularDataset:
        hfl_loader = HorizontalSharder(dataset, non_iid=non_iid)
        horizontal_datasets = hfl_loader.get_partition(num_clients)

        return TabularDataset(
            self.name,
            None,
            df=self._generate(horizontal_datasets, dataset.domain, workload),
            domain=dataset.domain,
        )

    def generate_from_partition(
        self, horizontal_datasets: List[TabularDataset], domain=None, workload=None
    ):
        if not domain:
            domain = horizontal_datasets[0].domain

        return TabularDataset(
            self.name,
            None,
            df=self._generate(horizontal_datasets, domain, workload),
            domain=domain,
        )
