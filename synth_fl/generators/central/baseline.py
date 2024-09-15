import numpy as np

from synth_fl.generators import Generator, GeneratorConfig
from synth_fl.utils.dataloaders import TabularDataset


class ZeroGenerator(Generator):
    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__(config=config)
        self.name = "zero_gen"
        self.type = "central"

    def generate(
        self, dataset: TabularDataset, workload: np.array, **kwargs
    ) -> TabularDataset:
        return None


class RealDataGenerator(Generator):
    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__(config=config)
        self.name = "real_data"
        self.type = "central"

    def generate(
        self, dataset: TabularDataset, workload: np.array, **kwargs
    ) -> TabularDataset:
        return dataset
