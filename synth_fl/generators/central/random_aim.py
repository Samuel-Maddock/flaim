import numpy as np

from synth_fl.generators import GeneratorConfig
from synth_fl.generators.central import AIM
from synth_fl.utils import logger
from typing import List, Tuple


class RandomAIM(AIM):
    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__(config)
        self.name = "random_aim"
        self.weight_by_size = True

    def _initialise_noise_params(self, rounds):
        gauss_sigma = np.sqrt(rounds / (2 * self.accountant.rho))
        return gauss_sigma, 0

    def _final_round(self, gauss_sigma, exp_epsilon, t):
        return self.accountant.rho < 2 * (0.5 / gauss_sigma**2)

    def _exp_query(
        self,
        client_answers,
        model,
        candidates,
        exp_eps,
        gauss_sigma,
        t=0,
        batch_queries=False,
    ):
        probs = None
        if self.weight_by_size:
            weights = 1 / np.array(list(candidates.values()))
            probs = weights / (weights.sum())

        return np.random.choice(list(candidates.keys()), p=probs)
