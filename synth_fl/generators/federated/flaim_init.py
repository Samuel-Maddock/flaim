from synth_fl.generators import FederatedGeneratorConfig
from synth_fl.generators.federated.flaim import FLAIM


class FLAIMInit(FLAIM):
    def __init__(self, config: FederatedGeneratorConfig) -> None:
        super().__init__(config=config)

        # Init only
        self.global_rounds = 0
        self.use_control_variates = False
        self.control_estimates = 0
        self.control_rounds = 0

        # Only estimates 1-way marginals once
        self.gauss_budget_alloc = 1
        self.exp_budget_alloc = 0
