from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, List

import jax.numpy as jnp
import numpy as np
import random

from synth_fl.utils.dataloaders import TabularDataset


@dataclass
class GeneratorConfig:
    """
    epsilon (float): Privacy budget (default=1)
    delta (float): Privacy parameter (default=1e-5)
    seed (int): Random seed, currently used only for PrivBayes, PGM (on its own), RAP
    aim_mech (str): Privacy mechanism for AIMs measurement step (default=gaussian)
    backend_model (str): AIM backend 'pgm' or 'rap' (default=pgm)
    pgm_train_iters (int): PGM optimisation iterations during training steps
    pgm_final_iters (int): Same as above but for the final PGM modl
    global_rounds (int): The number of AIM rounds (default = 0, use)
    normalize_measurements (bool): Whether to normalise measurements (used in FL)
    workload_type (str): Workload type from ['uniform', 'target'] (default=uniform)
    workload_seed (int): Workload seed (default=0)
    workload_k (int): Max k-way marginal (default=3)
    workload_num_marginals (int): Workload size (default=64)
    log_internal_metrics (str)
    log_decisions (bool): Whether to log certain training metrics for debugging (default=false)
    pgm_weight_method (str): How to weight measurements in PGM. Options from "sigma", "sigma_scaled" (default="sigma")
    gauss_budget_alloc (float): Budget allocation for gaussian mechanism in AIM
    anneal_type (str): Budget annealing condition for AIM
    save_client_answer_cache (bool): Whether to save client answers if not already cached (default=false)
    load_client_answer_cache (bool): Whether to load client answers if already cached (default=true)
    """

    epsilon: float = 1.0
    delta: float = 1e-5
    seed: int = 123
    aim_mech: str = "gaussian"
    backend_model: str = "pgm"
    pgm_train_iters: int = 30
    pgm_final_iters: int = 25
    global_rounds: int = 0
    normalize_measurements: bool = False
    workload_type: str = "uniform"
    workload_seed: int = 0
    workload_k: int = 3
    workload_num_marginals: int = 64
    log_internal_metrics: str = ""
    log_decisions: bool = False
    pgm_weight_method: str = "sigma"
    gauss_budget_alloc: float = 0.9
    selection_method: str = "exp"
    anneal_type: str = "majority"
    save_client_answer_cache: bool = False
    load_client_answer_cache: bool = True
    track_communication: bool = True


@dataclass
class FederatedGeneratorConfig(GeneratorConfig):
    """
    use_subsample_amplification (bool): Whether to use subsample amplification in RDP settings (default=false)
    num_clients (int): Total number of clients (default = 1)
    clients_per_round (float): Either an int (whole clients) or in [0,1] (probability) (default=1, e.g. all clients)
    local_rounds (int): Number of local rounds client perform (in FLAIM) (default=1)
    accounting_method (str): Accounting methods from 'global', 'ldp' (default="global")
    accountant_type (str): Accounting type from 'zcdp' or 'rdp' (default="zcdp")
    aggregation_method (str): Aggregation method in FLAIM from 'merge_all', 'average_all', 'aggregate_avg', 'fedavg' (default='merge_all')
    update_method (str): Method for how to update the global model in FLAIM option are '' or 'last_round' (default="")
    update_prune_method (str): Whether to prune recieved updates or not in FLAIM, options from '' or 'majority' (default = '', no _prune)
    client_prune_method (str): Whether to prune participating clients or not in FLAIM, options from '', 'sd_hetero', 'min_hetero' (defaul='', no prune)
    quality_score_type (str): Type of quality score to use, '' for default or 'no_covar' for no covariates (used for debugging)
    use_control_variates (bool): Whether or not to use covariates to augment local quality scores in FLAIM (default='false')
    control_type (str): Type of control variate to use from '', 'true_sub', 'private_sub', 'covering_sub', 'full_covering_sub' (can also use 'add' modifier), default = '' (no covars)
    control_rounds (int): Number of rounds to spend budget estimate covariates for (if 'private' in control_type), default=-1 (use global_rounds)
    control_estimates (int): Number of control estiamtes client send each round, default=-1 (send num_features for 'private')
    skip_hetero_metrics (bool): Whether to skip computing heterogeneity measure for non-IID datasets (default=true)
    match_data_shape (bool): Used in DistributedAIM, forces AIM to produce n samples that match training set n. Not needed if normalising (to n)
    backend (str): MPC backend, experimental/debug (default='')
    num_servers (int): Number of MPC servers experimental/debug (default=2)
    quantize_bits (int): Number of quantization bits (default=24)
    disable_quantization (bool): Whether to enable quantization in secagg mocking (default=false)
    """

    use_subsample_amplification: bool = False
    num_clients: int = 1
    clients_per_round: float = 1
    local_rounds: int = 1
    local_update_method: str = ""
    accounting_method: str = "global"
    accountant_type: str = "zcdp"
    aggregation_method: str = "merge_all"
    init_method: str = "distributed_resample"
    update_method: str = ""
    update_prune_method: str = ""
    client_prune_method: str = ""
    quality_score_type: str = ""
    use_control_variates: bool = False
    control_type: str = ""
    control_rounds: float = -1
    control_estimates: float = -1
    skip_hetero_metrics: bool = True
    match_data_shape: bool = False
    backend: str = ""
    num_servers: int = 2
    quantize_bits: int = 64
    disable_quantization: bool = False
    disable_local_candidate_filtering: bool = False


class Generator(ABC):
    def __init__(
        self,
        config: GeneratorConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.model_parameter_size, self.model_size_in_bytes = 0, 0
        self.type = "central"
        self.name = "generator"
        self.log_internal_metrics = config.log_internal_metrics
        self.internal_metrics = {}
        self.backend_model = config.backend_model  # model backend, pgm or rap
        self.save_client_answer_cache = config.save_client_answer_cache
        self.load_client_answer_cache = config.load_client_answer_cache
        self.track_communication = config.track_communication

        self.client_communication_log = defaultdict(int)  # Dict[str, int]
        self.server_communication_log = defaultdict(int)  # Dict[str, int]
        self.average_client_communication_sent = 0
        self.average_client_communication_received = 0
        self.total_server_communication_sent = 0
        self.total_server_communication_received = 0

    def _update_model_size(self, model_parameter_size):
        self.model_parameter_size = model_parameter_size
        self.model_size_in_bytes = 8 * model_parameter_size

        self.total_server_communication_received = self.model_size_in_bytes
        self.total_server_communication_sent = self.model_size_in_bytes

    @abstractmethod
    def generate(
        self, dataset: TabularDataset, workload=None, **kwargs
    ) -> TabularDataset:
        pass


class FederatedGenerator(Generator):
    def __init__(
        self,
        config: FederatedGeneratorConfig,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.type = "fl"
        self.backend = config.backend  # crypto backend

        # Base port for mpyc
        self.base_port = random.randint(1000, 65000)

        if self.backend == "secagg":
            pass # removed
        elif self.backend == "mpc":
            pass  # TODO: Initialise SecretSharer ?

    # --- General helpers ---

    def _parse_communication_log(self):
        send_client_map = defaultdict(int)
        receive_client_map = defaultdict(int)
        for log in self.client_communication_log:
            c_id, t, type, msg = log.split("_")
            if type == "send":
                send_client_map[c_id] += self.client_communication_log[log]
            else:
                receive_client_map[c_id] += self.client_communication_log[log]
        self.average_client_communication_sent = np.mean(list(send_client_map.values()))
        self.average_client_communication_received = np.mean(
            list(receive_client_map.values())
        )

        send_total, receive_total = 0, 0
        for log in self.server_communication_log:
            c_id, t, type, msg = log.split("_")
            if type == "send":
                send_total += self.server_communication_log[log]
            else:
                receive_total += self.server_communication_log[log]
        self.total_server_communication_sent = send_total
        self.total_server_communication_received = receive_total

    # Given a set of data items, batch into a single vector and return alongside indexes
    def _batch_data(self, data):
        batched_data = np.concatenate(data)
        indexes = [0]
        for x in data:
            indexes.append(indexes[-1] + len(x))

        return batched_data, indexes[1:-1]

    # Given batched data and indexes, return unbatched vector
    def _unbatch_data(self, batched_data, indexes):
        return np.split(batched_data, indexes)

    # Given a list of PGM datasets and a query, return the marginals for the query
    def _client_query_answers(self, horizontal_datasets: List[TabularDataset], query):
        return [td.project(query).datavector() for td in horizontal_datasets]

    # Internal method, actual logic for generating synth data
    @abstractmethod
    def _generate(self, horizontal_datasets, domain, workload):
        pass

    # Public method, partitions dataset into horizontal_datasets and call _generate
    @abstractmethod
    def generate(
        self, dataset: TabularDataset, workload=None, **kwargs
    ) -> TabularDataset:
        pass

    def generate_from_partition(
        self, horizontal_datasets: List[TabularDataset], domain=None, workload=None
    ):
        if not domain:
            domain = horizontal_datasets[0].domain

        return TabularDataset(
            self.name,
            None,
            self._generate(horizontal_datasets, domain, workload)[1],
            domain,
        )
