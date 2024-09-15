import numpy as np
import math

from synth_fl.generators import FederatedGenerator, FederatedGeneratorConfig
from synth_fl.generators.central import AIM
from synth_fl.utils.dataloaders import HorizontalSharder, TabularDataset
from synth_fl.libs.private_pgm.mbi import FactoredInference
from synth_fl.utils import logger

from collections import defaultdict


class DistributedAIM(AIM, FederatedGenerator):
    def __init__(
        self,
        config: FederatedGeneratorConfig,
    ) -> None:
        self.name = "distributedaim"
        self.match_data_shape = config.match_data_shape
        self.use_subsample_amplification = config.use_subsample_amplification
        self.clients_per_round = config.clients_per_round
        self.init_method = config.init_method

        if self.clients_per_round <= 1:
            self.clients_per_round = math.ceil(
                config.num_clients * self.clients_per_round
            )
        else:
            self.clients_per_round = config.num_clients

        # For accounting and subset selection (q)
        self.subsample_rate = (
            self.clients_per_round / config.num_clients
            if self.use_subsample_amplification
            else 1
        )
        if self.use_subsample_amplification:
            config.accountant_type = "rdp"
        AIM.__init__(self, config, subsample_rate=self.subsample_rate)
        FederatedGenerator.__init__(self, config)

        # TODO: If sampling_method = "subset" then AmplificationBySampling() needs PoissonSampling = False in autodp mechs
        # 'poisson' or 'subset', but accounting not supported for 'subset' yet
        self.sampling_method = "poisson"

        # self._log_init(config)
        self.log_internal_metrics = config.log_internal_metrics
        for metric in [
            "local_decisions",
        ]:
            self.internal_metrics[metric] = []
        self.client_participation_counter = defaultdict(int)

        # Only used when backend == "secagg"
        self.batch_queries = True

    # Logging / Debugging
    def _log_init(self, config):
        logger.debug(
            f"{self} initialised - num_clients={config.num_clients}, clients_per_round={self.clients_per_round}, sample_rate={self.subsample_rate}"
        )

    def _log_communication(self, client_ids, chosen_query, viable_candidates, model, t):
        if self.track_communication:
            logger.info(
                f"Full workload dim {sum([model.domain.size(q) for q in self.workload_closure])}"
            )
            for c_id in client_ids:
                mpspdz_factor = 1 if self.backend != "secagg" else 2 * 3
                self.client_communication_log[f"{c_id}_{t}_send_answers"] = (
                    sum([model.domain.size(q) for q in self.workload_closure])
                    * 8
                    * mpspdz_factor
                )
                self.server_communication_log[f"{c_id}_{t}_receive_answers"] = (
                    sum([model.domain.size(q) for q in self.workload_closure])
                    * 8
                    * mpspdz_factor
                )

    def _average_hetero(self, query, round_answers, all_client_answers, gauss_sigma):
        true_marginal = sum([ans[query] for ans in all_client_answers])
        round_marginals = [client[query] for client in round_answers]
        avg_marginal = np.mean(round_marginals, axis=0)

        return np.linalg.norm(
            self._normalise_measurement(true_marginal, 1)
            - self._normalise_measurement(avg_marginal, 1),
            1,
        )
        # for client in round_answers:
        #     round_marginal = client[query]
        #     tau_qs.append(
        #         np.linalg.norm(
        #             self._normalise_measurement(true_marginal, 1)
        #             - self._normalise_measurement(round_marginal, 1),
        #             1,
        #         )
        #     )
        # # weight = gauss_sigma / true_marginal.sum()
        # weight = 1
        # return np.mean(tau_qs) * weight

    def _commit_client_participation(self, round_answers):
        for client_answer in round_answers:
            self.client_participation_counter[client_answer[0]] += 1

    # Main
    def _subset_client_answers(self, client_answers, q=None):
        q = self.clients_per_round / len(client_answers) if q is None else q
        subset = np.array([])
        if self.sampling_method == "without_replacement":
            subset = np.random.choice(
                client_answers,
                self.clients_per_round,
                replace=False,  # todo: q is not used (!)
            )
        else:  # otherwise poisson sampling
            while subset.size == 0:
                subset = np.array(client_answers)[
                    np.random.choice(
                        [True, False],
                        p=[q, 1 - q],
                        size=len(client_answers),
                    )
                ]
        return subset

    def _initialise_model(
        self,
        dataset: TabularDataset,
        candidates,
        client_answers,
        gauss_sigma,
        warm_start=True,
        q=None,
    ):
        oneway = [cl for cl in candidates if len(cl) == 1]
        q = self.clients_per_round / len(client_answers) if q is None else q
        logger.debug(f"AIM - Initialising 1D marginals.. num={len(oneway)}, q={q}")

        if "resample" not in self.init_method:
            round_answers = self._subset_client_answers(client_answers, q=q)
        measurements = []  # concat measurements
        for query in oneway:
            if "resample" in self.init_method:
                round_answers = self._subset_client_answers(client_answers, q=q)
            client_ids, sampled_answers = zip(*round_answers)
            measurements += self._gauss_query(
                sampled_answers,
                [query],
                gauss_sigma,
                t=0,
            )
            if self.track_communication and self.backend != "secagg":
                for c_id in client_ids:
                    self.client_communication_log[f"{c_id}_{0}_send_init marginal"] += (
                        measurements[-1][1].size * 8
                    )
                    self.server_communication_log[
                        f"{c_id}_{0}_receive_init marginal"
                    ] += (measurements[-1][1].size * 8)

        if self.track_communication and self.backend == "secagg":  # secagg
            _ = self._mock_sec_agg(
                client_ids,
                client_answers=sampled_answers,
                candidates=oneway,
                log_msg="init",
                t=0,
            )

        logger.debug("AIM - Initialising 1D marginals... complete")
        if self.backend_model == "pgm":
            engine = FactoredInference(
                dataset.to_pgm_dataset().domain,
                iters=self.pgm_train_iters,
                warm_start=warm_start,
            )
        else:
            engine = RAPEngine(
                self._get_rap_config(dataset.to_rap_dataset(), candidates),
                candidates,
                dataset.domain,
                self.n,
            )

        model = engine.estimate(measurements, total=self.n)
        return model, measurements, engine

    def _aim_step(
        self,
        engine,
        model,
        measurements,
        client_answers,
        viable_candidates,
        exp_epsilon,
        gauss_sigma,
        t=1,
    ):
        subset_answers = self._subset_client_answers(
            client_answers, q=self.clients_per_round / len(client_answers)
        )
        self._commit_client_participation(subset_answers)
        client_ids, round_answers = zip(*subset_answers)  # (client ids, client answers)
        batch_query_ans = None
        logger.debug(f"Clients participating - {len(client_ids)} - {client_ids}")

        # Find worst-query via exp mech
        # if self.backend == "secagg":
        #     batch_query_ans = self._mock_sec_agg(
        #         client_ids, round_answers, viable_candidates, t=t, log_msg="exp"
        #     )
        if self.selection_method == "server_random":
            cl = np.random.choice(list(viable_candidates.keys()))
        else:
            cl = self._exp_query(
                round_answers,
                model,
                viable_candidates,
                exp_epsilon,
                gauss_sigma,
                t=t,
                batch_query_ans=batch_query_ans,
            )
        round_sample_size = sum([answer[cl] for answer in round_answers]).sum()
        logger.debug(f"Selected query: {cl}, sample_size={round_sample_size}")

        # Measure chosen query and add to measurements
        # if self.backend == "secagg":
        #     batch_query_ans = self._mock_sec_agg(
        #         client_ids, round_answers, [cl], t=t, log_msg="gauss"
        #     )
        measurements += self._gauss_query(
            round_answers, [cl], gauss_sigma, t=t, batch_query_ans=batch_query_ans
        )

        if "local_decisions" in self.log_internal_metrics:
            _, all_client_answers = zip(*client_answers)
            tau_q = self._average_hetero(
                cl, round_answers, all_client_answers, gauss_sigma
            )
            self.internal_metrics["local_decisions"].append(tau_q)

        # Log communication sent/receieved for this round
        self._log_communication(client_ids, cl, viable_candidates, model, t)

        # Calculate query improvement
        prev_model_est = model.project(cl).datavector()
        model = engine.estimate(measurements, total=self.n)
        new_model_est = model.project(cl).datavector()

        return model, prev_model_est, new_model_est, cl, measurements, round_sample_size

    def generate(
        self,
        dataset: TabularDataset,
        workload=None,
        num_clients: int = 2,
        non_iid: bool = "",
        **kwargs,
    ) -> TabularDataset:
        self.n = dataset.n
        self.target = dataset.y
        hfl_loader = HorizontalSharder(dataset, non_iid=non_iid)
        horizontal_datasets = hfl_loader.get_partition(num_clients)

        # Forces AIM (PGM) to generate n samples as in original df
        if self.match_data_shape:
            self.synth_rows = dataset.df.shape[0]

        result = self._generate(horizontal_datasets, workload)

        if "local_decisions" in self.log_internal_metrics:
            self.internal_metrics["mean_tau_q"] = np.mean(
                self.internal_metrics["local_decisions"]
            )
            self.internal_metrics["std_tau_q"] = np.std(
                self.internal_metrics["local_decisions"]
            )
            self.internal_metrics["local_decisions"] = []  # Save memory
            logger.warning(
                f"Average hetero of decisions - {self.internal_metrics['mean_tau_q']}"
            )

        # Cache communication metrics
        self._parse_communication_log()

        return TabularDataset(
            f"{self.name} {dataset.name}",
            None,
            result[1],
            dataset.domain,
        )
