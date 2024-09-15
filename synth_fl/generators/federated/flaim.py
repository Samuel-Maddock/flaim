import copy
import math
import numpy as np

from synth_fl.generators import FederatedGenerator, FederatedGeneratorConfig
from synth_fl.generators.central import AIM
from synth_fl.generators.federated import DistributedAIM
from synth_fl.utils.dataloaders import HorizontalSharder, TabularDataset
from synth_fl.libs.private_pgm.mbi import FactoredInference
from synth_fl.utils import logger

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List
from scipy.stats import pearsonr


@dataclass
class Update:
    client_id: int
    num_samples: int
    marginal: tuple
    measurement: np.array
    sigma: float


class FLAIM(DistributedAIM, FederatedGenerator):
    def __init__(self, config: FederatedGeneratorConfig) -> None:
        if config.accounting_method == "user":
            config.use_subsample_amplification = True
        DistributedAIM.__init__(self, config)
        config.backend = "none"
        self.name = "flaim"
        self.communication_log = defaultdict(int)

        self.accounting_method = config.accounting_method
        self.standard_noise_method = self.accounting_method in [
            "ldp",
            "distributed",
            "user",
        ]
        # todo: unsupported
        if not self.standard_noise_method:
            client_config.epsilon = 0  # Don't add local noise, manage globally

        self.aggregation_method = config.aggregation_method
        self.update_method = config.update_method
        self.update_prune_method = config.update_prune_method
        self.client_prune_method = config.client_prune_method
        self.quality_score_type = config.quality_score_type

        # Controls
        self.use_control_variates = config.use_control_variates
        self.control_type = config.control_type
        self.control_rounds = config.control_rounds  # No. rounds for server controls
        self.control_estimates = (
            config.control_estimates
        )  # No. estimates sent per round
        if self.control_type in ["", "filter"]:
            self.use_control_variates = False
            self.control_estimates = 0
            self.control_rounds = 0
        self.control_candidates = {}  # Control queries that need to be measured

        client_config = copy.deepcopy(config)
        client_config.log_decisions = False
        client_config.normalize_measurements = self.normalize_measurements
        self.client_aims = [
            AIM(client_config, disable_cdp_init=True) for i in range(config.num_clients)
        ]
        self.local_rounds = config.local_rounds
        self.local_update_method = config.local_update_method
        self.disable_local_candidate_filtering = (
            config.disable_local_candidate_filtering
        )
        self.expected_rounds = float("inf")

        # populated by generate/_generate()
        self.target = None
        self.gauss_sigma, self.exp_epsilon = None, None
        self.domain = None
        self.local_decision_map = {}  # todo: not logged to wandb (only decision_map is)
        self.marginal_counter = defaultdict(lambda: [None, None])
        self.weight_map = defaultdict(int)
        self.server_control_variates = {}
        self.client_true_variates = {}
        self.internal_metrics = defaultdict(list)
        self.internal_metrics["actual_rounds"] = 0
        self.skip_hetero_metrics = config.skip_hetero_metrics

        if self.init_method == "":
            logger.debug(
                f"No init method provided - using accounting default ({self.accounting_method})"
            )
            self.init_method = (
                "ldp_resample"
                if self.accounting_method == "ldp"
                else "distributed_resample"
            )

    def _subset_client_answers(self, client_answers, q=None):
        subset_answers = []
        while len(subset_answers) < 1:
            subset_answers = super()._subset_client_answers(
                client_answers=client_answers,
                q=q,
            )
            # Remove clients that exceeded expected num of participation rounds to prevent overspending privacy budget
            if "amplified" in self.accounting_method:
                subset_answers = list(
                    filter(
                        lambda x: (
                            self.client_participation_counter[x[0]]
                            < self.expected_rounds
                        ),
                        subset_answers,
                    )
                )
        # todo: committing should happen after pruning, but pruning currently unsupported
        self._commit_client_participation(subset_answers)
        return subset_answers

    # Init methods
    def _calibrate_privacy(self, global_rounds, t=0):
        global_rounds = global_rounds - t
        gauss_sigma = 1
        exp_epsilon = float("inf")
        logger.debug(f"Calibrating privacy - {self.accountant_type}, {self.accountant}")
        if (
            self.aggregation_method
            in [
                "average_all",
                "merge_all",
            ]
            or self.standard_noise_method
        ):
            local_rounds = (
                self.local_rounds if self.update_method != "last_update" else 1
            )
            total_gauss_rounds = global_rounds * local_rounds
            total_exp_rounds = total_gauss_rounds

        if "amplified" in self.accounting_method:
            self.expected_rounds = math.ceil(
                global_rounds * (self.clients_per_round / self.num_clients)
            )
            logger.debug(
                f"Amplifying ldp - number of expected rounds of participation {self.expected_rounds}"
            )
            total_gauss_rounds = self.expected_rounds * local_rounds
            total_exp_rounds = total_gauss_rounds

        if t == 0:
            total_gauss_rounds += self.num_oneways

        if "random" in self.selection_method:
            total_exp_rounds = 0

        covariate_rounds = 0
        if (
            self.use_control_variates
            and "private" in self.control_type
            and (
                self.quality_score_type != "no_covar" or self.client_prune_method != ""
            )
        ):
            # Default is d * T
            covariate_rounds += self.control_estimates * min(
                self.control_rounds, global_rounds - 1, self.expected_rounds
            )

        total_noisy_marginal_rounds = total_gauss_rounds + covariate_rounds
        if self.accountant_type == "rdp":
            gauss_sigma = self.gauss_mech.get_rdp_sigma(
                total_noisy_marginal_rounds,
                q=self.subsample_rate,
                epsilon=self.gauss_budget_alloc * self.accountant.epsilon,
            )
        else:
            gauss_sigma = np.sqrt(
                total_noisy_marginal_rounds
                / (2 * self.gauss_budget_alloc * self.accountant.rho)
            )
        if self.standard_noise_method and total_exp_rounds > 0:
            if self.accountant_type == "rdp":
                exp_epsilon = self.exp_mech.get_rdp_sigma(
                    total_exp_rounds,
                    q=self.subsample_rate,
                    epsilon=self.exp_budget_alloc * self.accountant.epsilon,
                )
            else:
                exp_epsilon = np.sqrt(
                    8 * self.exp_budget_alloc * self.accountant.rho / total_exp_rounds
                )
            logger.debug(
                f"Total gauss rounds = {total_gauss_rounds}, Total exp rounds = {total_exp_rounds}, Total covar rounds {covariate_rounds},  alloc=({self.gauss_budget_alloc}, {self.exp_budget_alloc})"
            )
        logger.debug(f"Noise calibrated - sigma={gauss_sigma}, exp_eps={exp_epsilon}")
        return gauss_sigma, exp_epsilon

    def _annealing_condition(self, est1, est2, domain_size, sigma=None):
        query_error = np.linalg.norm(est1 - est2, 1)
        if sigma is None:
            sigma = self.gauss_sigma
        # logger.debug(f"{query_error, self.gauss_sigma * np.sqrt(2 / np.pi) * domain_size}")
        return query_error <= self.gauss_sigma * np.sqrt(2 / np.pi) * domain_size

    def _initialise_model(
        self,
        dataset: TabularDataset,
        candidates,
        client_answers,
        gauss_sigma,
        warm_start=True,
        q=None,
    ):
        if "distributed" in self.init_method:
            logger.info(f"FLAIM - Distributed model init")
            return super()._initialise_model(
                dataset, candidates, client_answers, gauss_sigma, warm_start, q
            )

        # LDP initialisation
        logger.info(f"FLAIM - LDP model init")
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
            total_sample_size = sum(
                [sum(answers[query]) for answers in sampled_answers]
            )
            marginal_list = []
            for answers in sampled_answers:
                marginal_list.append(
                    self._noise_marginal(
                        answers[query],
                        gauss_sigma,
                        add_noise=True,
                        sensitivity=1,
                    )
                )

            calculate_sample_size = True
            if "ind" not in self.init_method:
                sample_size = (
                    np.sum(np.sum(marginal_list, axis=0))
                    if "n_est" in self.pgm_weight_method
                    else total_sample_size
                )
                marginal_list = [np.mean(marginal_list, axis=0)]
                calculate_sample_size = False

            for marginal in marginal_list:
                if calculate_sample_size:
                    sample_size = np.sum(marginal)
                weight = self._compute_measurement_weight(gauss_sigma, sample_size)
                if "squared" in self.pgm_weight_method:
                    weight = np.sqrt(weight)
                measurements.append(
                    [
                        None,
                        self._normalise_measurement(marginal, self.n),
                        weight,
                        query,
                    ]
                )

            if self.track_communication and self.backend != "secagg":
                for c_id in client_ids:
                    self.client_communication_log[f"{c_id}_{0}_send_init marginal"] += (
                        measurements[-1][1].size * 8
                    )
                    self.server_communication_log[
                        f"{c_id}_{0}_receive_init marginal"
                    ] += (measurements[-1][1].size * 8)

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
        self.gauss_mech.accumulate(gauss_sigma, len(oneway))
        model = engine.estimate(measurements, total=self.n)
        return model, measurements, engine

    def _get_model(self, dataset: TabularDataset, candidates, client_answers, q=1):
        model, measurements, engine = self._initialise_model(
            dataset, candidates, client_answers, gauss_sigma=self.gauss_sigma, q=q
        )

        for i, cl in enumerate(self.oneways):
            self.marginal_counter[cl] = [i, 1]  # index, count
            self.weight_map[cl] = dataset.n  # Weight for fedavg

        # Cache initialised marginal measurements as server control variates
        # (I, measurement, sigma, cl)
        for i, m in enumerate(measurements):
            _, measurement, _, cl = m
            self.server_control_variates[cl] = [measurement, 1]

        return model, measurements, engine

    def _get_client_workload(
        self,
        current_model,
        client_answers,
        viable_candidates,
        client_control_variates=None,
        t=0,
    ):
        client_candidates = viable_candidates
        if self.config.workload_type == "marginal_target":
            client_candidates = {}
            one_way_errs = []
            one_ways = [cl for cl in viable_candidates if len(cl) == 1]

            # Compute errors on marginals
            for cl in one_ways:
                local_ans = self._marginal_to_prob(client_answers[cl])
                model_ans = self._marginal_to_prob(
                    current_model.project(cl).datavector()
                )
                err = np.abs(model_ans - local_ans).sum() / local_ans.size
                one_way_errs.append((cl, err))

            # Filter candidates that only contain top-k marginals and target
            sorted_errs = sorted(one_way_errs, key=lambda tup: tup[1], reverse=True)
            targets = set(list(zip(*sorted_errs[:5]))[0])
            targets.add((self.target,))
            for cl in viable_candidates:
                client_candidates[cl] = viable_candidates[cl]
                for attr in cl:
                    if (attr,) not in targets:
                        client_candidates.pop(cl)
                        break

        # Apply additional filtering based on non-IIDness from previous round
        if client_control_variates and self.control_type == "local_filter" and t > 0:
            client_candidates = {}
            sorted_attrs = sorted(
                client_control_variates,
                key=lambda x: abs(client_control_variates.get(x)),
                reverse=False,
            )
            allowed_attrs = sorted_attrs[: len(sorted_attrs) // 2]
            for cl in viable_candidates:
                for attr in cl:
                    if attr not in allowed_attrs:
                        break
                client_candidates[cl] = viable_candidates[cl]

        if "filter" in self.control_type:
            client_candidates = {}
            for cl in viable_candidates:
                if cl not in self.oneways:
                    client_candidates[cl] = viable_candidates[cl]
            logger.debug(
                f"Candidate workload filtered - {len(viable_candidates)} to {len(client_candidates)}"
            )

        return client_candidates

    def _compute_measurement_weight(self, sigma, sample_size=1):
        weight = (
            sigma * 1 / sample_size
            if "sigma_scaled" in self.pgm_weight_method
            else sigma
        )
        if "nweight" in self.pgm_weight_method:
            weight *= self.n
        if "squared" in self.pgm_weight_method:
            weight = weight**2
        return weight

    # Update methods
    def _client_update(
        self,
        engine,
        model,
        measurements,
        client_answers,
        client_control_variates,
        client_dataset,
        viable_candidates,
        client_id=1,
        t=1,
        initial_model_estimates=None,
        cl=None,
    ):
        aim = self.client_aims[client_id]
        client_updates = []
        if self.local_update_method == "no_prior_measurements":
            client_measurements = []
        else:
            client_measurements = measurements.copy()
        local_engine = copy.deepcopy(engine)

        client_candidates = self._get_client_workload(
            model,
            client_answers,
            viable_candidates,
            client_control_variates=client_control_variates,
            t=t,
        )

        # TODO: Currently control variates updated after workload is filtered (!)
        # If workload is filtered based on control variates, do not add local optim term to exp_mech
        if (
            self.control_type == "local_filter"
            or self.quality_score_type == "no_covar"
            or "no_covar" in self.control_type
        ):
            client_control_variates = None

        # (I, y, gauss_sigma, cl)
        # global_model = copy.deepcopy(model)
        # suppress_gauss_query = (
        #     True if self.accounting_method in ["distributed", "user"] else False
        # )
        suppress_gauss_query = True if self.accounting_method in ["user"] else False

        for l in range(self.local_rounds):
            # Performance fix - only filter candidates if more than 1 local round
            if l > 0 and not self.disable_local_candidate_filtering:
                size_limit = self._get_size_limit(t=t + l)
                client_candidates = self._filter_candidates(
                    client_candidates, model, size_limit
                )
            model_estimates = None if l > 0 else initial_model_estimates
            model, _, _, query, client_measurements, _ = aim._aim_step(
                local_engine,
                model,
                client_measurements,
                [client_answers],
                client_candidates,
                exp_epsilon=self.exp_epsilon,
                gauss_sigma=self.gauss_sigma,
                n=self.n,
                control_variates=client_control_variates,
                suppress_model_update=(l == self.local_rounds - 1),
                suppress_estimates=True,
                suppress_gauss_query=suppress_gauss_query,
                model_estimates=model_estimates,
                random_query=self.selection_method == "local_random",
                cl=cl,
                disable_weighting=False,
            )

            # Debugging - log local decisions
            if "local_decisions" in self.log_internal_metrics:
                tau_q = self.client_true_variates[client_id][query]
                self.internal_metrics["local_decisions"].append(tau_q)
                if l == self.local_rounds - 1:
                    logger.warning(
                        f"Client hetero - {self.internal_metrics['local_decisions'][-self.local_rounds:]}"
                    )
                # self.internal_metrics["local_decisions"].append(
                #     [client_id, l, query, tau_q]
                # )

            sample_size = client_dataset.df.shape[0]
            if "n_est" in self.pgm_weight_method:
                sample_size = client_measurements[-1][2]

            pgm_sigma = self._compute_measurement_weight(
                self.gauss_sigma, sample_size=sample_size
            )

            # TODO: If distributed accounting, then replace measurements with unnoised versions
            # Noise is added after in a "secagg" routine
            measured_marginal = client_measurements[-1][1]
            if "distributed" in self.accounting_method:
                measured_marginal = client_answers[client_measurements[-1][3]]

            client_updates.append(
                Update(
                    client_id,
                    client_dataset.df.shape[0],
                    query,
                    measured_marginal,
                    pgm_sigma,
                )
            )

        if self.update_method == "last_update":
            client_updates = [client_updates[-1]]

        return client_updates

    def _noise_client_updates(self, client_updates, t=-1):
        if self.accounting_method in ["distributed", "user"]:
            new_updates = {}
            for client_update in client_updates:
                for update in client_update:
                    measurement = (
                        update.measurement
                        if self.accounting_method == "distributed"
                        else self._normalise_measurement(update.measurement, n=1)
                    )
                    if update.marginal in new_updates:
                        agg_update = new_updates[update.marginal]
                        new_updates[update.marginal] = [
                            agg_update[0] + [measurement],
                            agg_update[1] + update.num_samples,
                            agg_update[2] + [update.client_id],
                        ]
                    else:
                        new_updates[update.marginal] = [
                            [measurement],
                            update.num_samples,
                            [update.client_id],
                        ]
            client_updates = []
            for marginal, updates in new_updates.items():
                measurements, samples, client_ids = updates
                # weights = (
                #     np.sum(measurements, axis=1)
                #     if self.aggregation_method == "fedavg"
                #     else None
                # )
                weights = None
                agg_measure = np.average(measurements, weights=weights, axis=0)
                priv_measure = self.gauss_mech.apply(
                    agg_measure,
                    sigma=self.gauss_sigma,
                    sensitivity=1 / len(measurements),
                    no_accumulate=True,
                )
                # priv_measure = np.clip(priv_measure, 0, 2)
                n_est = priv_measure.sum() * len(measurements)
                priv_measure = self._normalise_measurement(priv_measure, n=self.n)
                if "n_est" in self.pgm_weight_method:
                    samples = n_est
                weight = self._compute_measurement_weight(
                    self.gauss_sigma, sample_size=samples
                )
                client_updates.append(
                    [Update(-1, samples, marginal, priv_measure, weight)]
                )
                if (
                    self.track_communication
                    and self.backend == "secagg"
                    and len(measurements) > 1
                ):
                    pass
                    # batched_measures = [[x] for x in measurements]
                    # _ = self._mock_sec_agg(
                    #     client_ids,
                    #     None,
                    #     None,
                    #     client_data=batched_measures,
                    #     log_msg="FLAIMround",
                    #     t=t,
                    # )
                elif (
                    self.track_communication
                    and self.backend == "secagg"
                    and len(measurements) == 1
                ):  # send ldp, no secagg needed
                    client_id = client_ids[0]
                    self.client_communication_log[f"{client_id}_{t}_send_marginal"] += (
                        measurements[0].size * 8
                    ) + 8
                    self.server_communication_log[
                        f"{client_id}_{t}_receive_marginal"
                    ] += (measurements[0].size * 8) + 8

        return client_updates

    def _update_measurements(
        self, client_updates, current_measurements, add_noise=False
    ):
        sigmas = {}
        # Send all measured queries straight to PGM
        if self.aggregation_method == "merge_all":
            # Convert all updates to measurements
            current_measurements.extend(
                self._updates_to_measurements(
                    [
                        update
                        for client_update in client_updates
                        for update in client_update
                    ],
                    add_noise=add_noise,
                )
            )
            sigmas = {
                update.marginal: update.sigma
                for client_update in client_updates
                for update in client_update
            }
        # Aggregate queries by averaging or weighting by local sample sizes (fedavg)
        elif self.aggregation_method in ["aggregate_avg", "fedavg"]:
            for client_update in client_updates:
                for update in client_update:
                    if self.marginal_counter[update.marginal][1]:
                        m_idx, m_count = self.marginal_counter[update.marginal]
                        measurement = current_measurements[m_idx]

                        multiplicative_factor = (
                            self.weight_map[update.marginal]
                            if self.aggregation_method == "fedavg"
                            else m_count
                        )

                        additive_factor = (
                            update.measurement * update.num_samples
                            if self.aggregation_method == "fedavg"
                            else update.measurement
                        )

                        additive_noise_factor = (
                            update.sigma * update.num_samples**2
                            if self.aggregation_method == "fedavg"
                            else update.sigma
                        )

                        divisor = (
                            self.weight_map[update.marginal] + update.num_samples
                            if self.aggregation_method == "fedavg"
                            else m_count + 1
                        )

                        measurement[1] = (
                            measurement[1] * multiplicative_factor + additive_factor
                        ) / divisor

                        if "squared" in self.pgm_weight_method:
                            measurement[2] = (
                                math.sqrt(
                                    (measurement[2] * multiplicative_factor) ** 2
                                    + additive_noise_factor
                                )
                            ) / divisor

                        self.marginal_counter[update.marginal][1] += 1
                        self.weight_map[update.marginal] += update.num_samples
                        logger.debug(
                            f"{update.marginal} - New sigma={measurement[2]}, count={self.marginal_counter[update.marginal][1]}"
                        )
                        sigmas[update.marginal] = measurement[2]
                    else:
                        update.sigma = (
                            math.sqrt(update.sigma)
                            if "squared" in self.pgm_weight_method
                            else update.sigma
                        )
                        current_measurements.extend(
                            self._updates_to_measurements([update], add_noise=add_noise)
                        )
                        self.weight_map[update.marginal] = update.num_samples
                        self.marginal_counter[update.marginal] = [
                            len(current_measurements) - 1,
                            1,
                        ]
                        sigmas[update.marginal] = update.sigma
        else:
            raise NotImplementedError(
                f"Update type='{self.aggregation_method}' is not implemented"
            )

        return current_measurements, sigmas

    def _update_global_model(
        self, engine, model, current_measurements, client_updates, final_round=False
    ):
        add_noise = self.config.epsilon > 0 and not self.standard_noise_method
        if add_noise:
            num_client_updates = len(client_updates[0])
            if self.aggregation_method == "average_most_common":
                num_client_updates = 1
            self.gauss_mech.accumulate(self.gauss_sigma, num_client_updates)

        # Get list of marginals measured by all clients
        current_round_queries = set()
        for client_update in client_updates:
            for update in client_update:
                current_round_queries.add(update.marginal)

        # Aggregate client updates to the current measurements list
        current_measurements, sigmas = self._update_measurements(
            client_updates, current_measurements, add_noise=add_noise
        )

        prev_ests, new_ests = {}, {}
        if self.adaptive_noise:
            for cl in current_round_queries:
                prev_ests[cl] = model.project(cl).datavector()
        logger.debug(f"Estimating new model after merging measurements...")
        if final_round:
            engine.iters = self.pgm_final_iters
        model = engine.estimate(current_measurements, total=self.n)
        logger.debug(f"New model estimated")
        if self.adaptive_noise:
            for cl in current_round_queries:
                new_ests[cl] = model.project(cl).datavector()

        return model, current_measurements, prev_ests, new_ests, sigmas

    # unsupported
    def _prune_updates(self, client_updates):
        return client_updates

    # Control variates
    def _compute_client_control_variates(
        self, client_id, client_answers, viable_candidates
    ):
        client_control_variates, covering_estimates = None, None
        if self.use_control_variates:
            if "true" in self.control_type:
                client_control_variates = {
                    cl: self.client_true_variates[client_id][cl]
                    for cl in viable_candidates
                }
            else:
                server_estimates = self.server_control_variates

                client_control_variates = {
                    cl: np.linalg.norm(
                        self._normalise_measurement(server_estimates[cl][0], n=1)
                        - self._normalise_measurement(client_answers[cl], n=1),
                        ord=1,
                    )
                    for cl in self.oneways
                }
                client_control_variates = {
                    cl: np.mean([client_control_variates[(attr,)] for attr in cl])
                    for cl in viable_candidates
                }

            if "sub" in self.control_type:
                client_control_variates = {
                    cl: -1 * c for cl, c in client_control_variates.items()
                }
        return client_control_variates

    def _update_server_control_variate(
        self,
        current_measurements,
        client_subset,
        subset_sample_size,
        current_round,
        final_round=False,
    ):
        stopping_list = []
        if final_round:
            return current_measurements, stopping_list

        if (
            self.use_control_variates
            and "true" not in self.control_type
            and (
                self.quality_score_type != "no_covar" or self.client_prune_method != ""
            )
            and current_round < self.control_rounds
        ):
            # Sample marginals from control candidates based on number of control_estimates
            feature_list = np.empty(len(self.control_candidates), dtype="object")
            for i, f in enumerate(self.control_candidates):
                if type(f) == str:
                    feature_list[i] = (f,)
                else:
                    feature_list[i] = tuple(f)

            client_submissions = {
                client_id: np.random.choice(
                    feature_list, size=self.control_estimates, replace=False
                )
                for client_id, _ in client_subset
            }

            # For each feature with client submissions aggregate and add noise
            round_oneways = defaultdict(list)
            stopping_list = []
            for cl in self.control_candidates:
                measure, weight = self.server_control_variates[cl]
                client_oneways = []
                for client_id, answers in client_subset:
                    for submitted_query in client_submissions[client_id]:
                        if submitted_query == cl:
                            sample_size = sum(answers[cl])
                            if self.accounting_method == "distributed":
                                client_oneways.append((answers[cl], sample_size))
                            else:
                                noisy_marginal = self._noise_marginal(
                                    answers[cl],
                                    self.gauss_sigma,
                                    add_noise=True,
                                    sensitivity=1,
                                )
                                sample_size = (
                                    noisy_marginal.sum()
                                    if "n_est" in self.pgm_weight_method
                                    else sample_size
                                )
                                client_oneways.append(
                                    (
                                        noisy_marginal,
                                        sample_size,
                                    )
                                )

                            # Log communication
                            if self.track_communication and self.backend != "secagg":
                                self.client_communication_log[
                                    f"{client_id}_0_send_covariate"
                                ] += (answers[cl].size * 8)
                                self.server_communication_log[
                                    f"{client_id}_0_receive_covariate"
                                ] += (answers[cl].size * 8)

                # If no clients contributed the marginal cl then skip updating
                if len(client_oneways) == 0:
                    print(cl, "skipped")
                    continue

                client_measurements, weights = zip(*client_oneways)
                # weights = None if self.aggregation_method != "fedavg" else weights
                weights = None
                client_ans_avg = np.average(
                    client_measurements, weights=weights, axis=0
                )
                if (
                    "private" in self.control_type
                    and self.accounting_method == "distributed"
                ):
                    client_ans_avg = self._noise_marginal(
                        client_ans_avg,
                        self.gauss_sigma,
                        add_noise=True,
                        sensitivity=1 / len(client_oneways),
                    )
                    n_est = client_ans_avg.sum() * len(client_oneways)

                client_ans_avg = self._normalise_measurement(client_ans_avg, self.n)
                # new_weight = (
                #     1 if self.aggregation_method != "fedavg" else subset_sample_size
                # )
                new_weight = 1
                if current_round == 0:
                    self.server_control_variates[cl] = [
                        client_ans_avg,
                        new_weight,
                    ]
                else:
                    logger.debug(f"Server covariate update {cl} - {weight + 1}")
                    old_control_variate = self.server_control_variates[cl][0]

                    self.server_control_variates[cl] = [
                        ((new_weight * client_ans_avg) + (weight * measure))
                        / (weight + new_weight),
                        weight + new_weight,
                    ]

                    if "covar" in self.anneal_type and self._annealing_condition(
                        self.server_control_variates[cl][0],
                        old_control_variate,
                        self.domain.size(cl),
                        sigma=self.gauss_sigma / math.sqrt(self.clients_per_round),
                    ):
                        logger.info(f"Adding {cl} to stopping list")
                        stopping_list.append(cl)

                if "distributed" in self.accounting_method:
                    subset_sample_size_est = n_est
                else:
                    _, weights = zip(*client_oneways)
                    subset_sample_size_est = np.sum(weights)

                sample_size = (
                    subset_sample_size_est
                    if "n_est" in self.pgm_weight_method
                    else subset_sample_size
                )
                round_oneways[cl] = [(client_ans_avg, sample_size)]
                # round_oneways[cl] = [
                #     (self._normalise_measurement(x[0], self.n), x[1])
                #     for x in client_oneways
                # ]
                # round_oneways[cl] = [(self.server_control_variates[cl][0], subset_sample_size)]

            if "private" in self.control_type:
                self.gauss_mech.accumulate(self.gauss_sigma, self.control_estimates)

            # Optional - If private covariates being used, then add oneways to measurements to estimate new model
            if "private_combine" in self.control_type:
                covariate_updates = []
                for cl in round_oneways:
                    for marginal, sample_size in round_oneways[cl]:
                        weight = self._compute_measurement_weight(
                            self.gauss_sigma, sample_size=sample_size
                        )
                        covariate_updates.append(
                            [
                                Update(
                                    -1,
                                    sample_size,
                                    cl,
                                    marginal,
                                    weight,
                                )
                            ]
                        )
                current_measurements, _ = self._update_measurements(
                    covariate_updates, current_measurements, add_noise=False
                )

                # secagg
                if self.track_communication and self.backend == "secagg":
                    pass
                    # client_ids, client_answers = zip(*client_subset)
                    # _ = self._mock_sec_agg(
                    #     client_ids,
                    #     client_answers,
                    #     self.oneways,
                    #     log_msg="covariate",
                    #     t=current_round,
                    # )

        return current_measurements, stopping_list

    def _update_control_candidates(self, t, stopping_list):
        if stopping_list and "private" in self.control_type:
            logger.warning(f"Covariate stopping list - {stopping_list}")
            self.control_candidates = list(
                filter(lambda x: x not in stopping_list, self.control_candidates)
            )
            self.control_estimates = len(self.control_candidates)
            if not self.adaptive_noise:
                self._calibrate_privacy(self.global_rounds, t=t + 1)

    # Debugging heterogeneity

    def _cache_client_hetero(self, client_answers, workload):
        client_dict = defaultdict(dict)
        for cl in workload:
            for i, answer in client_answers:
                client_answer = self._normalise_measurement(answer[cl], n=1)
                true_answer = self._normalise_measurement(
                    sum([ans[cl] for _, ans in client_answers]), n=1
                )
                client_dict[i][cl] = np.linalg.norm(client_answer - true_answer, ord=1)

        return client_dict

    def _debug_variates(
        self,
        client_id,
        client_control_variates,
        client_answers,
        client_subset,
        viable_candidates,
        model,
        t=1,
        verbose=True,
    ):

        if "hetero_errors" in self.log_internal_metrics:
            client_control_variate_arr = np.array(
                [client_control_variates[cl] for cl in viable_candidates]
            )
            true_control_variate_arr = -1 * np.array(
                [self.client_true_variates[client_id][cl] for cl in viable_candidates]
            )

            # Quality scores of local clients (no variate + variate)
            client_errors = {}
            client_errors_variate = {}
            for i, cl in enumerate(viable_candidates):
                client_errors[cl] = self._compute_quality_score(
                    cl, model, 0, [client_answers], viable_candidates, self.gauss_sigma
                )
                client_errors_variate[cl] = self._compute_quality_score(
                    cl,
                    model,
                    client_control_variates[cl],
                    [client_answers],
                    viable_candidates,
                    self.gauss_sigma,
                )

            max_hetero_cl = max(
                self.client_true_variates[client_id],
                key=self.client_true_variates[client_id].get,
            )

            priv_errors = {k: [] for k in range(1, self.k + 1)}
            true_errors = {k: [] for k in range(1, self.k + 1)}
            for cl in viable_candidates:
                true_var = -1 * self.client_true_variates[client_id][cl]
                client_var = client_control_variates[cl]
                error = abs(true_var - client_var)

                priv_errors[len(cl)].append(error)
                true_errors[len(cl)].append(self.client_true_variates[client_id][cl])

            for k in range(1, self.k + 1):
                self.internal_metrics["hetero_errors"].append(
                    (
                        t,
                        client_id,
                        k,
                        np.min(priv_errors[k]),
                        np.max(priv_errors[k]),
                        np.mean(priv_errors[k]),
                        np.std(priv_errors[k]),
                        np.min(true_errors[k]),
                        np.max(true_errors[k]),
                        np.mean(true_errors[k]),
                        np.std(true_errors[k]),
                    )
                )

            if verbose:
                logger.debug(
                    f"Client id={client_id} corr={pearsonr(client_control_variate_arr, true_control_variate_arr)}"
                )
                logger.debug(
                    f"Max heterogeneity cl={max_hetero_cl}, {self.client_true_variates[client_id][max_hetero_cl]}"
                )
                logger.debug(
                    f"Average heterogeneity cl={np.mean(list(self.client_true_variates[client_id].values()), axis=0)}"
                )
                full_priv_errors = np.concatenate(list(priv_errors.values()))
                logger.debug(f"Max error 1-way - {np.max(priv_errors[1])}")
                logger.debug(f"Average error 1-way - {np.mean(priv_errors[1])}")
                logger.debug(f"Max error 2-way - {np.max(priv_errors[2])}")
                logger.debug(f"Average error 2-way - {np.mean(priv_errors[2])}")
                logger.debug(f"Max error 3-way - {np.max(priv_errors[3])}")
                logger.debug(f"Average error 3-way - {np.mean(priv_errors[3])}")
                logger.debug(f"Max error all - {np.max(full_priv_errors)}")
                logger.debug(f"Average error all - {np.mean(full_priv_errors)}")
                logger.info("\n")

        if "quality_rank" in self.log_internal_metrics:
            subset_errors = {}
            population_errors = {}
            for i, cl in enumerate(viable_candidates):
                subset_errors[cl] = self._compute_quality_score(
                    cl,
                    model,
                    0,
                    [answers[1] for answers in client_subset],
                    viable_candidates,
                    self.gauss_sigma,
                )
                population_errors[cl] = self._compute_quality_score(
                    cl,
                    model,
                    0,
                    [answers[1] for answers in client_answers],
                    viable_candidates,
                    self.gauss_sigma,
                )
            # Quality scores of current client subset
            subset_errors = {cl: subset_errors[cl] for cl in viable_candidates}

            # Quality score of whole population
            population_errors = {cl: population_errors[cl] for cl in viable_candidates}

            for str_msg, name, e1, e2 in [
                (
                    "population (no variate)",
                    "population_errors",
                    client_errors,
                    population_errors,
                ),
                (
                    "population (w/ variate)",
                    "population_variate_errors",
                    client_errors_variate,
                    population_errors,
                ),
                (
                    "subsample (no variate)",
                    "subset_errors",
                    client_errors,
                    subset_errors,
                ),
                (
                    "subsample (w/ variate)",
                    "subset_variate_errors",
                    client_errors_variate,
                    subset_errors,
                ),
            ]:
                error = np.mean(
                    [abs(e1[cl] - e2[cl]) for cl in viable_candidates],
                    axis=0,
                )
                self.internal_metrics[name].append((t, client_id, error))
                logger.debug(f"Expected errors vs {str_msg} = {error}")

            subset_ranks = {
                item[0]: i
                for i, item in enumerate(Counter(subset_errors).most_common())
            }
            population_ranks = {
                item[0]: i
                for i, item in enumerate(Counter(population_errors).most_common())
            }
            client_ranks = {
                item[0]: i
                for i, item in enumerate(Counter(client_errors).most_common())
            }
            client_variate_ranks = {
                item[0]: i
                for i, item in enumerate(Counter(client_errors_variate).most_common())
            }

            total_client_rank_err, total_client_variate_rank_err = 0, 0
            total_client_rank_err_subset, total_client_variate_rank_err_subset = 0, 0
            for item in population_ranks:
                total_client_rank_err += abs(
                    population_ranks[item] - client_ranks[item]
                )
                total_client_variate_rank_err += abs(
                    population_ranks[item] - client_variate_ranks[item]
                )
                total_client_rank_err_subset += abs(
                    subset_ranks[item] - client_ranks[item]
                )
                total_client_variate_rank_err_subset += abs(
                    subset_ranks[item] - client_variate_ranks[item]
                )

            for metric, val in [
                ("population_quality_rank", total_client_rank_err),
                ("population_variate_quality_rank", total_client_variate_rank_err),
                ("subset_quality_rank", total_client_rank_err_subset),
                ("subset_variate_quality_rank", total_client_variate_rank_err_subset),
            ]:
                self.internal_metrics[metric].append(
                    (t, client_id, val / len(population_ranks))
                )

            logger.debug(
                f"Average client ranking error: {total_client_rank_err/len(population_ranks)}"
            )
            logger.debug(
                f"Average client variate ranking error: {total_client_variate_rank_err/len(population_ranks)}"
            )

    # Helper methods
    def _filter_client_answers(self, client_subset, client_control_variates):
        filtered_client_subset, filtered_client_control_variates = (
            client_subset,
            client_control_variates,
        )
        return filtered_client_subset, filtered_client_control_variates

    def _marginal_to_prob(self, marginal: np.array):
        return marginal.astype("float32") / marginal.sum()

    def _noise_marginal(self, x, gauss_sigma, add_noise=False, sensitivity=1):
        if add_noise:
            return self.gauss_mech.apply(
                x, gauss_sigma, no_accumulate=True, sensitivity=sensitivity
            )
        else:
            return x

    def _updates_to_measurements(
        self,
        client_updates: List[Update],
        add_noise=False,
        track_count=False,
    ):
        if not isinstance(client_updates, list):
            if len(client_updates) == 3:
                client_updates = [client_updates]
            else:
                raise TypeError(
                    f"Client updates has wrong type and shape - {type(client_updates), len(client_updates)}"
                )

        # Shape of updates - (client_id, marginal, measurement, sigma)
        # Required shape of measurements - (I, measurement, gauss_sigma, marginal)
        measurements = []
        for update in client_updates:
            gauss_sigma = update.sigma if update.sigma else self.gauss_sigma
            if (
                self.aggregation_method == "merge_all"
                and "squared" in self.pgm_weight_method
            ):
                gauss_sigma = math.sqrt(gauss_sigma)
            if track_count:
                self.marginal_counter[update.marginal] += 1
            measurements.append(
                [
                    None,  # I, set as None inferred by fix_measurements in FactoredInference
                    self._noise_marginal(
                        update.measurement, gauss_sigma, add_noise=add_noise
                    ),  # y
                    gauss_sigma,  # sigma
                    update.marginal,  # cl
                ]
            )
        return measurements

    def _log_measurements(self, measurements, t=0):
        # measurements - (I, y, sigma, cl)
        if self.log_decisions:
            for measurement in measurements:
                self.decision_map[measurement[3]].append((t, measurement[2]))

    def _init_control_params(self, client_answers, candidates):
        # Compute number of control rounds where users submit controls
        if self.control_rounds > 0 and self.control_rounds < 1:
            self.control_rounds = round(self.global_rounds * self.control_rounds)
        else:
            self.control_rounds = (
                self.control_rounds
                if self.control_rounds >= 0
                else self.global_rounds - 1
            )

        # Compute number of control estimates sent by a user per control round
        if self.control_estimates > 0 and self.control_estimates < 1:
            self.control_estimates = round(self.num_oneways * self.control_estimates)
        else:
            self.control_estimates = (
                len(self.control_candidates)
                if self.control_estimates < 0
                else self.control_estimates
            )

        # Initialise true variates
        if "true" in self.control_type or self.log_internal_metrics != "":
            self.client_true_variates = self._cache_client_hetero(
                client_answers, workload=candidates
            )
        logger.info(
            f"Control type: {self.control_type}, Control rounds: {self.control_rounds}, Control estimates: {self.control_estimates}"
        )

    def _log_communication_round(self, client_updates, model, t):
        if self.track_communication:
            for update_list in client_updates:
                c_id = update_list[0].client_id
                for update in update_list:
                    self.client_communication_log[f"{c_id}_{t+1}_send_decision"] = 8
                    if self.backend != "secagg":
                        self.client_communication_log[
                            f"{update.client_id}_{t}_send_marginal"
                        ] += (update.measurement.size * 8) + 8
                        self.server_communication_log[
                            f"{update.client_id}_{t}_receive_marginal"
                        ] += (update.measurement.size * 8) + 8

    # Generate methods
    def _generate(self, horizontal_datasets, workload):
        # Convert datasets to PGM datasets to use .project()
        original_dataset = horizontal_datasets[0]
        horizontal_datasets = [
            dataset.to_pgm_dataset() for dataset in horizontal_datasets
        ]

        # Get domain, workload and cache client answers
        self.domain = horizontal_datasets[0].domain
        candidates = self.client_aims[0]._compile_workload(workload)
        self.oneways = [cl for cl in candidates if len(cl) == 1]
        self.num_oneways = len(self.oneways)

        # Compute control candidates
        self.control_candidates = self.oneways

        # Workload + control candidate answers need to be cached
        control_candidates_dict = {k: "" for k in self.control_candidates}
        client_answers = self.client_aims[0]._compute_client_answers(
            horizontal_datasets,
            control_candidates_dict | candidates,
            original_dataset,
            save_cache=self.save_client_answer_cache,
            load_cache=self.load_client_answer_cache,
        )

        # Add client IDs
        client_answers = [(i, answer) for i, answer in enumerate(client_answers)]

        # Calibrate noise to privacy budget
        self.global_rounds = self._get_initial_num_rounds(self.num_oneways)

        # Init controls
        self._init_control_params(client_answers, candidates)

        # Initialise privacy budget
        self.gauss_sigma, self.exp_epsilon = self._calibrate_privacy(self.global_rounds)
        self.init_gauss_sigma, self.init_exp_epsilon = (
            self.gauss_sigma,
            self.exp_epsilon,
        )

        # Initialise model - as in DistAIM
        global_model, current_measurements, engine = self._get_model(
            original_dataset,
            candidates,
            client_answers,
            q=self.clients_per_round / len(client_answers),
        )

        # Begin FL training
        terminate, t = False, -1
        self.rounds = self.global_rounds  # Used when T is specified
        if self.rounds == 0:
            terminate = True
        while not terminate:
            t += 1

            # Use up whatever remaining budget there is for one last round
            exp_count = self.local_rounds
            gauss_count = (
                (1 + self.control_estimates) * self.local_rounds
                if t < self.control_rounds and "private" in self.control_type
                else self.local_rounds
            )
            if self._final_round(
                self.gauss_sigma,
                self.exp_epsilon,
                t + 1,
                gauss_count=gauss_count,
                exp_count=exp_count,
            ):
                self.gauss_sigma, self.exp_epsilon = self._calibrate_remaining_budget(
                    self.gauss_sigma, self.exp_epsilon, rounds=self.local_rounds
                )
                terminate = True

            # If logging control variate metrics, check errors for all clients
            if self.log_internal_metrics and (t < self.control_rounds or t == 0):
                for client_id, answers in client_answers:
                    client_controls = self._compute_client_control_variates(
                        client_id, answers, candidates
                    )
                    self._debug_variates(
                        client_id,
                        client_controls,
                        list(zip(*client_answers))[
                            1
                        ],  # todo: check this, likely broken
                        answers,
                        candidates,
                        global_model,
                        t=t,
                    )

            # Server filters workload based on model size
            size_limit = self._get_size_limit(t=t)
            viable_candidates = self._filter_candidates(
                candidates, global_model, size_limit
            )

            # Get subset of participating clients
            client_subset = self._subset_client_answers(
                client_answers, q=self.clients_per_round / len(horizontal_datasets)
            )

            subset_sample_size = sum(
                [sum(ans[self.oneways[0]]) for _, ans in client_subset]
            )

            # Optional debug - always include first client
            if "client_0" in self.log_internal_metrics:
                client_subset[0] = client_answers[0]

            client_control_variates = {
                client_id: self._compute_client_control_variates(
                    client_id, client_answer, viable_candidates
                )
                for client_id, client_answer in client_subset
            }

            # Optional - filter client subset via heterogeneity measure
            client_subset, client_control_variates = self._filter_client_answers(
                client_subset, client_control_variates
            )

            round_ids, _ = zip(*client_subset)
            logger.debug(
                f"Round {t+1} - Selected clients {round_ids}, subset sample size={subset_sample_size}, size_limit={size_limit}"
            )

            # Server sends init model to clients
            if self.track_communication:
                for k in round_ids:
                    self.server_communication_log[f"{k}_{t}_send_model"] = (
                        global_model.size * 8
                    )
                    self.client_communication_log[f"{k}_{t}_receive_model"] = (
                        global_model.size * 8
                    )

            # Precompute model estimates to speed up first step
            initial_model_estimates = {
                cl: global_model.project(cl).datavector() for cl in viable_candidates
            }

            cl = None
            if self.selection_method == "server_random":
                cl = np.random.choice(
                    np.array(list(viable_candidates.keys()), dtype="object")
                )
            if self.selection_method == "majority_vote":
                # each client performs local exp query
                votes = [
                    self.client_aims[i]._exp_query(
                        [client_answers],
                        global_model,
                        viable_candidates,
                        self.exp_epsilon,
                        self.gauss_sigma,
                    )
                    for i, client_answers in client_subset
                ]
                # Take majority of vote
                cl = [x[0] for x in Counter(votes).most_common()[: self.local_rounds]]
                logger.debug(f"Majority vote - {cl}")

            if not isinstance(cl, list):
                cl = [cl]
            client_updates = []
            for query in cl:
                client_updates.extend(
                    [
                        self._client_update(
                            engine,
                            global_model,
                            current_measurements,
                            client_data[1],
                            client_control_variates[client_data[0]],
                            horizontal_datasets[client_data[0]],
                            viable_candidates,
                            client_id=client_data[0],
                            t=t,
                            initial_model_estimates=initial_model_estimates,
                            cl=query,
                        )
                        for client_data in client_subset  # client_data[i] = [client_id, client_answers]
                    ]
                )

            self.gauss_mech.accumulate(self.gauss_sigma, self.local_rounds)
            if self.exp_epsilon > 0 and self.exp_epsilon != float("inf"):
                self.exp_mech.accumulate(self.exp_epsilon, self.local_rounds)

            # Log communication
            self._log_communication_round(client_updates, global_model, t)

            # Optional - pruning updates (unsupported, not used)
            client_updates = self._prune_updates(client_updates)

            # Noise updates (for distributed noise)
            client_updates = self._noise_client_updates(client_updates, t=t)

            # Server updates control variatess if t < control_rounds
            current_measurements, stopping_list = self._update_server_control_variate(
                current_measurements,
                client_subset,
                subset_sample_size,
                t,
                final_round=terminate,
            )
            self._update_control_candidates(t, stopping_list)

            # Global model update - aggregate local updates and learn new PGM
            (
                global_model,
                current_measurements,
                prev_ests,
                new_ests,
                sigmas,
            ) = self._update_global_model(
                engine,
                global_model,
                current_measurements,
                client_updates,
                final_round=terminate,
            )

            # Calculate query error and error thresh to decide whether to anneal budget
            if self.adaptive_noise:
                anneal_count = 0
                for cl in sigmas:
                    new_est = new_ests[cl]
                    prev_est = prev_ests[cl]
                    sigma = sigmas[cl]
                    anneal_count += int(
                        self._annealing_condition(
                            new_est, prev_est, self.domain.size(cl)
                        )
                    )
                    # logger.debug(
                    #     f"Threshold cond for {cl}, sigma={sigma} - {query_error} <= {error_thresh}"
                    # )

                logger.debug(f"Anneal count {anneal_count} <= {len(prev_ests)/2}")
                anneal_condition = (
                    anneal_count >= len(prev_ests) / 2
                    if self.anneal_type == "majority"
                    else anneal_count >= 1
                )
                if anneal_condition:
                    self.gauss_sigma, self.exp_epsilon = self._anneal_budget(
                        self.gauss_sigma, self.exp_epsilon
                    )
                    logger.debug(
                        f"Budget annealed - {self.gauss_sigma}, {self.exp_epsilon}"
                    )

            logger.debug(
                f"Num of measurements at end of round {t+1} - {len(current_measurements)}"
            )

        # Measurement stats
        measurement_set = set()
        for m in current_measurements:
            logger.debug(f"{m[3], m[2]}")
            measurement_set.add(m[3])
        m_dict = defaultdict(int)
        for m in measurement_set:
            m_dict[len(m)] += 1
        logger.debug(
            f"Number of unique measurements: {len(measurement_set)}, Count - {m_dict}"
        )

        self.internal_metrics["actual_rounds"] = t + 1

        if self.config.epsilon > 0:
            logger.info(f"Accountant eps {self.accountant.get_eps()}")

        self.client_aims = None
        self.model = global_model
        if self.backend_model == "rap":
            synth = engine.synthetic_data(rows=self.synth_rows)
            synth = [None, synth.df]  # match aim_results format
        else:
            synth = global_model.synthetic_data()
        return synth

    def generate(
        self,
        dataset: TabularDataset,
        workload=None,
        num_clients: int = 2,
        non_iid: bool = False,
        **kwargs,
    ) -> TabularDataset:
        self.n = dataset.n
        self.target = dataset.y
        self.num_clients = num_clients
        for aim in self.client_aims:
            aim.target = self.target

        hfl_loader = HorizontalSharder(dataset, non_iid=non_iid)
        horizontal_datasets = hfl_loader.get_partition(num_clients)
        self.internal_metrics["non_iid_param"] = hfl_loader.non_iid_param
        if not self.skip_hetero_metrics:
            hetero_measures = hfl_loader.measure_heterogeneity(
                horizontal_datasets,
                central_dataset=dataset,
                client_answers=self._compute_client_answers(
                    [data.to_pgm_dataset() for data in horizontal_datasets],
                    hfl_loader._generate_workload(dataset),
                    dataset,
                    save_cache=self.save_client_answer_cache,
                    load_cache=self.load_client_answer_cache,
                ),
            )
            self.internal_metrics["heterogeneity_mean"] = hetero_measures[0]
            self.internal_metrics["heterogeneity_sd"] = hetero_measures[1]
            self.internal_metrics["heterogeneity_min"] = hetero_measures[2]
            self.internal_metrics["heterogeneity_max"] = hetero_measures[3]
            logger.info(f"Dataset={dataset.name}, hetero type = {non_iid}")
            logger.info(f"Dataset heterogeneity measure = {hetero_measures}")
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
                f"Average hetero of decisions - {self.internal_metrics['mean_tau_q'], self.internal_metrics['std_tau_q']}"
            )
        if self.log_decisions:
            self.local_decision_map = {
                i: x.decision_map for i, x in enumerate(self.client_aims)
            }

        # Cache communication metrics
        self._parse_communication_log()

        return TabularDataset(
            f"{self.name} {dataset.name}",
            None,
            result[1],
            dataset.domain,
        )
