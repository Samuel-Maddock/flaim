import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from synth_fl.generators import FederatedGeneratorConfig
from synth_fl.generators.federated import FLGeneratorFactory
from synth_fl.generators.central import FACTORY_MAP as CENTRAL_FACTORY_MAP

from synth_fl.simulation.experiment_tasks import ExperimentTask
from synth_fl.utils import logger, train_and_benchmark_gbdt, AttrDict
from synth_fl.utils.dataloaders import TabularDataset, HorizontalSharder
from synth_fl.workloads import WorkloadManager

from dacite import from_dict

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class FLStatsTask(ExperimentTask):
    def __init__(
        self,
        sweep_id,
        sweep_size=1,
        sweep_manager_type="local",
        sweep_backend="local",
        sweep_name="",
        catch_task_errors=True,
    ) -> None:
        super().__init__(
            sweep_id,
            sweep_size=sweep_size,
            sweep_manager_type=sweep_manager_type,
            sweep_backend=sweep_backend,
            sweep_name=sweep_name,
            catch_task_errors=catch_task_errors,
        )
        self.task_name = "flstats"
        self.default_args = {
            "p": 0.9,
            "num_clients": 5,
            "clients_per_round": 1,  # 100%
            "epsilon": 1,
            "aim_mech": "gaussian",
            "generator": "aim",
            "match_data_shape": False,
            "non_iid": "",
            "dataset": "adult",
            "data_seed": 571,
            "workload_seed": None,
            "train_ml": False,
            "global_rounds": 0,
            "prune_repeated_job": True,
            "update_prune_method": "",
            "client_prune_method": "",
            "aggregation_method": "merge_all",
            "workload_type": "uniform",
            "workload_k": 3,
            "workload_num_marginals": 64,
            "skip_metrics": False,
            "control_type": "",
            "quality_score_type": "",
            "control_rounds": -1,
            "control_estimates": -1,
            "pgm_weight_method": "sigma",
            "anneal_type": "one",
            "num_bins": 32,
            "selection_method": "exp",
            "gauss_budget_alloc": 0.9,
            "accounting_method": "distributed",
            "init_method": "",
        }

        self.ml_options = {
            "central_dp_model",
            "central_model",
            "hfl_local_model",
            "hfl_local_dp_ensemble",
        }

        self.ignore_workload_metrics = True

    def _parse_and_run_ml_train(self, central_dataset, test_dataset, synth_data=None):
        if central_dataset.task_type == "classification":
            non_priv_model = GradientBoostingClassifier(n_estimators=100, max_depth=4)
        else:
            non_priv_model = GradientBoostingRegressor(n_estimators=100, max_depth=4)
        generator = self.task_config.generator
        model = non_priv_model  # default
        hfl_loader = HorizontalSharder(
            central_dataset, non_iid=self.task_config.non_iid
        )
        # GBDT training
        if generator == "central_model":
            # Non-DP GBDT central training
            synth_data = [central_dataset]
        elif generator == "hfl_local_model":
            # Local non-DP training, no FL
            synth_data = hfl_loader.get_partition(self.task_config.num_clients)
        model_data = [(generator, [dataset.df for dataset in synth_data], model)]
        return train_and_benchmark_gbdt(model_data, central_dataset, test_dataset)

    def _compute_test_workload_stats(
        self, metrics, generator, test_workload, synth_data, real_data
    ):
        metrics = {}
        metrics["test_max_err"] = 0
        metrics["test_l1_err"] = 0
        metrics["test_tv_err"] = 0
        metrics["test_l1_avg_err"] = 0
        metrics["test_l2_err"] = 0
        metrics["test_tv_avg_err"] = 0
        metrics["test_l1_avg_err"] = 0
        metrics["test_l2_avg_err"] = 0

        logger.info("\n")
        k_set = set()
        for w in test_workload:
            k_set.add(len(w))

        for i in k_set:
            metrics[f"test_l1_k={i}_err"] = 0
            metrics[f"test_k={i}_count"] = 0

        for marginal in test_workload:
            metrics[f"test_k={len(marginal)}_count"] += 1

        if not self.task_config.skip_metrics and len(test_workload) > 0:
            pgm_true_data = real_data.to_pgm_dataset()
            if generator not in ["baseline", "zero_gen"]:
                pgm_synth_data = synth_data.to_pgm_dataset()

            for marginal in test_workload:
                true_ans = pgm_true_data.project(marginal).datavector()
                if generator == "baseline" or generator == "zero_gen":
                    synth_ans = np.zeros(true_ans.shape)
                    synth_total = 1
                else:
                    synth_ans = pgm_synth_data.project(marginal).datavector()
                    synth_total = synth_ans.sum()

                metrics["test_l1_err"] += np.linalg.norm(
                    true_ans / true_ans.sum() - synth_ans / synth_total, ord=1
                )
                metrics[f"test_l1_k={len(marginal)}_err"] += np.linalg.norm(
                    true_ans / true_ans.sum() - synth_ans / synth_total, ord=1
                )

                metrics["test_l2_err"] += np.linalg.norm(
                    true_ans / true_ans.sum() - synth_ans / synth_total, ord=2
                )
                metrics["test_tv_err"] += 0.5 * np.linalg.norm(
                    true_ans / true_ans.sum() - synth_ans / synth_total, ord=1
                )
                metrics["test_max_err"] = max(
                    np.linalg.norm(true_ans - synth_ans, ord=1), metrics["test_max_err"]
                )

            metrics["test_l1_avg_err"] = metrics["test_l1_err"] / len(test_workload)
            for i in k_set:
                if metrics[f"test_k={i}_count"] != 0:
                    metrics[f"test_l1_k={i}_avg_err"] = (
                        metrics[f"test_l1_k={i}_err"] / metrics[f"test_k={i}_count"]
                    )
            metrics["test_tv_avg_err"] = metrics["test_tv_err"] / len(test_workload)
            metrics["test_l2_avg_err"] = metrics["test_l2_err"] / len(test_workload)

        for k, v in metrics.items():
            if "test" in k:
                logger.info(f"{self.task_config.generator} {k}={v}")

    def _compute_statistics(
        self,
        generator,
        workload,
        synth_data,
        real_data,
        average_client_communication_sent=0,
        average_client_communication_received=0,
        total_server_communication_sent=0,
        total_server_communication_received=0,
        workload_manager: WorkloadManager = None,
    ):
        metrics = {}
        metrics["max_err"] = 0
        metrics["l1_err"] = 0
        metrics["tv_err"] = 0
        metrics["l1_avg_err"] = 0
        metrics["l2_err"] = 0

        complete_workload = workload_manager._downward_closure(workload)

        for i in range(self.task_config.workload_k):
            metrics[f"l1_k={i+1}_err"] = 0
            metrics[f"k={i+1}_count"] = 0

        for marginal in complete_workload:
            metrics[f"k={len(marginal)}_count"] += 1

        if not self.task_config.skip_metrics:
            pgm_true_data = real_data.to_pgm_dataset()
            if generator not in ["baseline", "zero_gen"]:
                pgm_synth_data = synth_data.to_pgm_dataset()

            for marginal in complete_workload:
                true_ans = pgm_true_data.project(marginal).datavector()
                if generator == "baseline" or generator == "zero_gen":
                    synth_ans = np.zeros(true_ans.shape)
                    synth_total = 1
                else:
                    synth_ans = pgm_synth_data.project(marginal).datavector()
                    synth_total = synth_ans.sum()

                metrics["l1_err"] += np.linalg.norm(
                    true_ans / true_ans.sum() - synth_ans / synth_total, ord=1
                )
                metrics[f"l1_k={len(marginal)}_err"] += np.linalg.norm(
                    true_ans / true_ans.sum() - synth_ans / synth_total, ord=1
                )

                metrics["l2_err"] += np.linalg.norm(
                    true_ans / true_ans.sum() - synth_ans / synth_total, ord=2
                )
                metrics["tv_err"] += 0.5 * np.linalg.norm(
                    true_ans / true_ans.sum() - synth_ans / synth_total, ord=1
                )
                metrics["max_err"] = max(
                    np.linalg.norm(true_ans - synth_ans, ord=1), metrics["max_err"]
                )

            metrics["l1_avg_err"] = metrics["l1_err"] / len(complete_workload)
            for i in range(self.task_config.workload_k):
                if metrics[f"k={i+1}_count"] != 0:
                    metrics[f"l1_k={i+1}_avg_err"] = (
                        metrics[f"l1_k={i+1}_err"] / metrics[f"k={i+1}_count"]
                    )
            metrics["tv_avg_err"] = metrics["tv_err"] / len(complete_workload)
            metrics["l2_avg_err"] = metrics["l2_err"] / len(complete_workload)

        metrics["average_client_communication_sent"] = average_client_communication_sent
        metrics[
            "average_client_communication_received"
        ] = average_client_communication_received
        metrics["total_server_communication_sent"] = total_server_communication_sent
        metrics[
            "total_server_communication_received"
        ] = total_server_communication_received

        for k, v in metrics.items():
            if "communication" in k:
                logger.info(f"{self.task_config.generator} {k}={v/(1000**2)}mb")
            else:
                logger.info(f"{self.task_config.generator} {k}={v}")

        return metrics

    def _is_repeated(self, gen_name):
        return (
            self.task_config.clients_per_round != 0.1
            or self.task_config.aggregation_method != "merge_all"
            or self.task_config.control_type != ""
            or self.task_config.local_rounds != 1
            or self.task_config.accounting_method != "distributed"
            or self.task_config.pgm_weight_method != "sigma"
            or self.task_config.anneal_type != "one"
            or self.task_config.gauss_budget_alloc == 1
            or self.task_config.init_method != ""
        )

    def _check_fl_conds(self):
        if not self.task_config.prune_repeated_job:
            return False

        if (
            self.task_config.selection_method == "majority_vote"
            and self.task_config.accounting_method == "ldp"
        ):
            return True

        if self.task_config.selection_method == "none" and (
            self.task_config.global_rounds != 10
            or self.task_config.control_type != ""
            or self.task_config.aggregation_method != "merge_all"
            or self.task_config.gauss_budget_alloc
            != 0.9  # TODO: Needs to be changed for future sweeps
        ):
            return True

        if self.task_config.selection_method == "server_random" and (
            "combine" not in self.task_config.control_type
            and self.task_config.control_type != ""
        ):
            return True

        if "distributedaim" in self.task_config.generator and (
            self.task_config.client_prune_method != ""
            or self.task_config.update_prune_method != ""
            or self.task_config.aggregation_method != "merge_all"
            or self.task_config.local_rounds != 1
            or self.task_config.control_type != ""  # todo: change often
            or self.task_config.client_prune_method != ""
            or self.task_config.quality_score_type != ""
            or self.task_config.accounting_method != "distributed"
            or "covar" in self.task_config.anneal_type
        ):
            return True

        if (
            self.task_config.generator != "flaim"
            and self.task_config.workload_type == "marginal_target"
        ):
            return True

        if (
            self.task_config.workload_type == "marginal_target"
            and self.task_config.workload_num_marginals != -1
        ):
            return True

        if (
            self.task_config.generator == "flaim"
            and not self.task_config.use_control_variates
            and self.task_config.control_type != "add"
        ):
            return True

        if (
            self.task_config.generator == "flaim"
            and not self.task_config.use_control_variates
            and "private" not in self.task_config.control_type
            and "covar" in self.task_config.anneal_type
        ):
            return True

        if (
            self.task_config.generator == "flaim"
            and self.task_config.use_control_variates
            and "private" not in self.task_config.control_type
            and (
                self.task_config.control_estimates != -1
                or self.task_config.control_rounds != -1
            )
        ):
            return True

        if (
            self.task_config.generator == "flaim"
            and self.task_config.use_control_variates
            and self.task_config.control_type == "private_sub"
            and self.task_config.control_rounds == 0
            and self.task_config.control_estimates != -1
        ):
            return True

        return False

    def _filter_args(self, args):
        self.task_config = AttrDict(args)
        self._add_config_defaults()
        gen_name = self.task_config.generator
        if (
            self.task_config.prune_repeated_job
            and (gen_name.lower() in CENTRAL_FACTORY_MAP or gen_name == "NaiveHFL_AIM")
            and self._is_repeated(self.task_config.generator)
            or self._check_fl_conds()
        ):
            return False

        return True

    # From https://arxiv.org/pdf/2306.04803.pdf
    def compute_negative_log_likelihood(self, model, data):
        logZ = model.belief_propagation(model.potentials, logZ=True)
        log_probas = np.zeros(data.records)
        for cl in model.cliques:
            P = model.potentials[cl].values
            idxs = data.project(cl).df.values.astype(int)
            log_probas += np.array([P[tuple(i)] for i in idxs])
        return logZ - log_probas

    def run(self, args=None, filter_only=False):
        if filter_only:
            return self._filter_args(args)

        self._init_task(args)
        # TODO: This is a hack for MPCAim, need to separate out num_servers vs num_clients
        self.task_config.num_servers = self.task_config.num_clients

        metrics, ml_metrics = {}, {}
        fl_gen_factory = FLGeneratorFactory()
        gen_name = self.task_config.generator

        if (
            gen_name not in fl_gen_factory.get_factory_map()
            and gen_name not in fl_gen_factory.get_central_factory_map()
            and "_" not in gen_name  # for naivehfl_GEN
        ):
            logger.warning(
                f"Given generator '{gen_name}' not found in factory maps - defaulting to 'baseline' for statistics"
            )
            if self.task_config.train_ml and gen_name not in self.ml_options:
                raise RuntimeError(
                    f"Given generator name {gen_name} is not a valid generator OR ML option"
                )

            gen_name = "baseline"

        gen_config = from_dict(
            data_class=FederatedGeneratorConfig, data=self.task_config
        )

        # TODO: Could combine generator_name into GeneratorConfig
        generator = fl_gen_factory.create_obj(
            generator_name=gen_name, generator_cfg=gen_config
        )

        # TODO: Reduce repeated sweep runs with wandb - https://github.com/wandb/wandb/issues/1487
        # NB: If gen_name is an ML model then it defaults to 'baseline' and has .type == central
        # So repeats of ML models are also caught
        if (
            self.task_config.prune_repeated_job
            and (generator.type == "central" or generator.name == "NaiveHFL_AIM")
            and self._is_repeated(self.task_config.generator)
            or self._check_fl_conds()
        ):
            logger.warning(
                f"Detected repetition for generator {self.task_config.generator} - ending early"
            )
            self._get_sweep_progress()
            return pd.DataFrame()

        # Central DP aim - no clients
        if gen_name == "aim":
            self.task_config.clients_per_round = 1

        # Load dataset + test split for ML
        central_dataset = TabularDataset(
            name=self.task_config.dataset, num_bins=self.task_config.num_bins
        )
        central_dataset, test_dataset = central_dataset.to_subset(
            self.task_config.p, random_state=self.task_config.data_seed
        )
        if "synthetic" in central_dataset.name and generator.type == "central":
            central_dataset.df.drop("client_id", axis=1, inplace=True)

        # Pass workload args and generate workload
        workload_kwargs = self._parse_workload_args()
        workload_manager = WorkloadManager(**workload_kwargs)
        central_workload = workload_manager.generate_workload(central_dataset)

        test_workload = workload_manager.generate_test_workload(
            central_dataset, central_workload
        )

        # Generate synthetic data
        if self.catch_task_errors:
            try:
                synth_data = generator.generate(
                    central_dataset,
                    central_workload,
                    non_iid=self.task_config.non_iid,
                    num_clients=self.task_config.num_clients,
                )
            except Exception as e:
                logger.error(f"Task run has crashed with error: {str(e)}")
                logger.error(f"Safely continuing sweep...")
                return pd.DataFrame()
        else:
            synth_data = generator.generate(
                central_dataset,
                central_workload,
                non_iid=self.task_config.non_iid,
                num_clients=self.task_config.num_clients,
            )

        # Compute metrics on synth data
        metrics = self._compute_statistics(
            gen_name,
            central_workload,
            synth_data,
            central_dataset,
            generator.average_client_communication_sent,
            generator.average_client_communication_received,
            generator.total_server_communication_sent,
            generator.total_server_communication_received,
            workload_manager,
        )
        self._compute_test_workload_stats(
            metrics, generator, test_workload, synth_data, central_dataset
        )
        try:
            log_likelihood = self.compute_negative_log_likelihood(
                generator.model, test_dataset.to_pgm_dataset()
            )
        except IndexError:
            print(f"Index error  with config - {dict(self.task_config)}")
            raise Exception("END SWEEP")
        metrics["test_log_likelihood"] = log_likelihood.mean()
        logger.info(f"Test log likelihood: {log_likelihood.mean()}")

        # Optional - train GBDT model on synth data + negative log-likelihood
        if (
            self.task_config.train_ml
            and generator.name not in ["baseline", "zero_gen"]
            and "synthetic" not in self.task_config.dataset
        ):
            ml_metrics = self._parse_and_run_ml_train(
                central_dataset, test_dataset, synth_data=[synth_data]
            )
            ml_metrics = ml_metrics[self.task_config.generator]

        # Format metrics into columns for df
        final_metrics = (
            metrics
            | ml_metrics
            | {"train_n": central_dataset.n}
            | dict(self.task_config)
        )
        logger.debug(f"Exporting metrics - {final_metrics}")
        final_metrics = final_metrics | generator.internal_metrics
        rows = [final_metrics]
        exp_df = pd.DataFrame(rows)
        self._log_to_sweep_manager(exp_df)  # Log df row (to wandb or local file)
        # if gen_config.log_decisions and "aim" in gen_name:
        #     self._log_to_sweep_manager(generator.decision_map, name="decision_map")

        self._get_sweep_progress()  # Output sweep progress %
        return exp_df
