import warnings

import numpy as np
import pandas as pd
import gc
import torch

from sklearn.exceptions import UndefinedMetricWarning
from scipy.stats import gaussian_kde

from synth_fl.generators import FederatedDLConfig
from synth_fl.generators.federated import FLGeneratorFactory
from synth_fl.generators.central import FACTORY_MAP as CENTRAL_FACTORY_MAP

from synth_fl.experiment_tasks.fl_stats import FLStatsTask

from sweeper.utils import logger, AttrDict
from sweeper.utils.dataloaders import TabularDataset
from sweeper.utils.workloads import WorkloadManager

from dacite import from_dict

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


CTGAN_ARGS = set(
    [
        "epsilon",
        "epochs",
        "embedding_dim",
        "generator_lr",
        "discriminator_lr",
        "generator_decay",
        "discriminator_decay",
        "discriminator_steps",
        "batch_size",
        "pac",
        "grad_norm",
        "server_lr",
    ]
)


class DLTrainingTask(FLStatsTask):
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
        self.task_name = "dl_training"
        self.default_args = {
            "p": 0.9,
            "num_clients": 1,
            "clients_per_round": 1,  # 100%
            "epsilon": 1,
            "generator": "ctgan",
            "match_data_shape": False,
            "non_iid": "",
            "dataset": "adult",
            "data_seed": 571,
            "workload_seed": None,
            "train_ml": False,
            "global_rounds": 0,
            "epochs": 50,
            "prune_repeated_job": True,
            "workload_type": "uniform",
            "workload_k": 3,
            "workload_num_marginals": 64,
            "skip_metrics": False,
            "server_lr": 1,
        }

        self.ml_options = {
            "central_dp_model",
            "central_model",
            "hfl_local_model",
            "hfl_local_dp_ensemble",
        }

        self.ignore_workload_metrics = True

    def _is_repeated(self, gen_name):
        return False

    def _check_fl_conds(self):
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

        if gen_name == "ctgan" and self.task_config.server_lr != 1:
            return False

        return True

    # Overrides parent class
    def _update_name(self):
        name_dict = {}
        for k in self.task_config.keys():
            if k in CTGAN_ARGS:
                name_dict[k] = self.task_config[k]
        self.name_str = str(name_dict)

    # To generate directly from trainer post-training
    def compute_nll(self, generator, trainer, test_data):
        n = test_data.df.shape[0] * 10
        transformed_test_data = generator.transformer.transform(test_data.df.to_numpy())
        trainer.generator.eval()

        steps = n // self.task_config.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.task_config.batch_size, trainer.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std)

            condvec = trainer._data_sampler.sample_condvec(self.task_config.batch_size)

            if condvec is None:
                pass
            else:
                c1, m1, col, opt = condvec
                c1 = torch.Tensor(c1)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = trainer.generator(fakez)
            fakeact = trainer._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        synth_data = trainer._transformer.inverse_transform(data)

        actual_nll = 0
        col_list = []
        for col in synth_data.columns:
            # TODO: laplace smoothing?
            if len(synth_data[col].unique()) == 1:
                logger.warning(f"{col} is a point mass, dropping from synth data...")
                actual_nll = float("inf")
                col_list.append(col)
        synth_data = synth_data.drop(col_list, axis=1)
        t_data = test_data.df.drop(col_list, axis=1)
        kde = gaussian_kde(synth_data.T)
        filtered_nll = -1 * np.mean(kde.logpdf(t_data.T))
        actual_nll = float(filtered_nll) if actual_nll == 0 else actual_nll
        return actual_nll, float(filtered_nll)

    def run(self, args=None, filter_only=False):
        if filter_only:
            return self._filter_args(args)

        self._init_task(args)
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

        gen_config = from_dict(data_class=FederatedDLConfig, data=self.task_config)
        gen_config.wandb_enabled = self.sweep_manager_type == "wandb"

        # TODO: Could combine generator_name into GeneratorConfig
        generator = fl_gen_factory.create_obj(
            generator_name=gen_name, generator_cfg=gen_config
        )

        # NB: If gen_name is an ML model then it defaults to 'baseline' and has .type == central
        # So repeats of ML models are also caught
        if (
            self.task_config.prune_repeated_job
            and generator.type == "central"
            and self._is_repeated(self.task_config.generator)
            or self._check_fl_conds()
        ):
            logger.warning(
                f"Detected repetition for generator {self.task_config.generator} - ending early"
            )
            self._get_sweep_progress()
            return pd.DataFrame()

        # Load dataset + test split for ML
        central_dataset = TabularDataset(name=self.task_config.dataset)
        logger.info(f"Training transformer for CTGAN...")
        generator._set_transformer(central_dataset) 
        logger.info(f"CTGAN transformer trained")

        central_dataset, test_dataset = central_dataset.to_subset(
            self.task_config.p, random_state=self.task_config.data_seed
        )

        # Pass workload args and generate workload
        workload_kwargs = self._parse_workload_args()
        workload_manager = WorkloadManager(**workload_kwargs)
        central_workload = workload_manager.generate_workload(central_dataset)

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

                # Free memory for subsequent executions
                del generator
                del workload_manager
                del test_dataset
                del central_dataset
                gc.collect()
                return pd.DataFrame()
        else:
            synth_data = generator.generate(
                central_dataset,
                central_workload,
                non_iid=self.task_config.non_iid,
                num_clients=self.task_config.num_clients,
            )

        # Compute workloads statistics on synth data
        metrics = self._compute_statistics(
            gen_name,
            central_workload,
            synth_data,
            central_dataset,
            0,
            0,
            0,
            0,
            workload_manager,
        )

        actual_log_likelihood, filtered_log_likelihood = self.compute_nll(
            generator, generator.trainer, test_dataset
        )

        if generator.name != "real_data" and self.task_config.epsilon > 0:
            try:
                actual_log_likelihood, filtered_log_likelihood = self.compute_nll(
                    generator, generator.trainer, test_dataset
                )
            except Exception:
                logger.debug(f"Computing NLL failed... setting log_likelihood=None")
                filtered_log_likelihood = None
            metrics["test_log_likelihood"] = filtered_log_likelihood
            metrics["actual_log_likelihood"] = actual_log_likelihood
            logger.info(f"Test log likelihood: {filtered_log_likelihood}")
            logger.info(f"(actual) Test log likelihood: {actual_log_likelihood}")

        # Optional - train GBDT model on synth data
        if self.task_config.train_ml and generator.name not in ["baseline", "zero_gen"]:
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
        other_metrics = {
            "l1_err": final_metrics["l1_avg_err"],
            "test_auc": final_metrics["test_auc"],
        }
        # Log df row (to wandb or local file)
        self._log_to_sweep_manager(exp_df, other_metrics=other_metrics)
        self._get_sweep_progress()  # Output sweep progress %

        # Free memory for subsequent executions
        del generator
        del workload_manager
        del test_dataset
        del central_dataset
        gc.collect()

        return exp_df
