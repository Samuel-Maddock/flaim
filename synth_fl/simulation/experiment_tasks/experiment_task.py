import os
from abc import ABC, abstractmethod
import wandb
import time

from datetime import datetime
from synth_fl.utils import WANDB_PROJECT_NAME, AttrDict, logger

SLURM_PATH = os.path.dirname(__file__).split("/synth_fl")[0] + "/slurm/"


class ExperimentTask:
    def __init__(
        self,
        sweep_id,
        sweep_size=1,
        sweep_manager_type="local",
        sweep_backend="local",
        sweep_name="",
        catch_task_errors=True,
    ) -> None:
        self.sweep_id = sweep_id
        self.sweep_name = sweep_name
        self.name_str = ""
        self.task_config = None  # Recieved from wandb or sweep manager
        self.task_name = ""
        self.sweep_size = sweep_size
        self.default_args = {}  # Should be specified by child class
        self.sweep_manager_type = sweep_manager_type
        self.sweep_backend = sweep_backend
        # For logging
        self.start_time = None
        self.end_time = None

        # If true, experiment task should safely catch and parse errors during execution
        # Set false for debugging
        # Prevent multiprocessing pools from crashing if individual tasks break
        self.catch_task_errors = catch_task_errors

    def _add_config_defaults(self):
        for k, v in self.default_args.items():
            if k not in self.task_config:
                self.task_config.update({k: v})

    def _update_name(self):
        self.name_str = f"{str(self.task_config)}"
        if self.task_name:
            self.name_str = f"{self.task_name}_{str(self.task_config)}"

    # Should suffice for most tasks
    def _init_task(self, args):
        self.start_time = datetime.now()
        if args:  # if local sweep
            self.task_config = AttrDict(args)
            self._update_name()
        else:  # if wandb sweep, retrieve args
            wandb.init(project=WANDB_PROJECT_NAME)
            self.task_config = wandb.config

        self.task_config["seed"] = int(self.task_config["iter"].split("_")[1])
        self._add_config_defaults()
        self._update_name()
        if not args:
            wandb.run.name = self.name_str

        job_id = ""
        if "SLURM_JOB_ID" in os.environ:
            job_id = os.environ.get("SLURM_JOB_ID")
            self.task_config.update({"slurm_id": job_id})
            job_id = f"_{job_id}"

        if self.sweep_manager_type == "local" and "debug" not in self.sweep_name:
            logger.remove(logger._current_handler_id)
            logname = f"{SLURM_PATH}job_results/sweep-{self.sweep_id}{job_id}/r{self.task_config.sweep_rank}_process_log_{os.getpid()}"
            logger._current_handler_id = logger.add(logname, level="INFO", enqueue=True)
        logger.info("\n")

        logger.success(f"Beginning run {self.name_str}")

    def _parse_workload_args(self):
        workload_args = {}
        for k, v in self.task_config.items():
            arg_name = k.split("workload_")
            if len(arg_name) > 1:
                workload_args[arg_name[-1]] = v
        logger.info(f"Workload args parsed - {workload_args}")
        return workload_args

    # Should be overridden by task if custom logging is needed
    def _log_to_sweep_manager(self, df, other_metrics={}, name="data"):
        self.end_time = datetime.now()
        df["job_start_time"] = self.start_time
        df["job_end_time"] = self.end_time

        # Log to wandb as Table artifact
        if self.sweep_manager_type == "wandb":
            table = wandb.Table(dataframe=df)
            log_dict = {"table": table} | other_metrics
            wandb.log(log_dict)
        # Log locally
        if self.sweep_backend == "slurm":
            prefix = (
                ""
                if self.sweep_manager_type == "wandb"
                else f"{str(self.task_config.arg_id)}_r{self.task_config.sweep_rank}_"
            )
            df.to_csv(
                f"{SLURM_PATH}sweep_results/{self.sweep_id}/{prefix}task_result_{time.time()}.csv"
            )
        logger.success(f"Task logs saved to sweep manager...")

    def _access_sweep_api(self):
        api = wandb.Api()
        return api.sweep(f"synthfl/{self.sweep_id}")

    def _get_sweep_progress(self):
        if "local" not in self.sweep_id:
            api = self._access_sweep_api()
            completed_tasks = len(api.runs)
            logger.success(
                f"Progress of sweep - {completed_tasks} / {self.sweep_size} - {round((completed_tasks/self.sweep_size)*100,2)}%"
            )

    @abstractmethod
    def run(self, args=None):
        pass
