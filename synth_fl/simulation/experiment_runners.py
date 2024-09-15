import os
from abc import abstractmethod
from collections import defaultdict

import submitit
from pathos.multiprocessing import ProcessingPool

import wandb
from synth_fl.simulation.experiment_tasks import (
    FLStatsTask,
)
from synth_fl.simulation.sweep_managers import LocalSweepManager, WandbSweepManager
from synth_fl.utils import logger

from pathlib import Path

CONFIG_TO_TASK_MAP = defaultdict(lambda: FLStatsTask) | {}

SLURM_PATH = os.path.dirname(__file__).split("/synth_fl")[0] + "/slurm/"
LOG_PATH = SLURM_PATH + "/job_results"


class ExperimentRunner:
    def __init__(
        self,
        sweep_name,
        sweep_manager_type="wandb",
        n_workers=1,
        nodes=1,
        catch_task_errors=True,
        debug=False,
        multi_thread=False,
    ) -> None:
        self.sweep_name = sweep_name
        self.sweep_manager_type = sweep_manager_type
        self.sweep_manager = None
        self.sweep_backend = "local"
        self.n_workers = n_workers
        self.nodes = nodes
        self.config = None
        self.task = None
        self.catch_task_errors = catch_task_errors
        self.debug = debug
        self.multi_thread = multi_thread
        logger.debug("Experiment initialised :)")

    def _init_sweep_manager(self, sweep_id=""):
        if self.sweep_manager_type == "wandb":
            self.sweep_manager = WandbSweepManager(
                self.sweep_name,
                n_workers=self.n_workers,
                sweep_backend=self.sweep_backend,
                sweep_id=sweep_id,
                debug=self.debug,
            )
        else:
            self.sweep_manager = LocalSweepManager(
                self.sweep_name,
                n_workers=self.n_workers,
                sweep_backend=self.sweep_backend,
                sweep_id=sweep_id,
                debug=self.debug,
                multi_thread=self.multi_thread,
            )

        logger.success(
            f"Sweep initialised ({self.sweep_manager.config['name']}) - {self.sweep_manager.sweep_size} tasks"
        )

    def _init_experiment_task(self):
        self.task = CONFIG_TO_TASK_MAP[
            self.sweep_manager.config.get("task", "flstats")
        ](
            self.sweep_manager.sweep_id,
            sweep_size=self.sweep_manager.sweep_size,
            sweep_manager_type=self.sweep_manager.type,
            sweep_backend=self.sweep_backend,
            sweep_name=self.sweep_name,
            catch_task_errors=self.catch_task_errors,
        )

    @abstractmethod
    def begin(self):
        pass


class LocalRunner(ExperimentRunner):
    def __init__(
        self,
        sweep_name,
        sweep_manager_type="wandb",
        n_workers=1,
        nodes=1,
        catch_task_errors=True,
        debug=False,
    ) -> None:
        super().__init__(
            sweep_name=sweep_name,
            sweep_manager_type=sweep_manager_type,
            n_workers=n_workers,
            nodes=nodes,
            catch_task_errors=catch_task_errors,
            debug=debug,
        )

    def begin(self):
        self._init_sweep_manager()
        self._init_experiment_task()
        logger.success(
            f"Sweep initialised ({self.sweep_manager.config['name']}) - {self.sweep_manager.sweep_size} tasks"
        )

        if self.sweep_manager == "wandb":
            pool = ProcessingPool(self.n_workers)
            pool_func = lambda x: self.sweep_manager.run(func=self.task.run)

            # Create n_workers, if sweep_size < n_workers create worker for each task
            pool.map(
                pool_func,
                range(0, min(self.sweep_manager.sweep_size, self.n_workers)),
            )
            pool.close()
            pool.join()
        else:
            self.sweep_manager.run(
                func=self.task.run
            )  # if backend="local", defer multiprocessing to LocalSweepManager()


class SlurmRunner(ExperimentRunner):
    def __init__(
        self,
        sweep_name,
        sweep_manager_type="wandb",
        n_workers: int = 1,
        task_mem: int = 3,
        tasks_per_node: int = 1,
        cpus_per_task: int = 1,
        debug=False,
        sweep_id: str = "",
        sweep_rank: int = -1,
        nodes=1,
        multi_sweep=False,
        exclude="",
        catch_task_errors=True,
        multi_thread=False,
    ) -> None:
        super().__init__(
            sweep_name=sweep_name,
            sweep_manager_type=sweep_manager_type,
            n_workers=n_workers,
            nodes=nodes,
            catch_task_errors=catch_task_errors,
            debug=debug,
        )
        self.sweep_backend = "slurm"
        self.sweep_id = sweep_id
        self.sweep_rank = sweep_rank
        self.task_mem = task_mem
        self.tasks_per_node = tasks_per_node
        self.cpus_per_task = cpus_per_task
        self.multi_sweep = multi_sweep
        self.sweep_name_list = []
        self.exclude = exclude
        self.multi_thread = multi_thread

    def _setup_sweep_result_path(self):
        # Create directory for sweep results
        SWEEP_RESULTS_PATH = f"{SLURM_PATH}sweep_results/{self.sweep_manager.sweep_id}/"
        logger.info(f"Creating sweep results path - {SWEEP_RESULTS_PATH}")
        Path(SWEEP_RESULTS_PATH).mkdir(parents=True, exist_ok=True)

    def _setup_executor(self):
        # submitit to launch slurm
        log_folder = f"{LOG_PATH}/sweep-{self.sweep_manager.sweep_id}_%j"
        executor = (
            submitit.DebugExecutor(folder=log_folder)
            if self.debug
            else submitit.AutoExecutor(folder=log_folder)
        )
        return executor

    def _launch_wandb(self, partition, timeout_min):
        self._init_sweep_manager(self.sweep_id)
        self._init_experiment_task()
        if self.sweep_id and self.sweep_manager_type == "wandb":
            api = wandb.Api()
            completed_tasks = len(api.sweep(f"synthfl/{self.sweep_id}").runs)
            logger.success(
                f"Restored sweep ({self.sweep_manager.config['name']}) - {completed_tasks} / {self.sweep_manager.sweep_size} tasks completed"
            )
        self._setup_sweep_result_path()
        executor = self._setup_executor()
        executor.update_parameters(
            slurm_partition=partition,
            tasks_per_node=self.tasks_per_node,
            cpus_per_task=self.cpus_per_task,
            timeout_min=timeout_min,
            mem_gb=self.task_mem,
            name=f"wdb_agent_{self.sweep_manager.sweep_id}",
        )
        for _ in range(0, self.n_workers):
            job = executor.submit(self.sweep_manager.run, func=self.task.run)
            logger.success(f"Slurm job submitted with id={job.job_id}")
            if self.debug:
                logger.debug(f"Job Result - {job.result()}")

    def _launch_local(self, partition, timeout_min):
        for sweep_name in self.sweep_name_list:
            self.sweep_name = sweep_name
            self._init_sweep_manager(self.sweep_id)
            self._init_experiment_task()
            self._setup_sweep_result_path()

            executor = self._setup_executor()
            host_job_id = None
            nodes = self.nodes

            # If restoring a multinode sweep, only launch a single node job with that rank
            if self.nodes > 1 and self.sweep_rank > 0:
                nodes = 1

            for i in range(0, nodes):
                additional_slurm_args = {}
                if self.exclude:
                    additional_slurm_args["exclude"] = self.exclude
                executor.update_parameters(
                    tasks_per_node=1,
                    cpus_per_task=40,
                    timeout_min=timeout_min,
                    mem_gb=60,
                    name=f"{self.sweep_manager.sweep_id}_{sweep_name}_r{i}",
                    slurm_partition=partition,
                    slurm_exclusive=True,
                    slurm_additional_parameters=additional_slurm_args,
                )
                job = executor.submit(
                    self.sweep_manager.run,
                    func=self.task.run,
                    sweep_rank=i if self.sweep_rank < 0 else self.sweep_rank,
                    total_ranks=self.nodes,
                    host_job_id=host_job_id,
                )
                logger.success(f"Slurm job submitted with id={job.job_id}")
                if host_job_id is None:
                    host_job_id = job.job_id
                if self.debug:
                    logger.debug(f"Job Result - {job.result()}")

    def begin(self):
        self.sweep_name_list = [self.sweep_name]
        if self.multi_sweep:
            self.sweep_name_list = [
                x.replace(" ", "") for x in self.sweep_name.split(",")
            ]

        partition = "cpu-debug" if self.debug else "cpu-batch"
        timeout_min = 10 if self.debug else 2880

        if self.sweep_manager_type == "wandb":
            self._launch_wandb(partition, timeout_min)
        else:
            self._launch_local(partition, timeout_min)
