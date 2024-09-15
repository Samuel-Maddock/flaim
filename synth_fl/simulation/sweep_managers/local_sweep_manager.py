import itertools
import os
import math
import glob
from datetime import datetime
from collections import defaultdict

import pandas as pd
import uuid
from pathos.multiprocessing import ProcessingPool
from tqdm import tqdm

from synth_fl.simulation.sweep_managers.sweep_manager import SweepManager
from synth_fl.utils import DEFAULT_DATA_ROOT, logger

SLURM_PATH = os.path.dirname(__file__).split("/synth_fl")[0] + "/slurm/"


class LocalSweepManager(SweepManager):
    def __init__(
        self,
        sweep_name,
        n_workers=1,
        count=None,
        sweep_backend="local",
        sweep_id="",
        debug=False,
        multi_thread=False,
    ) -> None:
        super().__init__(
            sweep_name=sweep_name,
            n_workers=n_workers,
            count=count,
            sweep_backend=sweep_backend,
            debug=debug,
        )
        self.sweep_id = sweep_id if sweep_id else f"local_{uuid.uuid1().hex}"
        self.config = self._load_sweep_config()  # load wandb-style config
        self.sweep_args = (
            self._get_sweep_args()
        )  # process config to a set of args for each task
        self.n_workers = n_workers
        self.type = "local"
        self.restore_sweep = False
        logger.debug(f"Local sweep id={self.sweep_id}")
        if sweep_id:
            logger.debug(f"Restoring local sweep...")
            self.restore_sweep = True
        self.sweep_results_path = f"{SLURM_PATH}sweep_results/{self.sweep_id}/"
        self.multi_thread = multi_thread

    def _get_sweep_args(self):
        sweep_args = {
            k: self.config["parameters"][k]["values"]
            for k in self.config["parameters"].keys()
        }
        return [
            dict(zip(sweep_args, x)) for x in itertools.product(*sweep_args.values())
        ]

    def _get_sweep_checkpoint(self):
        results = glob.glob(f"{self.sweep_results_path}*.csv")
        sweep_ids = []
        for res in results:
            sweep_ids.append(int(res.split(os.sep)[-1].split("_")[0]))
        return set(sweep_ids)

    def _filter_sweep_args(self, filter_func, sweep_rank, total_ranks, host_job_id):
        logger.warning(f"Filtering sweep args - initial size = {len(self.sweep_args)}")
        filtered_args = list(
            filter(lambda x: filter_func(x, filter_only=True), self.sweep_args)
        )
        gen_counter = defaultdict(int)
        for i, arg in enumerate(filtered_args):
            arg["arg_id"] = i + 1
            arg["sweep_rank"] = sweep_rank
            arg["total_ranks"] = total_ranks
            arg["host_job_id"] = host_job_id
            gen_counter[arg["generator"]] += 1
        logger.warning(f"Filtered sweep args - filtered size = {len(filtered_args)}")
        logger.info(f"Generator counter {gen_counter}")

        if total_ranks > 1:
            logger.warning(f"Multi-node sweep - rank {sweep_rank} of {total_ranks}")
            num_args = math.floor(1 / total_ranks * len(filtered_args))
            filtered_args = (
                filtered_args[num_args * (sweep_rank) : num_args * (sweep_rank + 1)]
                if sweep_rank != total_ranks - 1
                else filtered_args[num_args * (sweep_rank) :]
            )
            logger.warning(
                f"Partitioned sweep args - partitioned size = {len(filtered_args)}"
            )

        if self.restore_sweep:
            checkpoint_ids = self._get_sweep_checkpoint()
            logger.warning(
                f"Sweep restored - {len(checkpoint_ids)} / {len(filtered_args)}"
            )
            filtered_args = list(
                filter(lambda x: x["arg_id"] not in checkpoint_ids, filtered_args)
            )
        return filtered_args

    def run(self, func, sweep_rank=1, total_ranks=1, host_job_id=None):
        filtered_args = self._filter_sweep_args(
            func, sweep_rank, total_ranks, host_job_id
        )

        if len(filtered_args) > 0:
            # Debugging only - allows pdb.set_trace()
            if self.debug or self.multi_thread:
                res = [func(args) for args in filtered_args]
                sweep_df = pd.concat(res)
            else:
                pool = ProcessingPool(self.n_workers)
                logger.success(f"Created pool with {self.n_workers} workers...")
                res = tqdm(pool.imap(func, filtered_args), total=len(filtered_args))
                sweep_df = pd.concat(res)
                if self.sweep_backend == "local":
                    if not os.path.exists(self.sweep_results_path):
                        os.makedirs(self.sweep_results_path)
                    sweep_df.to_csv(f"{self.sweep_results_path}/results.csv")
                pool.close()
                pool.join()
        else:
            logger.warning(
                f"Sweep has no args after filtering - Either the restored sweep is complete or no args are viable"
            )
        logger.success("Sweep complete!")
