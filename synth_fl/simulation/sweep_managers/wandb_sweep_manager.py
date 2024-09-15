import os
from datetime import datetime

import wandb
from synth_fl.simulation.sweep_managers.sweep_manager import SweepManager
from synth_fl.utils import WANDB_PROJECT_NAME, logger

# TODO: This silences all wandb output apart from sweep IDs/urls
os.environ["WANDB_SILENT"] = "true"  # fmt: off


class WandbSweepManager(SweepManager):
    def __init__(
        self,
        sweep_name,
        n_workers=1,
        count=None,
        sweep_backend="local",
        sweep_id: str = "",
        debug: bool = False,
    ) -> None:
        super().__init__(
            sweep_name=sweep_name,
            n_workers=n_workers,
            count=count,
            sweep_backend=sweep_backend,
            debug=debug,
        )
        self.sweep_backend = sweep_backend
        self.sweep_id = sweep_id
        self.type = "wandb"
        if not self.sweep_id:
            self._generate_sweep_id()
        else:
            self.config = self._load_sweep_config()
            logger.debug(f"Restoring wandb sweep manager with sweep_id={self.sweep_id}")
        logger.debug(f"Initialised wandb sweep backend with sweep_id={self.sweep_id}")

    def _generate_sweep_id(self):
        wandb.login()
        self.config = self._load_sweep_config()
        timestamp = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
        self.config["name"] = f"{timestamp}_{self.config['name']}"
        self.sweep_id = wandb.sweep(sweep=self.config, project=WANDB_PROJECT_NAME)
        wandb.finish()

    def run(self, func) -> None:
        if self.sweep_backend == "slurm":
            os.environ.pop(
                "WANDB_SERVICE", None
            )  # If using slurm, remove wandb env variable added by sweep process to force new ports in submitit process

        logger.success(f"Launching wandb sweep agent - sweep_id={self.sweep_id}")
        wandb.agent(
            self.sweep_id, project=WANDB_PROJECT_NAME, function=func, count=self.count
        )
        wandb.finish()
