import json
import os
from abc import ABC, abstractmethod

from synth_fl.utils import generate_seed

SWEEP_PATH = os.path.dirname(__file__).split("/synth_fl")[0] + "/sweep_configs"

from typing import Dict


class SweepManager(ABC):
    def __init__(
        self, sweep_name, n_workers=1, count=None, sweep_backend="local", debug=False
    ) -> None:
        self.sweep_name = sweep_name
        self.count = count
        self.n_workers = n_workers
        self.sweep_id = None
        self.config = None
        self.sweep_backend = sweep_backend
        self.type = None  # should be overridden
        self.debug = debug

    @property
    def sweep_size(self):
        sweep_size = 1
        for key in self.config["parameters"]:
            sweep_size *= len(self.config["parameters"][key]["values"])
        return sweep_size

    def _load_sweep_config(self) -> Dict:
        sweep_config = None
        with open(os.path.join(SWEEP_PATH, f"{self.sweep_name}.json")) as f:
            sweep_config = json.load(f)

        # Handle repeating sweeps multiple times e.g. to average results
        if "iters" in sweep_config:
            sweep_config["parameters"].update(
                {"iter": {"values": list(range(1, sweep_config["iters"] + 1))}}
            )

        # If seeds are specified, need a seed for each individual run
        # Concatenate seed to iters and let tasks parse the args
        if "seeds" in sweep_config:
            assert sweep_config["iters"] == len(sweep_config["seeds"])
            sweep_config["parameters"].update(
                {
                    "iter": {
                        "values": f"{i}_{sweep_config['seeds'][i]}"
                        for i in sweep_config["parameters"]["iters"]
                    }
                }
            )
        else:
            sweep_config["parameters"].update(
                {
                    "iter": {
                        "values": [
                            f"{sweep_config['parameters']['iter']['values'][i]}_{generate_seed(2**16)}"
                            for i in range(
                                0, len(sweep_config["parameters"]["iter"]["values"])
                            )
                        ]
                    }
                }
            )
        return sweep_config

    @abstractmethod
    def run(self, func):
        pass
