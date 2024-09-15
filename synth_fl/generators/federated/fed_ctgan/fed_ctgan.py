from typing import Any, Dict, Generator, List
from synth_fl.generators import FederatedGenerator, FederatedDLConfig

from sweeper.utils import logger
from sweeper.utils.dataloaders import TabularDataset, HorizontalSharder

from hydra.utils import instantiate
from synth_fl.libs.snsynth.transform import TableTransformer, OneHotEncoder

from synth_fl.libs.flsim.interfaces.metrics_reporter import Channel
from synth_fl.libs.flsim.utils.example_utils import (
    LEAFDataLoader,
)

from synth_fl.libs.snsynth.pytorch.nn.dpctgan import (
    Generator,
    Discriminator,
    DataSampler,
)

from opacus.accountants.utils import get_noise_multiplier


import math
import pandas as pd
import numpy as np
import torch
from torch import optim
import gc

# Must import for hydra to work
import synth_fl.libs.flsim.configs
from synth_fl.generators.federated.fed_ctgan.ctgan_sync_trainer import CTGANSyncTrainer
from synth_fl.libs.flsim.utils.config_utils import fl_config_from_json
from omegaconf import OmegaConf

from .flsim_helpers import (
    TorchTabularDataset,
    CTGANDataProvider,
    FLCTGANModel,
    CTGANMetricsReporter,
    flsim_config,
)


class FedCTGAN(FederatedGenerator):
    def __init__(self, config: FederatedDLConfig) -> None:
        super().__init__(config)
        self.epsilon = config.epsilon
        self.delta = config.delta
        self.epochs = config.epochs
        self.local_rounds = config.local_rounds  # Client epochs
        self.name = "fed_ctgan"

        self.clients_per_round = config.clients_per_round
        if self.clients_per_round <= 1:
            self.clients_per_round = math.ceil(
                config.num_clients * self.clients_per_round
            )
        else:
            self.clients_per_round = config.num_clients

        self.embedding_dim = config.embedding_dim
        self.generator_lr = config.generator_lr
        self.discriminator_lr = config.discriminator_lr
        self.generator_decay = config.generator_decay
        self.discriminator_decay = config.discriminator_decay

        self.discriminator_steps = config.discriminator_steps
        self.batch_size = config.batch_size

        self.pac = config.pac
        self.grad_norm = config.grad_norm
        self.sigma = config.sigma

        self.wandb_enabled = config.wandb_enabled

        self.server_lr = config.server_lr
        # Set one to the other if an lr is -1
        self.discriminator_lr = (
            self.generator_lr if self.discriminator_lr == -1 else self.discriminator_lr
        )
        self.generator_lr = (
            self.discriminator_lr if self.generator_lr == -1 else self.generator_lr
        )
        self.discriminator_decay = (
            self.generator_decay
            if self.discriminator_decay == -1
            else self.discriminator_decay
        )
        self.generator_decay = (
            self.discriminator_decay
            if self.generator_decay == -1
            else self.generator_decay
        )

        self.privacy_type = config.privacy_type
        self.use_cuda_if_available = False

    def _update_flsim_config(self, cfg):
        cfg["trainer"]["server"]["server_optimizer"]["lr"] = self.server_lr
        cfg["trainer"]["client"]["optimizer"]["lr"] = self.discriminator_lr
        cfg["trainer"]["epochs"] = self.epochs
        cfg["trainer"]["client"]["epochs"] = self.local_rounds
        cfg["trainer"]["users_per_round"] = self.clients_per_round

        # FLSim Privacy Parameters - User level DP
        if self.epsilon > 0:
            rounds_in_epoch = math.ceil(self.num_clients / self.clients_per_round)
            if self.privacy_type == "user_level":
                cfg["trainer"]["server"]["_base_"] = "base_sync_dp_server"
                priv_type = "server"
                sample_rate = self.clients_per_round / self.num_clients
                total_steps = self.epochs * rounds_in_epoch
            else:
                cfg["trainer"]["client"]["_base_"] = "base_dp_client"
                priv_type = "client"
                # todo: This is not accurate for sample-skew scenarios..
                sample_rate = self.batch_size / (self.n / self.num_clients)
                # todo: Expected number of steps, again not accurate...
                # todo: added an extra epoch to be more conservative, noise multipliers should be initialised separtely for each client
                total_steps = math.ceil(
                    (self.epochs + 1)
                    * rounds_in_epoch
                    * (self.clients_per_round / self.num_clients)  # sample rate
                    * self.discriminator_steps
                )

            cfg["trainer"][priv_type]["privacy_setting"] = {}
            cfg["trainer"][priv_type]["privacy_setting"]["clipping"] = {}
            cfg["trainer"][priv_type]["privacy_setting"]["target_delta"] = self.delta
            cfg["trainer"][priv_type]["privacy_setting"]["clipping"][
                "clipping_value"
            ] = self.grad_norm
            noise_multiplier = get_noise_multiplier(
                steps=total_steps,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                sample_rate=sample_rate,
            )
            cfg["trainer"][priv_type]["privacy_setting"][
                "noise_multiplier"
            ] = noise_multiplier

        return cfg

    def _generate(
        self,
        horizontal_datasets,
        transformer,
        data_sampler,
    ):
        # FLSim DataProvider construction
        # 1. Create PyTorch Dataset objects
        train_dataset = TorchTabularDataset(horizontal_datasets, transformer)

        # 2. Create DataLoader
        # todo: current hack for train datasets ....
        data_loader = LEAFDataLoader(
            train_dataset, train_dataset, train_dataset, batch_size=32
        )

        # 3. Create DataProvider
        data_provider = CTGANDataProvider(
            data_loader=data_loader,
            transformer=transformer,
            batch_size=self.batch_size,
            embedding_dim=self.embedding_dim,
            discriminator_steps=self.discriminator_steps,
        )

        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.set_default_dtype(torch.float64)

        # 4. Construct FLModel (Discriminator)
        # TODO: Param args for discrim dim + pac
        cuda_enabled = torch.cuda.is_available() and self.use_cuda_if_available
        device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")

        data_dim = transformer.output_width
        model = Discriminator(
            data_dim + data_sampler.dim_cond_vec(),
            discriminator_dim=(256, 256),
            loss="cross_entropy",
            pac=1,
        )
        model.double()
        global_model = FLCTGANModel(model, transformer=transformer, device=device)

        # 5. Construct generator model to pass to Trainer
        # TODO: Param args for generator dim
        generator = Generator(
            self.embedding_dim + data_sampler.dim_cond_vec(),
            generator_dim=(256, 256),
            data_dim=data_dim,
        )
        generator.double()
        generator_optim = optim.Adam(
            generator.parameters(),
            lr=self.generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self.generator_decay,
        )

        logger.info(f"{model}")
        logger.info(f"{generator}")
        logger.info(f"{generator_optim}")

        # 6. Construct config
        cfg = self._update_flsim_config(flsim_config)
        cfg = fl_config_from_json(cfg)
        logger.info(OmegaConf.to_yaml(cfg))

        # 7. Instantiate trainer and launch training
        if cuda_enabled:
            global_model.fl_cuda()

        # CTGANSyncTrainer
        trainer = instantiate(
            cfg.trainer,
            model=global_model,
            cuda_enabled=cuda_enabled,
            generator=generator,
            generator_optim=generator_optim,
            data_transformer=transformer,
            data_sampler=data_sampler,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            wandb_enabled=self.wandb_enabled,
        )
        print(f"Created {cfg.trainer._target_}")
        metrics_reporter = CTGANMetricsReporter([Channel.STDOUT])
        final_model, eval_score = trainer.train(
            data_provider=data_provider,
            metrics_reporter=metrics_reporter,
            num_total_users=data_provider.num_train_users(),
            distributed_world_size=1,
        )

        # 8. Generate data from model

        synth_data = trainer.generate(n=self.n)
        self.trainer = trainer

        # Free memory
        del generator
        del model
        del train_dataset
        gc.collect()

        return synth_data

    def _set_transformer(self, dataset):
        self.transformer = TableTransformer(
            transformers=[OneHotEncoder() for i in dataset.df.columns]
        )
        self.transformer.fit(dataset.df.to_numpy())

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

        # Train transformer for CTGAN
        transformed_data = self.transformer.transform(dataset.df.to_numpy())
        transformed_data = np.array(
            [
                [float(x) if x is not None else 0.0 for x in row]
                for row in transformed_data
            ]
        )
        data_sampler = DataSampler(
            transformed_data, transformers=self.transformer.transformers
        )

        # Get non-IID split of dataset
        hfl_loader = HorizontalSharder(dataset, non_iid=non_iid)
        horizontal_datasets = hfl_loader.get_partition(num_clients)
        self.internal_metrics["non_iid_param"] = hfl_loader.non_iid_param

        # Generate synthetic data via FLSim training
        result = self._generate(
            horizontal_datasets, transformer=self.transformer, data_sampler=data_sampler
        )

        # Cache communication metrics
        self._parse_communication_log()

        return TabularDataset(
            f"{self.name} {dataset.name}",
            None,
            result,
            dataset.domain,
        )
