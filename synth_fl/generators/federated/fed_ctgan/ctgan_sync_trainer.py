from synth_fl.libs.flsim.common.timeline import Timeline
from synth_fl.libs.flsim.interfaces.model import IFLModel
from synth_fl.libs.flsim.interfaces.metrics_reporter import IFLMetricsReporter
from synth_fl.libs.flsim.trainers.sync_trainer import SyncTrainer, SyncTrainerConfig
from synth_fl.libs.flsim.utils.config_utils import (
    fullclassname,
    init_self_cfg,
    is_target,
)

from synth_fl.libs.snsynth.pytorch.nn.dpctgan import CTGANSynthesizer

from dataclasses import dataclass

from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
from torch import optim, nn
from torch.nn import functional

from hydra.core.config_store import ConfigStore

import numpy as np

# Must import before using as config

import wandb
from sweeper.utils import logger


class CTGANSyncTrainer(SyncTrainer):
    def __init__(
        self,
        *,
        model: IFLModel,
        generator,
        generator_optim,
        data_transformer,
        data_sampler,
        embedding_dim,
        batch_size,
        cuda_enabled: bool = False,
        wandb_enabled: bool = False,
        **kwargs,
    ):
        super().__init__(model=model, cuda_enabled=cuda_enabled, **kwargs)
        self.generator = generator
        self.optimizerG = generator_optim
        self.loss = "cross_entropy"
        self.pac = 1
        self.embedding_dim = embedding_dim
        self.generator_losses = []

        self._transformer = data_transformer
        self._data_sampler = data_sampler

        self.batch_size = batch_size
        self.device = "cuda" if cuda_enabled else "cpu"
        self.wandb_enabled = wandb_enabled

    # # Override to disable torch.no_grad() because CTGAN requires autograd for loss calculations
    # def _calc_eval_metrics_on_clients(
    #     self,
    #     model: IFLModel,
    #     clients_data,
    #     data_split: str,
    #     metrics_reporter,
    # ) -> None:
    #     """Calculate eval metrics on `clients` and record in `metrics_reporter` using
    #     `model`. `model` is expected to be in eval mode.
    #     """
    #     for client_data in clients_data:
    #         for batch in getattr(client_data, f"{data_split}_data")():
    #             batch_metrics = model.get_eval_metrics(batch)
    #             metrics_reporter.add_batch_metrics(batch_metrics)

    # Pre-round hook - propagate generator to clients models
    def on_before_client_updates(self, **kwargs):
        self.global_model().generator = self.generator
        self.global_model().model.zero_grad()  # Clear generator forward passes
        return super().on_before_client_updates(**kwargs)

    # Post-round hook - Perform generator step before any other post training hooks
    def _post_train_one_round(
        self, timeline: Timeline, metrics_reporter: IFLMetricsReporter
    ):
        self._generator_step()
        # Log to wandb every epoch
        if self.wandb_enabled and timeline.global_round_num() % 10 == 0:
            wandb.log(
                {
                    "loss_g": self.generator_losses[-1],
                    "loss_d": metrics_reporter.cached_discrim_loss,
                }
            )
        return super()._post_train_one_round(timeline)

    # From CTGAN - Used for training the Generator (server-side)
    # TODO: Could move these to the actual Server object
    def _generator_step(self):
        self.generator.zero_grad()
        self.global_model().model.zero_grad()

        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        fakez = torch.normal(mean=mean, std=std)
        condvec = self._data_sampler.sample_condvec(self.batch_size)

        if condvec is None:
            c1, m1, col, opt = None, None, None, None
        else:
            c1, m1, col, opt = condvec
            c1 = torch.Tensor(c1).to(self.device)
            m1 = torch.Tensor(m1).to(self.device)
            fakez = torch.cat([fakez, c1], dim=1)

        fake = self.generator(fakez)
        fakeact = self._apply_activate(fake)

        if c1 is not None:
            y_fake = self.global_model().model(torch.cat([fakeact, c1], dim=1))
        else:
            y_fake = self.global_model().model(fakeact)

        real_label = 1.0
        # if condvec is None:
        cross_entropy = 0
        # else:
        #    cross_entropy = self._cond_loss(fake, c1, m1)
        criterion = nn.BCELoss()
        if self.loss == "cross_entropy":
            label_g = torch.full(
                (int(self.batch_size / self.pac),),
                real_label,
                device=self.device,
            )
            # label_g = torch.full(int(self.batch_size/self.pack,),1,device=self.device)
            loss_g = criterion(y_fake.squeeze(), label_g)
            loss_g = loss_g + cross_entropy
        else:
            loss_g = -torch.mean(y_fake) + cross_entropy

        self.optimizerG.zero_grad(set_to_none=True)
        loss_g.backward()
        self.optimizerG.step()
        self.global_model().model.zero_grad()

        generator_loss = loss_g.detach().cpu().item()
        self.generator_losses.append(generator_loss)
        logger.info(f"Server generator step - loss g = {generator_loss}")

    # Helpers from DP-CTGAN for generator forward pass
    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for t in self._transformer.transformers:
            if not t.is_categorical:
                # not discrete column
                st += t.output_width
            else:
                ed = st + t.output_width
                ed_c = st_c + t.output_width
                tmp = functional.cross_entropy(
                    data[:, st:ed],
                    torch.argmax(c[:, st_c:ed_c], dim=1),
                    reduction="none",
                )
                loss.append(tmp)
                st = ed
                st_c = ed_c

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for transformer in self._transformer.transformers:
            if transformer.is_continuous:
                ed = st + transformer.output_width
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif transformer.is_categorical:
                ed = st + transformer.output_width
                transformed = CTGANSynthesizer._gumbel_softmax(data[:, st:ed], tau=0.2)
                data_t.append(transformed)
                st = ed

        return torch.cat(data_t, dim=1)

    # To generate directly from trainer post-training
    def generate(self, n, condition_column=None, condition_value=None):
        """
        TODO: Add condition_column support from CTGAN
        """
        self.generator.eval()

        # output_info = self._transformer.output_info
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            condvec = self._data_sampler.sample_condvec(self.batch_size)

            if condvec is None:
                pass
            else:
                c1, m1, col, opt = condvec
                c1 = torch.Tensor(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())
        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)


@dataclass
class CTGANSyncTrainerConfig(SyncTrainerConfig):
    _target_: str = fullclassname(CTGANSyncTrainer)
    client_metrics_reported_per_epoch: int = 1


ConfigStore.instance().store(
    name="ctgan_sync_trainer",
    node=CTGANSyncTrainerConfig,
    group="trainer",
)
