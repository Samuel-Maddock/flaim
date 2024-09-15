import torch
from torch import nn
from torch.utils.data import Dataset
from synth_fl.libs.flsim.interfaces.metrics_reporter import (
    Channel,
    Metric,
    TrainingStage,
)
from sweeper.utils.dataloaders import TabularDataset

from typing import Dict, Generator, List, Optional, Iterator, Any, Tuple

from synth_fl.libs.flsim.utils.example_utils import (
    FLModel,
    FLBatchMetrics,
    UserData,
    DataProvider,
    IFLUserData,
    FLMetricsReporter,
)
from synth_fl.libs.snsynth.pytorch.nn.dpctgan import DataSampler, CTGANSynthesizer

import torch.nn as nn
import torch
from tqdm import tqdm

import numpy as np
import math


class CTGANMetricsReporter(FLMetricsReporter):
    def __init__(self, channels: List[Channel], log_dir: str = None):
        super().__init__(channels, log_dir)
        # For wandb in post-train round hook
        self.cached_discrim_loss = 0

    def compare_metrics(self, eval_metrics, best_metrics) -> bool:
        """One should provide concrete implementation of how to compare
        eval_metrics and best_metrics.
        Return True if eval_metrics is better than best_metrics
        """
        return False

    def compute_scores(self) -> Dict[str, Any]:
        """One should override this method to specify how to compute scores
        (e.g. accuracy) of the model based on metrics.
        Return dictionary where key is name of the scores and value is
        score.
        """
        return {}

    def create_eval_metrics(
        self, scores: Dict[str, Any], total_loss: float, **kwargs
    ) -> Any:
        """One should provide a concrete implementation of how to construct
        an object that represents evaluation metrics based on scores and
        total loss. Usually, one would just pick one of the scores or
        total loss as the evaluation metric to pick the better model, but
        this interface also allows one to make evaluation metrics more
        complex and use them in conjunction with the compare_metrics()
        function to determine which metrics and corresponding model are
        better.
        """
        return {}

    def report_metrics(
        self,
        reset: bool,
        stage: TrainingStage,
        extra_metrics: List[Metric] = None,
        **kwargs,
    ) -> Tuple[Any, bool]:
        self.cached_discrim_loss = np.sum(self.losses) / len(self.losses)
        return super().report_metrics(reset, stage, extra_metrics, **kwargs)


# For handling CTGAN sampling
class CTGANUserData(UserData):
    def __init__(
        self,
        user_data: Dict[str, Generator],
        transformer,
        batch_size,
        embedding_dim,
        discriminator_steps,
        eval_split: float = 0,
    ):
        self._train_batches = []
        self._num_train_batches = 0
        self._num_train_examples = 0

        self._eval_batches = []
        self._num_eval_batches = 0
        self._num_eval_examples = 0
        self._eval_split = eval_split

        self._batch_size = batch_size
        self._embedding_dim = embedding_dim

        user_features = list(user_data["features"])
        self._user_data = torch.vstack([torch.stack(batch) for batch in user_features])
        self._num_train_batches = discriminator_steps
        self._num_train_examples = len(self._user_data)

        self._batch_size = min(self._batch_size, self._num_train_examples)
        if self._num_train_batches * self._batch_size > self._num_train_examples:
            self._num_train_batches = max(
                1, math.ceil(self._num_train_examples / self._batch_size)
            )
        # print(self._num_train_batches, self._num_train_examples, self._batch_size)
        self._transformer = transformer
        self._data_sampler = DataSampler(
            self._user_data.numpy(), self._transformer.transformers
        )

    def train_data(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterator to return a user batch data for training
        """
        self._sample_train_batches()
        for batch in self._train_batches:
            yield batch

    def _sample_train_batches(self):
        self._train_batches = []
        for i in range(0, self._num_train_batches):
            # batch_size = min(
            #     self._num_train_examples - self._batch_size * i, self._batch_size
            # )
            batch_size = self._batch_size
            mean = torch.zeros(batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std)

            condvec = self._data_sampler.sample_condvec(batch_size)
            if condvec is None:
                c1, m1, col, opt = None, None, None, None
                real = self._data_sampler.sample_data(batch_size, col, opt)
            else:
                c1, m1, col, opt = condvec
                c1 = torch.Tensor(c1)
                m1 = torch.Tensor(m1)
                fakez = torch.cat([fakez, c1], dim=1)

                perm = np.arange(batch_size)
                np.random.shuffle(perm)
                real = self._data_sampler.sample_data(batch_size, col[perm], opt[perm])
                c2 = c1[perm]

            # Tensors get moved to device in CTGANFLModel training
            self._train_batches.append(
                {
                    "real": real,
                    "c1": c1,
                    "c2": c2,
                    "fakez": fakez,
                }
            )


class CTGANDataProvider(DataProvider):
    def __init__(
        self, data_loader, transformer, batch_size, embedding_dim, discriminator_steps
    ):
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.transformer = transformer
        self.discriminator_steps = discriminator_steps
        super().__init__(data_loader)

    def _create_fl_users(
        self, iterator: Iterator, eval_split: float = 0.0
    ) -> Dict[int, IFLUserData]:
        return {
            user_index: CTGANUserData(
                user_data,
                transformer=self.transformer,
                batch_size=self.batch_size,
                embedding_dim=self.embedding_dim,
                discriminator_steps=self.discriminator_steps,
                eval_split=eval_split,
            )
            for user_index, user_data in tqdm(
                enumerate(iterator), desc="Creating FL User", unit="user"
            )
        }


class FLCTGANModel(FLModel):
    def __init__(
        self,
        model: nn.Module,
        transformer,
        device: str = None,
    ):
        super().__init__(model, device)
        self.transformer = transformer

        # Is set in the pre-hook of trainer
        self.generator = None

        self.loss = "cross_entropy"
        self.pac = 1

    # Overriden to remove no_grad -> autograd reuqired for ctgan evalua
    def get_eval_metrics(self, batch) -> FLBatchMetrics:
        return self.fl_forward(batch)

    def fl_forward(self, batch) -> FLBatchMetrics:
        self.model.zero_grad()

        real = batch["real"]  # Discrim features
        batch_size = len(real)
        c1, c2 = batch["c1"], batch["c2"]
        fakez = batch["fakez"]
        real = torch.Tensor(real).to(self.device)
        # print(real.shape, fakez.shape, batch_size, c1.shape, c2.shape)
        if self.device is not None:
            c1 = c1.to(self.device)
            c2 = c2.to(self.device)
            fakez = fakez.to(self.device)

        fake = self.generator(fakez)
        fakeact = self._apply_activate(fake)

        if c1 is not None:
            fake_cat = torch.cat([fakeact, c1], dim=1)
            real_cat = torch.cat([real, c2], dim=1)
        else:
            real_cat = real
            fake_cat = fakeact

        y_fake = self.model(fake_cat)
        y_real = self.model(real_cat)
        criterion = nn.BCELoss()

        if self.loss == "cross_entropy":
            # Train with fake
            label_fake = torch.full(
                (int(batch_size / self.pac),),
                0.0,
                device=self.device,
            )
            error_d_fake = criterion(y_fake.squeeze(), label_fake)

            # Train with real
            label_true = torch.full(
                (int(batch_size / self.pac),),
                1.0,
                device=self.device,
            )
            error_d_real = criterion(y_real.squeeze(), label_true)
            loss_d = error_d_real + error_d_fake
        else:
            one = torch.tensor(1).to(self.device)
            mone = one * -1

            mean_fake = torch.mean(y_fake)
            mean_fake.backward(one)

            mean_real = torch.mean(y_real)
            mean_real.backward(mone)

            loss_d = -(mean_real - mean_fake)

        # TODO: Set PAC properly (currently defaults to 1)
        # pen = self.model.calc_gradient_penalty(real_cat, fake_cat, self.device, pac=1)
        # pen.backward(retain_graph=True)

        return FLBatchMetrics(
            loss=loss_d,
            num_examples=len(real),
            predictions=None,
            targets=None,
            model_inputs=[],
        )

    def get_num_examples(self, batch) -> int:
        return len(batch["real"])

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for transformer in self.transformer.transformers:
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


flsim_config = {
    "trainer": {
        # there are different types of aggregator
        # fed avg doesn't require lr, while others such as fed_avg_with_lr or fed_adam do
        "_base_": "ctgan_sync_trainer",
        "server": {
            "_base_": "base_sync_server",
            "server_optimizer": {
                "_base_": "base_fed_avg_with_lr",
                "lr": 1e-2,
                "momentum": 0,
            },
            # type of user selection sampling
            "active_user_selector": {
                "_base_": "base_uniformly_random_active_user_selector"
            },
        },
        "client": {
            # number of client's local epoch
            "epochs": 1,
            "optimizer": {
                "_base_": "base_optimizer_adam",
                # client's local learning rate
                "lr": 2e-4,
                "weight_decay": 0,
                # client's local momentum
                "momentum": 0,
            },
        },
        # number of users per round for aggregation
        "users_per_round": 20,
        # total number of global epochs
        # total #rounds = ceil(total_users / users_per_round) * epochs
        "epochs": 10,
        # frequency of reporting train metrics
        "train_metrics_reported_per_epoch": 100,
        # frequency of evaluation per epoch
        "eval_epoch_frequency": 1,
        "do_eval": False,
        # should we report train metrics after global aggregation
        "report_train_metrics_after_aggregation": True,
    }
}


class TorchTabularDataset(Dataset):
    def __init__(self, tabular_datasets: List[TabularDataset], transformer):
        self.y = tabular_datasets[0].y
        self.data = {}
        self.labels = {}
        self.num_classes = len(tabular_datasets[0].df[self.y].unique())

        # Populate self.data and self.targets
        for i, dataset in enumerate(tabular_datasets):
            # TODO: y column not dropped from self.data, fine for GANs, will cause leakage otherwise
            transformed_data = transformer.transform(dataset.df)
            self.data[i] = torch.Tensor(transformed_data)
            self.labels[i] = torch.Tensor(dataset.df[self.y].to_numpy())

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for user_id in self.data.keys():
            yield self.__getitem__(user_id)

    def __getitem__(self, user_id: str):
        if user_id not in self.data:
            raise IndexError(f"User {user_id} is not in dataset")
        return self.data[user_id], self.labels[user_id]
