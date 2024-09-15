from synth_fl.generators import Generator, FederatedDLConfig

from synth_fl.libs.ctgan import CTGAN as _CTGAN
from synth_fl.libs.snsynth.pytorch.nn.dpctgan import DPCTGAN
from synth_fl.libs.snsynth.transform import TableTransformer, OneHotEncoder

from sweeper.utils import logger
from sweeper.utils.dataloaders import TabularDataset


class CTGAN(Generator):
    def __init__(self, config: FederatedDLConfig) -> None:
        super().__init__(config)
        self.epsilon = config.epsilon
        self.delta = config.delta
        self.epochs = config.epochs
        self.name = "ctgan"

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
        self.transformer = None

    def _init_ctgan(self):
        # Original ctgan - uses GMMs for cts columns + log frequency conditional vectors
        if self.epsilon <= 0:
            ctgan = _CTGAN(
                verbose=True,
                epochs=self.epochs,
                embedding_dim=self.embedding_dim,
                generator_lr=self.generator_lr,
                discriminator_lr=self.discriminator_lr,
                discriminator_steps=self.discriminator_steps,
                generator_decay=self.generator_decay,
                discriminator_decay=self.discriminator_decay,
                batch_size=self.batch_size,
                wandb_enabled=self.wandb_enabled,
            )
        else:
            # DP-CTGAN is privacy-friendly, samples conditional vectors equiv to poisson sampling
            # GMMs disabled for cts features?
            ctgan = DPCTGAN(
                epochs=self.epochs,
                epsilon=self.epsilon,
                delta=self.delta,
                verbose=True,
                disabled_dp=self.epsilon == 0,
                embedding_dim=self.embedding_dim,
                generator_dim=(256, 256),
                discriminator_dim=(256, 256),
                generator_lr=self.generator_lr,
                generator_decay=self.generator_decay,
                discriminator_lr=self.discriminator_lr,
                discriminator_decay=self.discriminator_decay,
                discriminator_steps=self.discriminator_steps,
                batch_size=self.batch_size,
                pac=self.pac,
                sigma=self.sigma,
                max_per_sample_grad_norm=self.grad_norm,
                loss="cross_entropy",
                wandb_enabled=self.wandb_enabled,
            )
        return ctgan

    def _set_transformer(self, dataset):
        if self.epsilon > 0:
            self.transformer = TableTransformer(
                transformers=[OneHotEncoder() for i in dataset.df.columns]
            )
            self.transformer.fit(dataset.df.to_numpy())

    def _generate(self, data, discrete_columns):
        ctgan = self._init_ctgan()
        ctgan.fit(
            data, categorical_columns=discrete_columns, transformer=self.transformer
        )
        self.transformer = ctgan._transformer
        self.trainer = ctgan
        self.trainer.generator = self.trainer._generator
        return ctgan.generate(self.n)

    def generate(
        self, dataset: TabularDataset, workload=None, **kwargs
    ) -> TabularDataset:
        self.n = dataset.n
        discrete_columns = list(dataset.df.columns)
        synth_data = self._generate(dataset.df, discrete_columns)
        return TabularDataset(
            f"{self.name} {dataset.name}", None, synth_data, dataset.domain
        )
