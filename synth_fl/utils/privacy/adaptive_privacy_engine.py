# Implementation from https://github.com/m-lecuyer/adaptive_rdp

from opacus import PrivacyEngine
import opacus.accountants.analysis.rdp as tf_privacy
import numpy as np
from typing import List, Optional, Tuple, Union


class AdaptivePrivacyEngine(PrivacyEngine):
    def __init__(
        self,
        *args,
        target_delta=1e-5,
        n_accumulation_steps=1,
        sample_size=None,
        **kwargs,
    ):
        kwargs["accountant"] = "rdp"
        super(AdaptivePrivacyEngine, self).__init__(*args, **kwargs)
        self.alphas = self.accountant.DEFAULT_ALPHAS
        self.privacy_ledger = {}
        self.sample_size = sample_size
        self.n_accumulation_steps = n_accumulation_steps
        self.target_delta = target_delta

    def update_batch_size(self, new_batch_size, new_n_accumulation_steps):
        self._commit_to_privacy_ledger()
        self.batch_size = new_batch_size
        self.n_accumulation_steps = new_n_accumulation_steps
        self.sample_rate = self.batch_size / self.sample_size

    def update_noise_multiplier(self, new_noise_multiplier):
        self._commit_to_privacy_ledger()
        self.noise_multiplier = new_noise_multiplier

    def _commit_to_privacy_ledger(self):
        privacy_ledger_key = (self.sample_rate, self.noise_multiplier)
        if privacy_ledger_key not in self.privacy_ledger:
            self.privacy_ledger[privacy_ledger_key] = 0
        self.privacy_ledger[privacy_ledger_key] += self.steps

        self.steps = 0

    def get_renyi_divergence(self, sample_rate, noise_multiplier):
        return tf_privacy.compute_rdp(
            q=sample_rate,
            noise_multiplier=noise_multiplier,
            steps=1,
            orders=self.alphas,
        )

    def add_query_to_ledger(self, sample_rate, noise_multiplier, n):
        privacy_ledger_key = (sample_rate, noise_multiplier)
        if privacy_ledger_key not in self.privacy_ledger:
            self.privacy_ledger[privacy_ledger_key] = 0
        self.privacy_ledger[privacy_ledger_key] += n

    def get_privacy_spent(
        self,
        target_delta: Optional[float] = None,
        commit_to_ledger: Optional[bool] = False,
    ) -> Tuple[float, float]:
        """
        Computes the (epsilon, delta) privacy budget spent so far.
        This method converts from an (alpha, epsilon)-DP guarantee for all alphas that
        the ``PrivacyEngine`` was initialized with. It returns the optimal alpha together
        with the best epsilon.
        Args:
            target_delta: The Target delta. If None, it will default to the privacy
                engine's target delta.
        Returns:
            Pair of epsilon and optimal order alpha.
        """

        if target_delta is None:
            target_delta = self.target_delta
        if commit_to_ledger:
            self._commit_to_privacy_ledger()
        rdp = 0.0
        for (sample_rate, noise_multiplier), steps in self.privacy_ledger.items():
            rdp += self.get_renyi_divergence(sample_rate, noise_multiplier) * steps

        return tf_privacy.get_privacy_spent(
            orders=self.alphas, rdp=rdp, delta=target_delta
        )


class PrivacyFilterEngine(AdaptivePrivacyEngine):
    def __init__(self, epsilon, *args, **kwargs):
        super(PrivacyFilterEngine, self).__init__(*args, **kwargs)
        self.epsilon = epsilon

    def halt(
        self,
        batch_size: Optional[int] = None,
        sample_rate: Optional[float] = None,
        noise_multiplier: Optional[float] = None,
        steps: Optional[int] = 1,
        commit_to_ledger: Optional[bool] = False,
    ) -> bool:
        r"""
        Returns whether the filter would halt if asked to perform one more step
        at the proposed batch_size/sample_rate and noise_multiplier. If None the
        current PrivacyEngine values are used.
        Args:
            batch_size: The proposed new query batch size.
            sample_rate: The proposed new query sample rate (either or both
                batch_size and sample_rate have to be None).
            noise_multiplier: The proposed new query noise multiplier.
            steps: Would the filter halt within this number of steps.
        Returns:
            True (halt) or False (don't halt).
        """
        assert batch_size is None or sample_rate is None

        # TODO: implement through a max epsilon for each order alpha, and a
        # direct check of positivity for at least one alpha. Should be much more
        # efficient.

        if batch_size is not None:
            sample_rate = batch_size / self.sample_sfize
        elif sample_rate is None:
            sample_rate = self.sample_rate

        if noise_multiplier is None:
            noise_multiplier = self.noise_multiplier

        if commit_to_ledger:
            self._commit_to_privacy_ledger()
        self.add_query_to_ledger(sample_rate, noise_multiplier, steps)
        halt = self.get_privacy_spent(target_delta=self.target_delta)[0] > self.epsilon
        self.add_query_to_ledger(sample_rate, noise_multiplier, -steps)

        return halt

    def step(self):
        if not self.halt():
            super(PrivacyFilterEngine, self).step()


class PrivacyOdometerEngine(AdaptivePrivacyEngine):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        r"""
        Args:
            *args: Arguments for the underlying PrivacyEngine. See
                https://opacus.ai/api/privacy_engine.html.
            **kwargs: Keyword arguments for the underlying PrivacyEngine.
        """
        super(PrivacyOdometerEngine, self).__init__(*args, **kwargs)

        self.gamma = (
            2**-2
            * np.log(2 * len(self.alphas) / self.target_delta)
            / (np.atleast_1d(self.alphas) - 1)
        )

    def get_privacy_spent(
        self, commit_to_ledger: Optional[bool] = False
    ) -> Tuple[float, float]:
        """
        Computes the (epsilon, delta) privacy budget spent so far.
        This method converts from an (alpha, epsilon)-DP guarantee for all alphas that
        the ``PrivacyEngine`` was initialized with. It returns the optimal alpha together
        with the best epsilon.
        Args:
            target_delta: The Target delta. If None, it will default to the privacy
                engine's target delta.
        Returns:
            Pair of epsilon and optimal order alpha.
        """
        if commit_to_ledger:
            self._commit_to_privacy_ledger()
        rdp = 0.0

        for (sample_rate, noise_multiplier), steps in self.privacy_ledger.items():
            rdp += self.get_renyi_divergence(sample_rate, noise_multiplier) * steps

        rdp = np.maximum(rdp, self.gamma)
        f = np.ceil(np.log2(rdp / self.gamma))
        target_delta = self.target_delta / (len(self.alphas) * 2 * np.power(f + 1, 2))
        rdp = self.gamma * np.exp2(f)

        return self.get_privacy_spent_heterogeneous_delta(
            self.alphas, rdp, target_delta
        )

    def get_privacy_spent_heterogeneous_delta(
        self,
        orders: Union[List[float], float],
        rdp: Union[List[float], float],
        delta: Union[List[float], float],
    ) -> Tuple[float, float]:
        r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
        multiple RDP orders and target ``delta``.
        Args:
            orders: An array (or a scalar) of orders (alphas).
            rdp: A list (or a scalar) of RDP guarantees.
            delta: A list (or a scalar) of target deltas for each order.
        Returns:
            Pair of epsilon and optimal order alpha.
        Raises:
            ValueError
                If the lengths of ``orders`` and ``rdp`` are not equal.
        """
        orders_vec = np.atleast_1d(orders)
        rdp_vec = np.atleast_1d(rdp)
        delta_vec = np.atleast_1d(delta)

        if len(orders_vec) != len(rdp_vec) or len(orders_vec) != len(delta_vec):
            raise ValueError(
                f"Input lists must have the same length.\n"
                f"\torders_vec = {orders_vec}\n"
                f"\trdp_vec = {rdp_vec}\n"
                f"\tdelta_vec = {delta_vec}\n"
            )

        eps = rdp_vec - np.log(delta) / (orders_vec - 1)

        # special case when there is no privacy
        if np.isnan(eps).all():
            return np.inf, np.nan

        idx_opt = np.nanargmin(eps)  # Ignore NaNs
        return eps[idx_opt], orders_vec[idx_opt]
