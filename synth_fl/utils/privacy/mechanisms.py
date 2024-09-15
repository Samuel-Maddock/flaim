import math
import warnings
from functools import partial

import numpy as np
from autodp.calibrator_zoo import generalized_eps_delta_calibrator
from autodp.mechanism_zoo import GaussianMechanism as ADPGaussMech
from autodp.mechanism_zoo import LaplaceMechanism as ADPLapMech
from autodp.mechanism_zoo import ExponentialMechanism as ADPExpMech
from autodp.mechanism_zoo import Mechanism as ADPMechanism
from autodp.mechanism_zoo import (
    PureDP_Mechanism,
    SubsampleGaussianMechanism,
)
from autodp.transformer_zoo import AmplificationBySampling, Composition
from scipy.special import softmax

from synth_fl.utils.privacy.accountants import PrivacyAccountant
from synth_fl.utils import logger

warnings.filterwarnings(action="ignore", category=RuntimeWarning)


class Mechanism:
    def __init__(self, accountant: PrivacyAccountant, prng=np.random) -> None:
        self.accountant = accountant
        self.privacy_cost = {
            "zcdp": lambda x: float("inf"),
            "rdp": lambda x: float("inf"),
        }
        self.prng = prng

    def accumulate(self, noise_param, T=1):
        for _ in range(T):
            self.accountant.accumulate_cost(
                self.privacy_cost[self.accountant.type](noise_param)
            )

    def apply(self, x, sigma, sensitivity=1, no_accumulate=False):
        return x

    def get_privacy_cost(self, noise_param):
        return self.privacy_cost[self.accountant.type](noise_param)


class GaussianMechanism(Mechanism):
    def __init__(self, accountant: PrivacyAccountant, prng=np.random) -> None:
        super(GaussianMechanism, self).__init__(accountant, prng)
        self.privacy_cost = {
            "zcdp": lambda sigma: np.float64(0.5) / sigma**2,
            "rdp": lambda sigma: float("inf"),
            "autodp": lambda sigma: (ADPGaussMech(sigma, name="GM"), 1),
        }
        self.sigma = None

    def _gaussian_noise(self, sigma, size):
        return self.prng.normal(0, sigma, size)

    def get_cdp_sigma(self, num_rounds, rho):
        return np.sqrt(num_rounds / (2 * rho))

    def get_rdp_sigma(self, num_rounds, q=1, epsilon=None, delta=None):
        params = {"prob": q, "coeff": num_rounds, "sigma": None}
        general_calibrate = generalized_eps_delta_calibrator()

        epsilon = self.accountant.epsilon if epsilon is None else epsilon
        delta = self.accountant.delta if delta is None else delta
        mech = general_calibrate(
            SubsampleGaussianMechanism,
            epsilon,
            delta,
            [0.001, 1000],
            params=params,
            para_name="sigma",
            name="Subsampled_Gaussian",
        )
        return mech.params["sigma"]

    def apply(self, x, sigma, sensitivity=1, no_accumulate=False):
        if not no_accumulate:
            self.accountant.accumulate_cost(
                self.privacy_cost[self.accountant.type](sigma)
            )
        return x + self._gaussian_noise(sigma * sensitivity, len(x))


# Autodp mechanism - used for calibration
class SubsampleExponentialMechanism(ADPMechanism):
    def __init__(
        self, params, RDP_off=False, neighboring="remove_only", name="SubsampleGaussian"
    ):
        ADPMechanism.__init__(self)
        self.name = name
        self.params = {
            "prob": params["prob"],
            "eps": params["eps"],
            "coeff": params["coeff"],
        }

        if not RDP_off:
            # create such a mechanism as in previously
            subsample = (
                AmplificationBySampling()
            )  # by default this is using poisson sampling
            mech = ADPExpMech(eps=params["eps"])
            # mech = PureDP_Mechanism(eps=params["eps"])

            # Create subsampled Gaussian mechanism
            Subsample_exp_mech = subsample(
                mech, params["prob"], improved_bound_flag=True
            )

            # Now run this for niter iterations
            compose = Composition()
            mech = compose([Subsample_exp_mech], [params["coeff"]])

            # Now we get it and let's extract the RDP function and assign it to the current mech being constructed
            rdp_total = mech.RenyiDP
            self.propagate_updates(rdp_total, type_of_update="RDP")


class ExponentialMechanism(Mechanism):
    def __init__(self, accountant: PrivacyAccountant, prng=np.random) -> None:
        super().__init__(accountant, prng)
        self.privacy_cost = {
            "zcdp": lambda eps: 1 / 8 * eps**2,
            "autodp": lambda eps: (ADPExpMech(eps, name="ExpMech"), 1),
            "rdp": lambda sigma: float("inf"),
        }

    def get_cdp_sigma(self, num_rounds, rho):
        return np.sqrt(8 * rho / num_rounds)

    def get_rdp_sigma(self, num_rounds, q=1, epsilon=None, delta=None):
        params = {"prob": q, "coeff": num_rounds, "eps": None}
        general_calibrate = generalized_eps_delta_calibrator()
        epsilon = self.accountant.epsilon if epsilon is None else epsilon
        delta = self.accountant.delta if delta is None else delta
        mech = general_calibrate(
            SubsampleExponentialMechanism,
            epsilon,
            delta,
            [0, 20],
            params=params,
            para_name="eps",
            name="SubsampledExpMech",
        )
        return mech.params["eps"]

    def apply(self, qualities, eps, sensitivity=1.0, base_measure=0):
        if isinstance(qualities, dict):
            keys = list(qualities.keys())
            qualities = np.array([qualities[key] for key in keys])
            if base_measure != 0:
                base_measure = np.log([base_measure[key] for key in keys])
        else:
            qualities = np.array(qualities)
            keys = np.arange(qualities.size)

        self.accountant.accumulate_cost(self.privacy_cost[self.accountant.type](eps))

        if eps == float("inf"):
            return keys[np.argmax(qualities)]
        else:
            q = qualities - qualities.max()
            p = softmax(0.5 * eps / sensitivity * q + base_measure)
            logger.debug(
                f"Exponential mech eps={eps}, sens={sensitivity}, maxp={np.max(p)}"
            )
            return keys[self.prng.choice(p.size, p=p)]


class LaplaceMechanism(Mechanism):
    def __init__(self, accountant: PrivacyAccountant, prng=np.random) -> None:
        super().__init__(accountant, prng)
        self.privacy_cost = {
            "zcdp": lambda b: 0.5 * (1 / b) ** 2,
            "rdp": lambda b: None,
            "autodp": lambda b: (ADPLapMech(b, name="LapMech"), 1),
        }
        NotImplementedError("laplace not implemented")

    def get_cdp_sigma(self, num_rounds, rho):
        return 1 / np.sqrt(2 * rho / num_rounds)

    def _laplace_noise(self, b, size):
        return self.prng.laplace(0, b, size)

    def apply(self, x, b, sensitivity=1, no_accumulate=False):
        if not no_accumulate:
            self.accountant.accumulate_cost(self.privacy_cost[self.accountant.type](b))
        return x + self._laplace_noise(b * sensitivity, size=len(x))


# TODO: Integrate the below functions into mechanisms if needed


def pareto_efficient(costs):
    eff = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if eff[i]:
            eff[eff] = np.any(
                costs[eff] <= c, axis=1
            )  # Keep any point with a lower cost
    return np.nonzero(eff)[0]


def generalized_em_scores(q, ds, t):
    q = -q
    idx = pareto_efficient(np.vstack([q, ds]).T)
    r = q + t * ds
    r = r[:, None] - r[idx][None, :]
    z = ds[:, None] + ds[idx][None, :]
    s = (r / z).max(axis=1)
    return -s


def generalized_exponential_mechanism(
    qualities, sensitivities, epsilon, t=None, base_measure=None
):
    if t is None:
        t = 2 * np.log(len(qualities) / 0.5) / epsilon
    if isinstance(qualities, dict):
        keys = list(qualities.keys())
        qualities = np.array([qualities[key] for key in keys])
        sensitivities = np.array([sensitivities[key] for key in keys])
        if base_measure is not None:
            base_measure = np.log([base_measure[key] for key in keys])
    else:
        keys = np.arange(qualities.size)
    scores = generalized_em_scores(qualities, sensitivities, t)
    key = self.exponential_mechanism(scores, epsilon, 1.0, base_measure=base_measure)
    return keys[key]


def permute_and_flip(qualities, epsilon, sensitivity=1.0):
    """Sample a candidate from the permute-and-flip mechanism"""
    q = qualities - qualities.max()
    p = np.exp(0.5 * epsilon / sensitivity * q)
    for i in np.random.permutation(p.size):
        if np.random.rand() <= p[i]:
            return i


def best_noise_distribution(l1_sensitivity, l2_sensitivity, epsilon, delta):
    """Adaptively determine if Laplace or Gaussian noise will be better, and
    return a function that samples from the appropriate distribution"""
    b = self.laplace_noise_scale(l1_sensitivity, epsilon)
    sigma = self.gaussian_noise_scale(l2_sensitivity, epsilon, delta)
    dist = self.gaussian_noise if np.sqrt(2) * b > sigma else self.laplace_noise
    if np.sqrt(2) * b < sigma:
        return partial(self.laplace_noise, b)
    return partial(self.gaussian_noise, sigma)
