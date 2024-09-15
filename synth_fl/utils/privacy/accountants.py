import math

from autodp.mechanism_zoo import GaussianMechanism, PureDP_Mechanism, Mechanism
from autodp.transformer_zoo import Composition, AmplificationBySampling

from typing import Optional, Union, List


class PrivacyAccountant:
    def __init__(self, epsilon, delta) -> None:
        self.epsilon = epsilon
        self.delta = delta

    def accumulate_cost(self, cost):
        pass


# CDP methods come from libs/private_pgm/mechanisms/cdp2adp.py which come from thomas steinke
class zCDPAccountant(PrivacyAccountant):
    def __init__(self, epsilon, delta, rho=None) -> None:
        super().__init__(epsilon, delta)
        self.type = "zcdp"
        self.rho = rho if rho is not None else self._cdp_rho(epsilon, delta)
        self.starting_rho = self.rho

    @property
    def rho_used(self) -> float:
        return self.starting_rho - self.rho

    # Our new bound:
    # https://arxiv.org/pdf/2004.00010v3.pdf#page=13
    def _cdp_delta(self, rho, eps):
        assert rho >= 0
        assert eps >= 0
        if rho == 0:
            return 0  # degenerate case

        # search for best alpha
        # Note that any alpha in (1,infty) yields a valid upper bound on delta
        # Thus if this search is slightly "incorrect" it will only result in larger delta (still valid)
        # This code has two "hacks".
        # First the binary search is run for a pre-specificed length.
        # 1000 iterations should be sufficient to converge to a good solution.
        # Second we set a minimum value of alpha to avoid numerical stability issues.
        # Note that the optimal alpha is at least (1+eps/rho)/2. Thus we only hit this constraint
        # when eps<=rho or close to it. This is not an interesting parameter regime, as you will
        # inherently get large delta in this regime.
        amin = 1.01  # don't let alpha be too small, due to numerical stability
        amax = (eps + 1) / (2 * rho) + 2
        for i in range(1000):  # should be enough iterations
            alpha = (amin + amax) / 2
            derivative = (2 * alpha - 1) * rho - eps + math.log1p(-1.0 / alpha)
            if derivative < 0:
                amin = alpha
            else:
                amax = alpha
        # now calculate delta
        delta = math.exp(
            (alpha - 1) * (alpha * rho - eps) + alpha * math.log1p(-1 / alpha)
        ) / (alpha - 1.0)
        return min(delta, 1.0)  # delta<=1 always

    # Above we compute delta given rho and eps, now we compute eps instead
    # That is we wish to compute the smallest eps such that rho-CDP implies (eps,delta)-DP
    def _cdp_eps(self, rho, delta):
        assert rho >= 0
        assert delta > 0
        if delta >= 1 or rho == 0:
            return 0.0  # if delta>=1 or rho=0 then anything goes
        epsmin = 0.0  # maintain cdp_delta(rho,eps)>=delta
        epsmax = rho + 2 * math.sqrt(
            rho * math.log(1 / delta)
        )  # maintain cdp_delta(rho,eps)<=delta
        # to compute epsmax we use the standard bound
        for i in range(1000):
            eps = (epsmin + epsmax) / 2
            if self._cdp_delta(rho, eps) <= delta:
                epsmax = eps
            else:
                epsmin = eps
        return epsmax

    # Now we compute rho
    # Given (eps,delta) find the smallest rho such that rho-CDP implies (eps,delta)-DP
    def _cdp_rho(self, eps, delta):
        assert eps >= 0
        assert delta > 0
        if delta >= 1:
            return 0.0  # if delta>=1 anything goes
        rhomin = 0.0  # maintain cdp_delta(rho,eps)<=delta
        rhomax = eps + 1  # maintain cdp_delta(rhomax,eps)>delta
        for i in range(1000):
            rho = (rhomin + rhomax) / 2
            if self._cdp_delta(rho, eps) <= delta:
                rhomin = rho
            else:
                rhomax = rho
        return rhomin

    def get_eps(self):
        return self._cdp_eps(self.starting_rho - self.rho, self.delta)

    def accumulate_cost(self, cost):
        self.rho -= cost


class AutoDPAccountant(PrivacyAccountant):
    def __init__(self, epsilon, delta, sample_rate=1) -> None:
        super().__init__(epsilon, delta)
        self.type = "autodp"
        self.mechanism_list = []
        self.count_list = []
        self.composed_mech = None
        self.sample_rate = sample_rate

        self.cache_mech_details = [self.mechanism_list, self.count_list]

        self.epsilon = epsilon
        self.current_epsilon = 0

    def _cache_mech(self):
        self.cache_mech_details = [self.mechanism_list.copy(), self.count_list.copy()]

    def _has_mech_changed(self):
        return (
            (self.mechanism_list != self.cache_mech_details[0])
            or (self.count_list != self.cache_mech_details[1])
            or (self.composed_mech is None)
        )

    def _compose_mech(self, force_compose=False):
        if self._has_mech_changed() and len(self.mechanism_list) > 0 or force_compose:
            mech_list = self.mechanism_list
            if self.sample_rate < 1:
                subsample = AmplificationBySampling()
                mech_list = [
                    subsample(mech, self.sample_rate, improved_bound_flag=True)
                    for mech in mech_list
                ]
            compose = Composition()
            self.composed_mech = compose(mech_list, self.count_list)
            self._cache_mech()

    def get_eps(self, force_compose=False):
        if len(self.mechanism_list) > 0:
            self._compose_mech(force_compose=force_compose)
            self.current_epsilon = self.composed_mech.get_eps(delta=self.delta)
            return self.current_epsilon
        else:
            return 0

    def accumulate_cost(self, cost):
        mech, count = cost
        self.mechanism_list.append(mech)
        self.count_list.append(count)

    # Lecuyer et al. RDP Privacy Filter https://arxiv.org/pdf/2103.01379.pdf
    def halt(
        self,
        mechanism: Union[Mechanism, List[Mechanism]],
        steps: Optional[Union[int, List[int]]] = 1,
    ):
        if not isinstance(mechanism, list):
            mechanism, steps = [mechanism], [steps]

        self.mechanism_list.extend(mechanism)
        self.count_list.extend(steps)
        halt = self.get_eps() > self.epsilon

        # Remove added mock mechs
        for i in range(len(mechanism)):
            self.mechanism_list.pop()
            self.count_list.pop()

        return halt


# TODO: Implement?
class RDPAccountant(PrivacyAccountant):
    def __init__(self, epsilon, delta) -> None:
        super().__init__(epsilon, delta)
        self.type = "rdp"

    def accumulate_cost(self, cost):
        pass
