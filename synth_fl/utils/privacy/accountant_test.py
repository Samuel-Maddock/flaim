import numpy as np

import sys

sys.path.extend("../../../")
from synth_fl.utils.privacy import (
    AutoDPAccountant,
    ExponentialMechanism,
    GaussianMechanism,
    zCDPAccountant,
)


def zcdp_example():
    x = np.random.uniform(size=100)

    epsilon, delta = 1, 1e-5
    accountant = zCDPAccountant(epsilon, delta)

    gauss_mech = GaussianMechanism(accountant=accountant)
    exp_mech = ExponentialMechanism(accountant=accountant)
    total_rho = accountant.rho

    sigma_gauss = gauss_mech.get_cdp_sigma(5, total_rho / 2)
    sigma_exp = exp_mech.get_cdp_sigma(3, total_rho / 2)

    utilities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for i in range(0, 5):
        x_prime = gauss_mech.apply(x, sigma_gauss)

    for i in range(0, 3):
        exp_mech.apply(utilities, eps=sigma_exp)

    assert round(accountant.get_eps()) == epsilon
    print("zCDP Accountant Passed...")


def autodp_example_gauss():
    x = np.random.uniform(size=100)

    epsilon, delta = 1, 1e-5
    accountant = AutoDPAccountant(epsilon, delta)
    gauss_mech = GaussianMechanism(accountant=accountant)
    sigma_gauss = gauss_mech.get_rdp_sigma(num_rounds=5)

    for i in range(0, 5):
        x_prime = gauss_mech.apply(x, sigma_gauss)

    assert round(accountant.get_eps()) == epsilon
    print("AutoDPAccountant (Gaussian Mech) Passed...")


def autodp_example_exp():
    x = np.random.uniform(size=100)
    num_rounds = 100
    epsilon, delta = 1, 1e-5
    accountant = AutoDPAccountant(epsilon, delta)
    exp_mech = ExponentialMechanism(accountant=accountant)
    exp_eps = exp_mech.get_rdp_sigma(num_rounds=num_rounds)

    print(f"Calibrated eps {exp_eps}")

    for i in range(0, num_rounds):
        exp_mech.apply(x, exp_eps)

    print(f"Privacy budget - {epsilon}")
    print(f"Privacy budget from accountant - {accountant.get_eps()}")

    assert round(accountant.get_eps()) == epsilon
    print("AutoDPAccountant (Exp Mech) Passed...")


def autodp_subsample_example():
    x = np.random.uniform(size=100)
    # sample_rate = 0.1

    sample_rate = 0.1
    epsilon, delta = 1, 1e-5

    accountant = AutoDPAccountant(epsilon, delta, sample_rate=sample_rate)
    gauss_mech = GaussianMechanism(accountant=accountant)
    exp_mech = ExponentialMechanism(accountant=accountant)

    sigma_gauss = gauss_mech.get_rdp_sigma(
        num_rounds=5, q=sample_rate, epsilon=epsilon / 2
    )
    print("Finding exp eps")
    exp_eps = exp_mech.get_rdp_sigma(num_rounds=1, q=sample_rate, epsilon=epsilon / 2)
    print("Found...")

    for i in range(0, 5):
        x_prime = gauss_mech.apply(x, sigma_gauss)
    exp_mech.apply(x_prime, exp_eps)

    print(f"Privacy budget - {epsilon}")
    print(f"Privacy budget from accountant - {accountant.get_eps()}")

    assert round(accountant.get_eps()) == epsilon
    print("AutoDPAccountant (Subsample + Composition) Passed...")


# zcdp_example()
# autodp_example_gauss()
# autodp_example_exp()
autodp_subsample_example()
