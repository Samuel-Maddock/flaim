import argparse
import sys

import numpy as np

# Import from private-PGM
sys.path.append("./")
from mbi import mechanism

from examples import benchmarks
from examples.privbayes import privbayes_inference, privbayes_measurements


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params["dataset"] = "adult"
    params["iters"] = 5000
    params["epsilon"] = 1.0
    params["seed"] = 0

    return params


description = ""
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
parser.add_argument("--dataset", choices=["adult"], help="dataset to use")
parser.add_argument("--iters", type=int, help="number of optimization iterations")
parser.add_argument("--epsilon", type=float, help="privacy  parameter")
parser.add_argument("--seed", type=int, help="random seed")

parser.set_defaults(**default_params())
args = parser.parse_args()

data, workload = benchmarks.adult_benchmark()
total = data.df.shape[0]

measurements = privbayes_measurements(data, 1.0, args.seed)

est, _ = privbayes_inference(data.domain, measurements, total=total)

elim_order = [m[3][0] for m in measurements][::-1]

projections = [m[3] for m in measurements]
est2, _, _ = mechanism.run(
    data, projections, eps=args.epsilon, frequency=50, seed=args.seed, iters=args.iters
)


def err(true, est):
    return np.sum(np.abs(true - est)) / true.sum()


err_pb = []
err_pgm = []
for p, W in workload:
    true = W.dot(data.project(p).datavector())
    pb = W.dot(est.project(p).datavector())
    pgm = W.dot(est2.project(p).datavector())
    err_pb.append(err(true, pb))
    err_pgm.append(err(true, pgm))

print("Error of PrivBayes    : %.3f" % np.mean(err_pb))
print("Error of PrivBayes+PGM: %.3f" % np.mean(err_pgm))
