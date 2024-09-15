from synth_fl.libs.ektelo import workload
from synth_fl.libs.private_pgm.mbi import Dataset
from synth_fl.utils import DEFAULT_DATA_ROOT


def adult_benchmark(subset=None):
    data = Dataset.load(f"{DEFAULT_DATA_ROOT}/adult.csv", f"{DEFAULT_DATA_ROOT}/adult-domain.json", subset=subset)

    projections = [
        ("occupation", "race", "capital-loss"),
        ("occupation", "sex", "native-country"),
        ("marital-status", "relationship", "income>50K"),
        ("age", "education-num", "sex"),
        ("workclass", "education-num", "occupation"),
        ("marital-status", "occupation", "income>50K"),
        ("race", "native-country", "income>50K"),
        ("occupation", "capital-gain", "income>50K"),
        ("marital-status", "hours-per-week", "income>50K"),
        ("workclass", "race", "capital-gain"),
        ("marital-status", "relationship", "capital-gain"),
        ("workclass", "education-num", "capital-gain"),
        ("education-num", "relationship", "race"),
        ("fnlwgt", "hours-per-week", "income>50K"),
        ("workclass", "sex", "native-country"),
    ]

    lookup = {}
    for attr in data.domain:
        n = data.domain.size(attr)
        lookup[attr] = workload.Identity(n)

    lookup["age"] = workload.Prefix(85)
    lookup["fnlwgt"] = workload.Prefix(100)
    lookup["capital-gain"] = workload.Prefix(100)
    lookup["capital-loss"] = workload.Prefix(100)
    lookup["hours-per-week"] = workload.Prefix(99)

    workloads = []

    for proj in projections:
        W = workload.Kronecker([lookup[a] for a in proj])
        workloads.append((proj, W))

    return data, workloads
