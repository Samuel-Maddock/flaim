# FLAIM: AIM-based synthetic data generation in the federated setting
Code for the paper ['FLAIM: AIM-based synthetic data generation in the federated setting'](https://arxiv.org/abs/2310.03447)

## Citing this work
```bibtex
@inproceedings{flaim,
author = {Maddock, Samuel and Cormode, Graham and Maple, Carsten},
title = {FLAIM: AIM-based Synthetic Data Generation in the Federated Setting},
year = {2024},
isbn = {9798400704901},
url = {https://doi.org/10.1145/3637528.3671990},
doi = {10.1145/3637528.3671990},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {2165–2176},
numpages = {12},
location = {Barcelona, Spain},
series = {KDD '24}
}
```

# Installation Instructions

Install the required Python environment via conda and pip

```
conda create -n "flaim" python=3.9 
conda activate flaim
pip install -r ./requirements.txt
```
**AutoDP Dependency:** Download the repo [here](https://github.com/yuxiangw/autodp) and run in the root of autodp.

```
pip install .
```

## Datasets

All datasets are downloaded automatically via [PMLB](https://epistasislab.github.io/pmlb/) or the [Synthetic Data Vault](https://sdv.dev) during the first run except for the Adult, Covtype and Marketing datasets which require manual downloading:
* Download 'adult.csv' from [here](https://github.com/ryan112358/private-pgm/tree/master/data) and place under `/synth_fl/data/`
* Download 'covtype.csv' from [here](https://archive.ics.uci.edu/dataset/31/covertype) and place under `/synth_fl/data/`
* Download 'marketing.csv' from [here](https://archive.ics.uci.edu/dataset/222/bank+marketing) and place under `synth_fl/data/`

## Caching client partitions

In order to run experiments the partitions of client data into the federated setting must be formed. The partitions are generated and stored under `synth_fl/data`

Run the following Python code to produce non-IID partitions of the benchmark datasets:

```
python3.9 launcher.py --sweep-name paper/cache_answers/cache_split_answers1.json --sweep-manager-type local --sweep-backend local
```

and to produce the SynthFS synthetic dataset:

```
python3.9 launcher.py --sweep-name paper/cache_answers/cache_split_answers2.json --sweep-manager-type local --sweep-backend local
```

Note this may take a while (around ~10-20 minutes) and will save client partition splits to `synth_fl/data`.


# Replication Instructions

Further configs for experiments are contained within `sweep_configs/paper/`.

These can be run locally (over 4 CPU threads) as follows:
```
python3.9 launcher.py --sweep-backend local --sweep-manager-type local --sweep-name paper/SWEEP_NAME --workers 4
```

Data from experiments is saved under `slurm/job_results` and `slurm/sweep_results`.

`SWEEP_NAME` should be one of the following config files contained within `sweep_configs/paper`:

* `varying_eps` - Used to produce Figure 3(a) in the main paper and Figure 6 in the Appendix.
* `varying_feature_skew` - Used to produce Figure 1.
* `varying_local_rounds` - Used to produce Figures 3(e,f) in the main paper and Figure 10 in the Appendix.
* `varying_p` - Used to produce Figure 3(c) in the main paper and Figure 8 in the Appemndix.
* `varying_t` - Used to produce Figures 3(b) and Table 1 in the main paper and Figure 7 and Table 6 in the Appendix.
* `varying_beta` - prodcues Figures 3(d) in the main paper and Figure 9 in the Appendix.
* `baselines.json` - Will train FLAIM baseline methods, used to produce Table 1
* `communication_tracking` - Used to produce Table 3 in the main paper and Table 7 in the Appendix.
* `appendix_non_iid_split` - Used to produce Table 5 in the Appendix.

# Acknowledgements

We would like to acknowledge the following code that is used by this repo:
* [PMLB](https://epistasislab.github.io/pmlb/) - For dataset loading
* [Synthetic data vault](https://sdv.dev) - For additional datasets
* [Private-PGM](https://github.com/ryan112358/private-pgm) by Ryan McKenna
* [AutoDP](https://github.com/yuxiangw/autodp) by Yu-Xiang Wang
* [FLSim](https://github.com/facebookresearch/FLSim) by Facebook Research (for running federated CTGAN examples)
