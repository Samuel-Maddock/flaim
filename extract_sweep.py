import argparse
import glob
import os
import pandas as pd

from synth_fl.simulation.experiment_runners import SLURM_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="extract_sweep")

    parser.add_argument(
        "--sweep-id",
        type=str,
        default="",
        help="wandb sweep id to extract on slurm",
    )

    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = parse_args()
    path = f"{SLURM_PATH}sweep_results/{args.sweep_id}/"
    extension = "csv"
    os.chdir(path)
    results = glob.glob("*.{}".format(extension))
    output_file = f"sweep-{args.sweep_id}_full.csv"
    output_path = f"{SLURM_PATH}sweep_results/{args.sweep_id}/{output_file}"
    final_df = pd.DataFrame([])
    for result in results:
        if result != output_file and "full" not in result:
            print(f"Reading {result}...")
            df = pd.read_csv(result)
            final_df = pd.concat([final_df, df])
    final_df.to_csv(output_path, index=False)
    print(f"Sweep data saved to {output_path}")
