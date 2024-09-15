import argparse

from synth_fl.simulation.experiment_runners import LocalRunner, SlurmRunner


def parse_args():
    parser = argparse.ArgumentParser(description="SynthFL")

    parser.add_argument(
        "--sweep-name",
        type=str,
        default="debug",
        help="Sweep .json name (under synth_fl/sweep_configs/)",
    )

    parser.add_argument(
        "--sweep-manager-type",
        type=str,
        default="local",  # wandb or local
        help="Sweep manager - options: wandb, local",
    )

    parser.add_argument(
        "--sweep-backend",
        type=str,
        default="local",
        help="Sweep backend, options: local, slurm",  # local, slurm
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers for local multiprocessing or slurm, default=1",
    )

    parser.add_argument(
        "--sweep-id",
        type=str,
        default="",
        help="wandb sweep id to resume sweep on slurm",
    )

    parser.add_argument(
        "--task-mem", type=float, default=3.0, help="Memory (in GB) for slurm tasks"
    )

    parser.add_argument(
        "--tasks-per-node", type=int, default=1, help="Tasks per node for slurm"
    )

    parser.add_argument(
        "--cpus-per-task", type=int, default=1, help="CPUs per tasks for slurm"
    )

    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to run local slurm sweep over",
    )

    parser.add_argument(
        "--sweep-rank",
        type=int,
        default=-1,
        help="When running sweep from checkpoint, override for sweep rank",
    )

    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Slurm exclude string",
    )

    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--multi-thread", default=False, action="store_true")

    parser.add_argument("--multi-sweep", default=False, action="store_true")

    parser.add_argument(
        "--disable-catching-task-errors", default=True, action="store_false"
    )

    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = parse_args()

    if args.sweep_backend == "slurm":
        runner = SlurmRunner(
            sweep_name=args.sweep_name,
            sweep_manager_type=args.sweep_manager_type,
            n_workers=args.workers,
            task_mem=args.task_mem,
            tasks_per_node=args.tasks_per_node,
            cpus_per_task=args.cpus_per_task,
            debug=args.debug,
            sweep_id=args.sweep_id,
            sweep_rank=args.sweep_rank,
            nodes=args.nodes,
            multi_sweep=args.multi_sweep,
            exclude=args.exclude,
            catch_task_errors=args.disable_catching_task_errors,
            multi_thread=args.multi_thread
        )
    else:
        runner = LocalRunner(
            sweep_name=args.sweep_name,
            sweep_manager_type=args.sweep_manager_type,
            n_workers=args.workers,
            nodes=args.nodes,
            catch_task_errors=args.disable_catching_task_errors,
            debug=args.debug,
        )

    runner.begin()
