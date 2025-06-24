"""Setting up Parsl utilities for experiment parallelization."""

from __future__ import annotations

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider


def get_parsl_config() -> Config:
    """Initialize Parsl config.

    One experiment per GPU.
    Multiple experiment per node.
    """
    provider = SlurmProvider(
        launcher=SrunLauncher(
            overrides='--gpus-per-node 4 -c 64'
        ),  # Must supply GPUs and CPU per node
        walltime='00:30:00',
        nodes_per_block=1,  # how many nodes to request
        scheduler_options="""#SBATCH -C gpu&hbm80g
            #SBATCH --qos=debug
            #SBATCH --mail-user=sakarvadia@uchicago.edu""",
        # Switch to "-C cpu" for CPU partition
        account='m1266',
        worker_init="""
module load conda
conda activate /pscratch/sd/m/mansisak/operator_aliasing/env/
cd /pscratch/sd/m/mansisak/operator_aliasing/operator_aliasing/

# Print to stdout to for easier debugging
module list
nvidia-smi
which python
hostname
pwd""",
    )

    config = Config(
        executors=[
            HighThroughputExecutor(
                label='train_fno',
                available_accelerators=4,  # number of GPUs
                max_workers_per_node=4,
                cpu_affinity='block',
                provider=provider,
            )
        ]
    )
    return config
