"""Simple Darcy Dataset Example."""

from __future__ import annotations

from neuralop.data.datasets.darcy import DarcyDataset
from neuralop.data.datasets.pt_dataset import PTDataset
from neuralop.utils import get_project_root


def get_darcy_data() -> tuple[PTDataset, PTDataset]:
    """Simple darcy dataset."""
    root_dir = get_project_root() / 'neuralop/data/datasets/data'
    data = DarcyDataset(
        root_dir=root_dir,
        n_train=100,
        n_tests=[32, 32, 32, 32],
        batch_size=16,
        test_batch_sizes=[16, 16, 16, 16],
        train_resolution=16,  # change resolution to download different data
        test_resolutions=[16, 32, 64, 128],
    )

    return data.train_db, data.test_dbs
