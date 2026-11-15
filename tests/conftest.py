"""Shared fixtures and test data paths for the KIF test suite."""

from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytest
from MDAnalysis.tests.datafiles import DCD, PSF

from key_interactions_finder.utils import download_prep_tutorial_dataset

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data"
CONTACTS_EXPECTED = DATA_DIR / "contacts_psf_dcd_10_frames_expected.csv"
TUTORIAL_DATASETS_DIR = TESTS_DIR.parent / "tutorials" / "tutorial_datasets"
TUTORIAL_PTP1B_DIR = TUTORIAL_DATASETS_DIR / "PTP1B_Tutorial" / "Input_data"
# Same link as in tutorials/Tutorial_PTP1B_Classification_ML_Stats.ipynb.
PTP1B_TUTORIAL_URL = "https://drive.google.com/file/d/1hJbwCCuTTgI4xglwu1vXyzo-yaZJbmUY/view?usp=share_link"
TUTORIAL_KE07_DIR = TUTORIAL_DATASETS_DIR / "KE07_Tutorial" / "Input_data"
# Same link as in tutorials/Tutorial_KE07_Regression_ML_Stats.ipynb.
KE07_TUTORIAL_URL = "https://drive.google.com/file/d/1pqFUMMjt9gDYOxtkVpDmyXwKXiHp0wFi/view?usp=share_link"

PTP1B_CONTACTS = DATA_DIR / "ptp1b_contacts_subset.csv"
PTP1B_CLASSES = DATA_DIR / "ptp1b_classes_subset.txt"
PTP1B_PDB = DATA_DIR / "ptp1b_closed.pdb"


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="Also run the slow tests in tests/slow/.")


def pytest_collection_modifyitems(config, items):
    """Skip tests marked slow unless --run-slow is given."""
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test, run with --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def ptp1b_tutorial_dir() -> Path:
    """The full PTP1B tutorial dataset (68 MB). Downloaded from Google Drive on first use."""
    if not TUTORIAL_PTP1B_DIR.exists():
        download_prep_tutorial_dataset(drive_url=PTP1B_TUTORIAL_URL, save_dir=str(TUTORIAL_DATASETS_DIR))
    return TUTORIAL_PTP1B_DIR


@pytest.fixture(scope="session")
def ke07_tutorial_dir() -> Path:
    """The full KE07 tutorial dataset (22 MB). Downloaded from Google Drive on first use."""
    if not TUTORIAL_KE07_DIR.exists():
        download_prep_tutorial_dataset(drive_url=KE07_TUTORIAL_URL, save_dir=str(TUTORIAL_DATASETS_DIR))
    return TUTORIAL_KE07_DIR


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """Run the test from inside a temporary directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


SYNTH_FEATURES = [
    "1Ala 5Leu Hydrophobic",  # perfectly separates the two classes.
    "2Ser 8Thr Hbond",  # low occupancy (non-zero in 10% of frames).
    "3Lys 9Glu Saltbr",  # very weak (mean ~0.005).
    "4Gly 10Gly Other",  # noise.
    "11Val 21Ile Hydrophobic",  # noise, used for residue-number regex checks (1 vs 11 vs 21).
]


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    """60 frames x 5 features. The first 30 frames are class "A", the last 30 are class "B"."""
    rng = np.random.default_rng(42)
    n_frames = 60
    separator = np.r_[rng.uniform(0.0, 0.2, 30), rng.uniform(0.8, 1.0, 30)]
    low_occupancy = np.zeros(n_frames)
    low_occupancy[rng.choice(n_frames, size=6, replace=False)] = rng.uniform(0.5, 1.0, 6)
    return pd.DataFrame(
        {
            SYNTH_FEATURES[0]: separator,
            SYNTH_FEATURES[1]: low_occupancy,
            SYNTH_FEATURES[2]: rng.uniform(0.0, 0.01, n_frames),
            SYNTH_FEATURES[3]: rng.uniform(0.0, 1.0, n_frames),
            SYNTH_FEATURES[4]: rng.uniform(0.0, 1.0, n_frames),
        }
    )


@pytest.fixture
def synthetic_class_labels() -> list:
    return ["A"] * 30 + ["B"] * 30


@pytest.fixture
def synthetic_target_file(tmp_path, synthetic_class_labels) -> str:
    """Class labels file (with header) matching synthetic_features."""
    path = tmp_path / "classes.csv"
    pd.DataFrame({"Class": synthetic_class_labels}).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def synthetic_supervised_df(synthetic_features, synthetic_class_labels) -> pd.DataFrame:
    """synthetic_features with a "Target" column first, like SupervisedFeatureData.df_processed."""
    df = synthetic_features.copy()
    df.insert(0, "Target", synthetic_class_labels)
    return df


@pytest.fixture
def synthetic_regression_df(synthetic_features) -> pd.DataFrame:
    """synthetic_features with a continuous "Target" driven by the first feature."""
    rng = np.random.default_rng(7)
    df = synthetic_features.copy()
    df.insert(0, "Target", 10 * df[SYNTH_FEATURES[0]] + rng.normal(0, 0.1, len(df)))
    return df


@pytest.fixture(scope="session")
def ptp1b_contacts() -> pd.DataFrame:
    """100 frames x 989 contacts from the PTP1B tutorial (every 100th frame)."""
    return pd.read_csv(PTP1B_CONTACTS)


@pytest.fixture
def ptp1b_classes_file() -> str:
    """Per-frame class labels ("Open", "Closed", "Neither") matching ptp1b_contacts. No header."""
    return str(PTP1B_CLASSES)


@pytest.fixture
def ptp1b_pdb() -> str:
    """Closed-state PTP1B structure (298 residues)."""
    return str(PTP1B_PDB)


def write_first_frames(out_dcd: Path, n_frames: int = 10) -> None:
    """Write the first n_frames of the MDAnalysis PSF/DCD test system (214 residues) to a new DCD."""
    universe = mda.Universe(PSF, DCD)
    with mda.Writer(str(out_dcd), universe.atoms.n_atoms) as writer:
        for _ in universe.trajectory[:n_frames]:
            writer.write(universe.atoms)


@pytest.fixture(scope="session")
def short_psf_dcd(tmp_path_factory) -> tuple:
    """The MDAnalysis PSF/DCD test system cut down to its first 10 frames. Returns (topology, trajectory)."""
    out_dcd = tmp_path_factory.mktemp("traj") / "first_10_frames.dcd"
    write_first_frames(out_dcd)
    return PSF, str(out_dcd)
