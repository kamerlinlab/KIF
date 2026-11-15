"""
Build the committed test data in tests/data/.

You only need to run this if you deliberately want to change the test data.
Run from the repo root:
    uv run python -m tests.make_test_data

It writes:
1. contacts_psf_dcd_10_frames_expected.csv
    Expected output of calculate_contacts() on the first 10 frames of the MDAnalysis PSF/DCD
    test system. tests/e2e/test_e2e_contact_identification.py checks against it, so only
    regenerate it when a change in contact output is intended, and check the git diff.

2. ptp1b_contacts_subset.csv, ptp1b_classes_subset.txt, ptp1b_closed.pdb
    A subset of the PTP1B tutorial dataset, used by the end-to-end pipeline tests.
    Needs the PTP1B tutorial dataset to be downloaded first (see
    tutorials/Tutorial_PTP1B_Classification_ML_Stats.ipynb). Skipped if it is not present.
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
from MDAnalysis.tests.datafiles import PSF

from key_interactions_finder.contact_identification import calculate_contacts
from tests.conftest import (
    CONTACTS_EXPECTED,
    DATA_DIR,
    PTP1B_CLASSES,
    PTP1B_CONTACTS,
    PTP1B_PDB,
    TUTORIAL_PTP1B_DIR,
    write_first_frames,
)

FRAME_STRIDE = 100  # 10,000 frames -> 100 frames.
DECIMALS = 2


def make_contacts_expected() -> None:
    """Run calculate_contacts on the first 10 frames of the PSF/DCD system and save the result."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        short_dcd = Path(tmp_dir, "first_10_frames.dcd")
        write_first_frames(short_dcd)
        calculate_contacts(
            parm_file=PSF, traj_file=str(short_dcd), out_file=str(CONTACTS_EXPECTED), report_timings=False
        )
    print(f"Written {CONTACTS_EXPECTED}")


def make_ptp1b_subset() -> None:
    """Write the PTP1B contacts subset, matching class labels and the closed-state PDB."""
    if not TUTORIAL_PTP1B_DIR.exists():
        print(f"Skipping PTP1B subset: {TUTORIAL_PTP1B_DIR} not found (download the tutorial dataset first).")
        return

    blocks = [pd.read_csv(TUTORIAL_PTP1B_DIR / f"PTP1B_block{i}.csv") for i in range(1, 5)]
    contacts = pd.concat(blocks, join="outer", axis=1)
    classes = pd.read_csv(TUTORIAL_PTP1B_DIR / "WT_PTP1B_Class_Assingments.txt", header=None)

    contacts_subset = contacts.iloc[::FRAME_STRIDE].round(DECIMALS).reset_index(drop=True)
    classes_subset = classes.iloc[::FRAME_STRIDE].reset_index(drop=True)

    contacts_subset.to_csv(PTP1B_CONTACTS, index=False)
    classes_subset.to_csv(PTP1B_CLASSES, index=False, header=False)
    shutil.copyfile(TUTORIAL_PTP1B_DIR / "WT_PTP1B_Phospho_Enzyme_Closed.pdb", PTP1B_PDB)

    print(f"PTP1B contacts subset: {contacts_subset.shape}, classes: {classes_subset[0].value_counts().to_dict()}")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    make_contacts_expected()
    make_ptp1b_subset()
