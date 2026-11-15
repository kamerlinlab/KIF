"""
End-to-end: contact identification on an MD trajectory -> contacts CSV.

Uses the MDAnalysis testing PSF/DCD files.
"""

import re

import pandas as pd
import pytest
from MDAnalysis.tests.datafiles import DCD, PSF

from key_interactions_finder.contact_identification import calculate_contacts
from tests.conftest import CONTACTS_EXPECTED


def _pairs_from_columns(columns) -> list:
    return [tuple(int(n) for n in re.findall(r"(\d+)[A-Za-z]", col)[:2]) for col in columns]


@pytest.fixture(scope="module")
def contacts_10_frames(short_psf_dcd, tmp_path_factory) -> pd.DataFrame:
    parm, traj = short_psf_dcd
    out_file = tmp_path_factory.mktemp("contacts") / "contacts.csv"
    calculate_contacts(parm_file=parm, traj_file=traj, out_file=str(out_file), report_timings=False)
    return pd.read_csv(out_file)


def test_matches_expected_output(contacts_10_frames):
    """
    Contacts for the first 10 frames of the PSF/DCD system match the committed expected output.

    If you deliberately change the output (e.g. new interaction typing rules), regenerate the
    expected output with `uv run python -m tests.make_test_data` and check the git diff.
    """
    expected = pd.read_csv(CONTACTS_EXPECTED)
    pd.testing.assert_frame_equal(contacts_10_frames, expected, check_exact=False, atol=1e-4, rtol=0)


def test_output_format(contacts_10_frames):
    df = contacts_10_frames
    assert len(df) == 10

    # Column names are "[resnum][Resname] [resnum][Resname] [type]".
    label_regex = re.compile(r"^\d+[A-Z][a-z]{2} \d+[A-Z][a-z]{2} (Hbond|Saltbr|Hydrophobic|Other)$")
    assert all(label_regex.match(col) for col in df.columns)

    # Columns are ordered by (res1, res2).
    pairs = _pairs_from_columns(df.columns)
    assert pairs == sorted(pairs)

    # Neighbouring residues (gap < 3) are never reported.
    assert all(res2 - res1 >= 3 for res1, res2 in pairs)

    # Only contacts with an average score >= 0.1 are kept, and scores are non-negative.
    assert (df.mean() >= 0.1).all()
    assert (df >= 0).all().all()


def test_residue_subset(short_psf_dcd, tmp_path):
    """first_res/last_res restricts the first residue of each pair."""
    parm, traj = short_psf_dcd
    out_file = tmp_path / "contacts_subset.csv"
    calculate_contacts(
        parm_file=parm, traj_file=traj, out_file=str(out_file), first_res=10, last_res=20, report_timings=False
    )
    pairs = _pairs_from_columns(pd.read_csv(out_file).columns)
    assert pairs
    assert all(10 <= res1 <= 20 for res1, _ in pairs)


def test_full_trajectory(tmp_path):
    """Full 98 frame run, exactly as tutorials/identify_contacts.py does it (~10 s)."""
    out_file = tmp_path / "contacts_all_res.csv"
    calculate_contacts(parm_file=PSF, traj_file=DCD, out_file=str(out_file), report_timings=False)
    df = pd.read_csv(out_file)
    assert df.shape == (98, 658)
    pairs = _pairs_from_columns(df.columns)
    assert pairs == sorted(pairs)
