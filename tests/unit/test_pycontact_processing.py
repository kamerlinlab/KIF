"""Tests for pycontact_processing: loading, merging and cleaning PyContact files."""

import pandas as pd
import pytest

from key_interactions_finder.pycontact_processing import PyContactInitializer, modify_column_residue_numbers


@pytest.fixture
def block_a() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "1Glu 2Met Hbond sc-bb": [1.0, 0.0],  # too close for an Hbond (gap <= 2), removed.
            "1Glu 4Lys Saltbr sc-sc": [2.0, 3.0],
            "4Lys 1Glu Saltbr sc-sc": [2.0, 3.0],  # duplicate of the above with residues swapped, removed.
            "5Glu 8Gln Other sc-sc": [1.0, 1.0],  # vdW type with gap <= 3, removed.
            "5Glu 9Ile Other sc-sc": [0.5, 0.5],
            "10Ala 20Leu Hbond sc-bb": [0.1, 0.2],
            "20Leu 10Ala Hbond bb-sc": [0.1, 0.2],  # sc-bb / bb-sc duplicate, removed.
        }
    )


@pytest.fixture
def block_b() -> pd.DataFrame:
    return pd.DataFrame({"30Ser 40Thr Hbond sc-sc": [0.7, 0.8]})


def test_single_file_custom_script(tmp_path, block_a):
    block_a.to_csv(tmp_path / "a.csv", index=False)
    pyc = PyContactInitializer(pycontact_files="a.csv", multiple_files=False, in_dir=str(tmp_path))
    assert list(pyc.prepared_df.columns) == [
        "1Glu 4Lys Saltbr sc-sc",
        "5Glu 9Ile Other sc-sc",
        "10Ala 20Leu Hbond sc-bb",
    ]


def test_keep_false_interactions(tmp_path, block_a):
    block_a.to_csv(tmp_path / "a.csv", index=False)
    pyc = PyContactInitializer(
        pycontact_files="a.csv", multiple_files=False, in_dir=str(tmp_path), remove_false_interactions=False
    )
    pd.testing.assert_frame_equal(pyc.prepared_df, block_a)


def test_merge_horizontally(tmp_path, block_a, block_b):
    block_a.to_csv(tmp_path / "a.csv", index=False)
    block_b.to_csv(tmp_path / "b.csv", index=False)
    pyc = PyContactInitializer(
        pycontact_files=["a.csv", "b.csv"],
        multiple_files=True,
        merge_files_method="horizontal",
        in_dir=str(tmp_path),
        remove_false_interactions=False,
    )
    assert pyc.prepared_df.shape == (2, 8)


def test_merge_horizontally_needs_equal_lengths(block_a):
    with pytest.raises(ValueError, match="not identical"):
        PyContactInitializer._merge_pycontact_datasets_horizontally([block_a, pd.concat([block_a, block_a])])


def test_multiple_files_need_merge_method(tmp_path, block_a):
    block_a.to_csv(tmp_path / "a.csv", index=False)
    with pytest.raises(ValueError, match="merge_files_method"):
        PyContactInitializer(pycontact_files=["a.csv", "a.csv"], multiple_files=True, in_dir=str(tmp_path))


def test_bad_output_type(tmp_path, block_a):
    block_a.to_csv(tmp_path / "a.csv", index=False)
    with pytest.raises(ValueError, match="custom_script"):
        PyContactInitializer(
            pycontact_files="a.csv", multiple_files=False, in_dir=str(tmp_path), pycontact_output_type="other"
        )


def test_clean_gui_feature_name():
    assert PyContactInitializer._clean_gui_feature_name("12ALA-45GLU hbond") == "12Ala 45Glu Hbond bb-bb"


def test_process_gui_file(tmp_path):
    """
    Minimal input shaped the way _process_gui_file parses it: a header row, then per line the contact,
    one summary value (discarded) and a bracketed list of per frame scores. Not a real GUI export.
    """
    gui_file = tmp_path / "gui.txt"
    gui_file.write_text("header line\n12ALA-45GLU hbond 0.5 [1.0, 0.0, 0.5]\n3LYS-50ASP saltbr 0.2 [0.0, 0.3, 0.3]\n")
    pyc = PyContactInitializer(
        pycontact_files="gui.txt",
        multiple_files=False,
        in_dir=str(tmp_path),
        pycontact_output_type="GUI",
        remove_false_interactions=False,
    )
    assert list(pyc.prepared_df.columns) == ["12Ala 45Glu Hbond bb-bb", "3Lys 50Asp Saltbr bb-bb"]
    assert pyc.prepared_df["12Ala 45Glu Hbond bb-bb"].tolist() == [1.0, 0.0, 0.5]


def test_modify_column_residue_numbers(block_b):
    df = pd.DataFrame({"1Glu 4Lys Saltbr sc-sc": [1.0], "10Ala 20Leu Hbond": [2.0]})
    shifted = modify_column_residue_numbers(df, constant_to_add=5)
    assert list(shifted.columns) == ["6Glu 9Lys Saltbr sc-sc", "15Ala 25Leu Hbond"]
    # Original is untouched.
    assert list(df.columns) == ["1Glu 4Lys Saltbr sc-sc", "10Ala 20Leu Hbond"]
    assert list(modify_column_residue_numbers(shifted, constant_to_add=-5).columns) == list(df.columns)
