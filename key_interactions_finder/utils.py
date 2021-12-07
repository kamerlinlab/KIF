"""
Random helper functions.
"""
import os
import pandas as pd


def _prep_out_dir(out_dir) -> str:
    """Makes the folder if doesn't exist and appends a '/' if not present at end of name."""
    if out_dir != "":
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    if out_dir[-1] != "/":
        out_dir += "/"

    return out_dir


def _filter_features(
        dataset: pd.core.frame.DataFrame, interaction_types_included: list) -> pd.core.frame.DataFrame:
    """
    Filter PyContact features to include only those a certain interaction type
    in the network analysis, based on user selection.

    Parameters
    ----------
    dataset : pd.core.frame.DataFrame
        Input dataset to filter.

    interaction_types_included : list
        List of interaction types to include in the returned dataframe.

    Returns
    ----------
    pd.core.frame.DataFrame
        Dataframe with features filtered to only include those the user wants.

    """
    # Build the search term
    str_rep = "Classes|"  # preserves classes if already there.
    for list_item in interaction_types_included:
        str_rep += list_item + "|"

    # must remove the final "|", otherwise it will keep all columns.
    str_rep = str_rep[:-1]

    # Filter.
    filtered_dataset = dataset.loc[:, dataset.columns.str.contains(str_rep)]

    return filtered_dataset
