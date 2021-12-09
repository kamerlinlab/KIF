"""
Random helper functions.
"""
import os
import pandas as pd


def _prep_out_dir(out_dir) -> str:
    """
    Makes the folder if it doesn't exist and appends a '/' if not present at end of name.

    Parameters
    ----------
    out_dir : str
        Name of output directory user defines.

    Returns
    ----------
    str
        Corrected name of output directory to meet standardization criteria.

    """
    if out_dir != "":
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    if out_dir[-1] != "/":
        out_dir += "/"

    return out_dir


def _filter_features_by_strings(dataset: pd.core.frame.DataFrame,
                                strings_to_preserve: list) -> pd.core.frame.DataFrame:
    """
    Filter PyContact features to only include those that match one of the strings
    in the list provided. (Helper function.)

    Parameters
    ----------
    dataset : pd.core.frame.DataFrame
        Input dataset to filter.

    strings_to_preserve : list
        List of search terms to preserve.

    Returns
    ----------
    pd.core.frame.DataFrame
        Dataframe with features filtered to only include those the user wants.
    """
    # Build the search term
    str_rep = "Classes|"  # preserves classes if already there.
    for list_item in strings_to_preserve:
        str_rep += list_item + "|"

    # must remove the final "|", otherwise it will keep all columns.
    str_rep = str_rep[:-1]

    # Filter.
    filtered_dataset = dataset.loc[:, dataset.columns.str.contains(str_rep)]

    return filtered_dataset
