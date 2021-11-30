"""
Random helper functions.
"""
import os


def _prep_out_dir(out_dir) -> str:
    """Makes the folder if doesn't exist and appends a '/' if not present at end of name."""
    if out_dir != "":
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    if out_dir[-1] != "/":
        out_dir += "/"

    return out_dir
