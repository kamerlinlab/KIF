"""
Reformats and (optionally) merges PyContact datafiles generated using the custom python script.

Special Note:
A pycontact job run with overlapping residue selections can obtain many false interactions such as:
1. A vdw interaction with a residue to itself.
2. Duplicate interactions with the only difference being residue ordering swapped.

If a job is run in the above way, the 'remove_false_interactions' parameter must be set
to True (default) when the class is initialised in order to remove these false interactions.
No harm if a PyContact job is not run in this way but 'remove_false_interactions' is True anyway.
"""
import re
from typing import Union, Optional
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class PyContactInitializer():
    """
    Handles PyContact output files generated by the custom Python script.
    (Can merge, reformat and clean duplicates/false interactions).

    Attributes
    ----------
    pycontact_files : str or list
        String for a single file or list of strings for many files

    multiple_files: bool
        True or False, do you have multiple files to merge together.

    merge_files_method: Optional[str]
        If you have multiple files, clarify if they should be merged in a
        'vertical' or 'horizontal' manner.
        If 'multiple_files' = True, must be specfied.

    remove_false_interactions: bool
        Whether to run a method to clean false interactions from the input datasets.
        Default is True.

    in_dir : str
        Directory where input files are stored.
        Default is "".

    prepared_df : pd.DataFrame
        Final processed dataframe of all PyContact features.
        Generated dynamically.
    """
    # Generated when instantiated.
    pycontact_files: Union[str, list]
    multiple_files: bool
    merge_files_method: Optional[str]
    remove_false_interactions: bool = True
    in_dir: str = ""

    # Generated later.
    prepared_df: pd.DataFrame = field(init=False)

    # This is called at the end of the dataclass's initialization procedure.
    def __post_init__(self):
        """Processes the provided PyContact files."""
        if self.in_dir[-1] != "/":
            self.in_dir += "/"

        if self.multiple_files:
            individ_dfs = [self._load_pycontact_dataset(
                i) for i in self.pycontact_files]

            if self.merge_files_method == "vertical":
                full_df = self._merge_pycontact_datasets_vertically(
                    individ_dfs)
            elif self.merge_files_method == "horizontal":
                full_df = self._merge_pycontact_datasets_horizontally(
                    individ_dfs)
            else:
                error_message = (
                    "You said you had multiple files but you did not define the " +
                    "'merge_files_method' parameter as either 'vertical' or 'horizontal'."
                )
                raise ValueError(error_message)
        else:
            full_df = self._load_pycontact_dataset(self.pycontact_files)

        if self.remove_false_interactions:
            self.prepared_df = self._rm_false_interactions(full_df)
        else:
            self.prepared_df = full_df

        num_feats = len(self.prepared_df.columns)
        num_obs = len(self.prepared_df)
        print("Your PyContact file(s) have been succefully processed.")
        print(f"You have {num_feats} features and {num_obs} observations.")
        print("The fully processed dataframe is accesible from the '.prepared_df' class attribute.")

    def _load_pycontact_dataset(self, input_file) -> pd.DataFrame:
        """Load a single PyContact dataset."""
        file_in_path = self.in_dir + input_file
        return pd.read_csv(file_in_path)

    @staticmethod
    def _merge_pycontact_datasets_horizontally(individ_dfs) -> pd.DataFrame:
        """
        Function to merge multiple PyContact dataframes horizontally.

        This would be used when a user has analysed different parts of their protein
        with PyContact and wants to put them all together (i.e. each file is from the
        same trajectory/trajectories.)

        Parameters
        ----------
        individ_dfs: list
            List of dataframes to merge.

        Returns
        ----------
        pd.DataFrame
            A complete dataframe with the individual dfs merged.
        """
        df_lengths = [len(df) for df in individ_dfs]

        if len(set(df_lengths)) != 1:
            except_message = (
                "The number of rows in each of your datasets are not identical. " +
                "This is weird because you asked to merge your files horizontally. " +
                "If you are using this approach then your different files should all be from the " +
                "same trajectory just with different contacts measured in each of them. " +
                "If not, you likely want to set the 'merge_files_method' parameter to 'vertical'.")
            raise Exception(except_message)

        merged_df = pd.concat(individ_dfs, axis=1)
        merged_df = merged_df.fillna(0.0)
        return merged_df

    @staticmethod
    def _merge_pycontact_datasets_vertically(individ_dfs) -> pd.DataFrame:
        """
        Function to merge multiple PyContact dfs vertically. This would be used when a user has
        multiple replicas or has broken their trajectories into blocks of separate frames.

        Parameters
        ----------
        individ_dfs: list
            List of dataframes to merge.

        Returns
        ----------
        pd.DataFrame
            Complete df with individual dfs merged in the same order as they were provided.
        """
        merged_df = pd.concat(
            individ_dfs, ignore_index='True', sort='False')
        merged_df = merged_df.fillna(0.0)
        return merged_df

    @staticmethod
    def _interaction_is_duplicate(contact_parts: list, contacts_to_keep: list) -> bool:
        """
        Check if current interaction is a duplicate of an already kept contact.
        Duplicates are identical but have the residue order swapped.
        i.e. resX + resY + [other columns] vs. resY + resX + [other columns]

        Parameters
        ----------
        contact_parts : list
            The feature/contact to test whether it is a duplicate of a pre-existing contact.
            The list items are different parts of the contact.

        contacts_to_keep : list
            A list of all current features that will be kept.
            (Used to determine if new feature is duplicate of these.)

        Returns
        ----------
        bool
            True if contact is a duplicate.
        """
        duplicate = False
        for saved_contact in contacts_to_keep:
            if (
                    (contact_parts[2] == saved_contact[0]) and
                    (contact_parts[0] == saved_contact[2]) and
                    (contact_parts[4] == saved_contact[4]) and
                    (contact_parts[5] == saved_contact[5])
            ):
                duplicate = True
                break

        return duplicate

    def _rm_false_interactions(self, full_df) -> pd.DataFrame:
        """
        Remove non-meaningful (too close to one another) or duplicate contacts/features.
        Required if in the PyContact job run a user sets the second residue selection.
        group to be something other than "self" and the residue selections overlap.
        Doesn't hurt to be run if not anyway.

        Parameters
        ----------
        full_df : pd.DataFrame
            Dataframe of pycontact data (already merged if needed) but not cleaned/proccessed
            yet.

        Returns
        ----------
        pd.DataFrame
            Dataframe with "false interactions" removed.
        """
        contacts_to_del = []
        contacts_to_keep = []
        column_names = full_df.columns
        for idx, contact in enumerate(column_names):
            contact_parts = re.split("(\d+|\s)", contact)
            # remove the list items with empty or single spaces from the above regex.
            for list_index in sorted([0, 3, 4, 7, 9], reverse=True):
                del contact_parts[list_index]

            residue_gap = abs(int(contact_parts[0]) - int(contact_parts[2]))
            # Remove vdw type interactions if <= 3 residues of each other.
            if contact_parts[4] in ["Other", "Hydrophobic"]:
                if residue_gap <= 3:
                    contacts_to_del.append(idx)
                else:
                    contacts_to_keep.append(contact_parts)
            # Remove Hbond and Saltbr if <= 1 residues of each other.
            else:
                if residue_gap <= 1:
                    contacts_to_del.append(idx)
                else:
                    contacts_to_keep.append(contact_parts)

            if self._interaction_is_duplicate(contact_parts, contacts_to_keep):
                contacts_to_del.append(idx)

        prepared_df = full_df.drop(
            full_df.columns[contacts_to_del], axis=1)
        return prepared_df
