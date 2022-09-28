"""
Reformats and (optionally) merges PyContact datafiles generated using
either the (1) custom python script provided with this package or
(2) the PyContact GUI.

Special Notes
1. A pycontact job run with overlapping residue selections can obtain many false interactions, e.g.:
    - A vdw interaction with a residue to itself.
    - Duplicate interactions with the only difference being residue ordering swapped.

If a job is therefore run in the above way, the 'remove_false_interactions' parameter must be set
to True (the default) when the class is initialised in order to remove these false interactions.
There is no harm if a PyContact job is not run in this way
but 'remove_false_interactions' is set to True anyway.

2. Depending on the md trajectory format used, the residue numbers assigned by PyContact
can be off by 1 residue number. If this happens to you, you can use the
function "modify_column_residue_numbers" to edit/renumber all the features in your dataframe.
"""
from pathlib import Path
import re
from typing import Union, Optional
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class PyContactInitializer():
    """
    Handles PyContact output files generated by the custom Python script or the
    PyContact GUI.
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
        Directory where the input files are stored.
        Default is "".

    pycontact_output_type : str
        Define whether the PyContact output files were made using the custom script provided
        by us (recommended) or using the PyContact GUI.
        Options are: "custom_script" or "GUI".
        Default is "custom_script"

    prepared_df : pd.DataFrame
        Final processed dataframe of all PyContact features.
        Generated once class is initialized.
    """
    # Generated when instantiated.
    pycontact_files: Union[str, list]
    multiple_files: bool
    merge_files_method: Optional[str] = None
    remove_false_interactions: bool = True
    in_dir: str = ""
    pycontact_output_type: str = "custom_script"

    # Generated later.
    prepared_df: pd.DataFrame = field(init=False)

    # This is called at the end of the dataclass's initialization procedure.
    def __post_init__(self):
        """Processes the provided PyContact files."""
        if (self.in_dir != "") and (self.in_dir[-1] != "/"):
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
        """
        Load a single PyContact dataset into a dataframe.

        Dataset can either be generated by the custom script provided with this package
        or be generated by the PyContact GUI.

        Parameters
        ----------
        input_file: str
            Input file name.

        Returns
        ----------
        pd.DataFrame
            A df with each interaction found by PyContact a row in the dataframe.
        """
        file_in_path = Path(self.in_dir, input_file)

        if self.pycontact_output_type == "custom_script":
            return pd.read_csv(file_in_path)

        if self.pycontact_output_type == "GUI":
            return self._process_gui_file(pycontact_gui_file=file_in_path)

        error_message = (
            "You must choose between either 'custom_script' or 'GUI' " +
            "for the parameter pycontact_output_type."
        )
        raise ValueError(error_message)

    def _process_gui_file(self, pycontact_gui_file: str):
        """
        Process a PyContact file generated by the PyContact GUI.

        Parameters
        ----------
        pycontact_gui_file: str
            complete file path to the GUI generated file.

        Returns
        ----------
        pd.DataFrame
            A dataframe with all interactions in the gui file and interactions names
            reformatted to match the format used throughout this programm.
        """
        with open(pycontact_gui_file, 'r') as file:
            filedata = file.read()

        filedata = filedata.replace("[", ",").replace("]", ",")
        # Removing any double or more spaces and tabs etc...
        filedata = re.sub(' +', ' ', filedata)

        # Standardize formatting.
        filedata = filedata.replace("hbond", "Hbond,").replace(
            "hydrophobic", "Hydrophobic,").replace("other", "Other,").replace("saltbr", "Saltbr,")
        file_data_list = (filedata.split("\n"))[1:]  # skip top row of headers.

        all_features = {}
        for line in file_data_list:
            feature = line.split(",")[0]

            # [1], contains the unwanted averages, stdevs etc...
            scores_str = line.split(",")[2:]
            scores_flt = []
            for score in scores_str:
                try:
                    scores_flt.append(round(float(score), 5))
                except ValueError:
                    # Handles ValueError: could not convert string to float: ''
                    # This occurs if user also requested to output hbond occupancy data.
                    # These (unwanted) occpancies come after the ValueError,
                    # so break removes them if they are present in the file.
                    break

            if scores_flt:  # otherwise get an empty feature at the end.
                feature_cleaned = self._clean_gui_feature_name(
                    feature_name=feature)
                all_features.update({feature_cleaned: scores_flt})

        return pd.DataFrame.from_dict(all_features)

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

        Most duplicates are identical but have the residue order swapped.
        i.e. resX + resY + [other columns] vs. resY + resX + [other columns]
        This is true if the interaction type is sc-sc or bb-bb.

        Some duplicates don't just have the residue order swapped but as they are of type:
        sc-bb or bb-sc the test for a duplicate needs to be different.
        Here, "contact_parts[5]" will NOT be identical if they are duplicates because
        the residue ordering is swapped in the duplicates!

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
            if contact_parts[5] in ("sc-sc", "bb-bb"):
                if (
                        (contact_parts[2] == saved_contact[0]) and
                        (contact_parts[0] == saved_contact[2]) and
                        (contact_parts[4] == saved_contact[4]) and
                        (contact_parts[5] == saved_contact[5])
                ):
                    duplicate = True
                    break

            else:
                if (
                        (contact_parts[2] == saved_contact[0]) and
                        (contact_parts[0] == saved_contact[2]) and
                        (contact_parts[4] == saved_contact[4]) and
                        (contact_parts[5] != saved_contact[5])
                ):
                    duplicate = True
                    break

        return duplicate

    def _rm_false_interactions(self, full_df) -> pd.DataFrame:
        """
        Remove non-meaningful (too close to one another) or duplicate contacts/features.
        Required if in the PyContact job run a user sets the second residue selection
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
            # Remove Hbond and Saltbr if <= 2 residues of each other.
            else:
                if residue_gap <= 2:
                    contacts_to_del.append(idx)
                else:
                    contacts_to_keep.append(contact_parts)

            if self._interaction_is_duplicate(contact_parts, contacts_to_keep):
                contacts_to_del.append(idx)

        prepared_df = full_df.drop(
            full_df.columns[contacts_to_del], axis=1)
        return prepared_df

    @staticmethod
    def _clean_gui_feature_name(feature_name: str) -> str:
        """
        Reformats a feature name from how it is labelled in the GUI output of PyContact
        to the standardized way that is expected throughout this program.

        Note that with the GUI, it is not possible to save whether the interaction
        is between the sidechain or backbone for both residues,
        so for consistency with the rest of program, all features
        are assinged as "bb-bb" (backbone-backbone) interactions.

        Parameters
        ----------
        feature_name: str
            GUI formatted feature to reformat.

        Returns
        ----------
        str
            Reformatted feature name.
        """
        res1 = feature_name.split('-')[0]
        res1_numb = re.findall(r'\d+', res1)[0]
        res1_name = (re.findall(r'[A-Z|a-z]+', res1)[0]).capitalize()

        res2_plus_info = feature_name.split('-')[1]
        res2_numb = re.findall(r'\d+', res2_plus_info)[0]
        res2_name = (re.findall(r'[A-Z|a-z]+', res2_plus_info)[0]).capitalize()

        # info is interaction type (e.g. Hbond, Vdw etc...)
        info = (re.findall(r'[A-Z|a-z]+', res2_plus_info)[1]).capitalize()

        return res1_numb + res1_name + " " + res2_numb + res2_name + " " + info + " bb-bb"


def modify_column_residue_numbers(dataset: pd.DataFrame, constant_to_add: int = 1) -> pd.DataFrame:
    """
    Take a dataframe of PyContact generated features and add a constant value to all residue
    numbers in each feature. This function exists as in some cases mdanalysis (used by PyContact)
    may renumber residue numbers to start from 0 as opposed to most MD engines which
    start from 1.

    This function will NOT update the class attribute ".prepared_df".
    Instead it returns a new dataframe with the updated residue numbers.

    Parameters
    ----------
    dataset: pd.DataFrame
        Input dataframe with Pycontact features you wish to modify.

    constant_to_add: int
        Value of the constant you want to add from each residue.
        You can use negative numbers to subtract.
        Default = 1

    Returns
    ----------
    pd.DataFrame
        Dataframe with residue numbers updated accordingly.
    """
    updated_names = []
    all_ori_names = list(dataset)
    for column_name in all_ori_names:
        res_split = re.split("(\d+)", column_name)

        res1_numb = int(res_split[1])
        res1_name = res_split[2]
        res2_numb = int(res_split[3])
        remainder = res_split[4]

        res1_numb += constant_to_add
        res2_numb += constant_to_add

        updated_name = str(res1_numb) + res1_name + \
            str(res2_numb) + remainder
        updated_names.append(updated_name)

    # don't want to overwrite class dataframe in case user runs twice or by accident etc...
    new_dataset = dataset.copy(deep=True)
    new_dataset.columns = updated_names
    return new_dataset
