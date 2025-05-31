"""
Responsible for identifying all non-covalent interactions in a trajectory.

This bypasses the need for the install of PyContact and reproduces the scoring
function used by PyContact.

Output will be a csv file with each column an interaction pair.
Each column has the following format:
[residue1] [residue2] [interaction type]

Where "residue1" and "residue2" are the names and residue numbers of the pair.
and "interaction type" is one of:
"Hbond" - Hydrogen bond
"Saltbr" - Salt Bridge
"Hydrophobic" - VdW interaction between two hydrophobic residues.
"VdW" - Unspecified VdW interaction.
"""

import time
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from MDAnalysis import Universe
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA

# Amino acid definitions - helps define interaction types.
POSITIVE_SB_RESIDUES = ("LYS", "ARG")
NEGATIVE_SB_RESIDUES = ("GLU", "ASP")
HYDROPHOBIC_RESIDUES = ("ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "Cys")

# Used to prefilter residue pairs.
MAX_CA_DIST = 20
MAX_HEAVY_DIST = 6  # Min heavy atom dist, otherwise contact score = 0.

# From clarification on GitHub, this message can be safely ignored.
# https://github.com/MDAnalysis/mdanalysis/issues/3889
warnings.filterwarnings("ignore", message="DCDReader currently makes independent timesteps")


def calculate_contacts(
    parm_file: str,
    traj_file: str,
    out_file: str,
    first_res: Optional[int] = None,
    last_res: Optional[int] = None,
    report_timings: bool = True,
) -> None:
    """

    Identify all non-covalent interactions present in the simulation and save the output.
    Output has each non-covalent interaction as a column
    each column has the following information: [residue1] [residue2] [interaction type]

    Parameters
    ----------

    parm_file: str
        The file path to your topology file. All MDAnalysis allowed topologies can be used.
        Please do not use a PDB file for this, use something with charge information.
        This is important for the hydrogen bonding part of the calculation to work.

    traj_file: str
        The file path to your trajectory file.
        All MDAnalysis allowed trajectory file types can be used.

    out_file: str
        Path to write the csv output file too.

    first_res: Optional[int]
        First residue to analyse, useful if you want to break the analysis into blocks.
        If not provided, the first residue in the trajectory will be used.

    last_res: Optional[int]
        Last residue to analyse, useful if you want to break the analysis into blocks.
        If not provided, the last residue in the trajectory will be used.

    report_timings: bool = True
        Choose whether to print to the console how long the job took to run.
        Optional, default is True.

    Returns
    -------

    None
        Output written to file. Optional timings printed to the console.
    """
    if report_timings:
        start_time = time.monotonic()

    universe = Universe(parm_file, traj_file)

    if first_res is None:
        first_res = 1

    biggest_protein_res = max(universe.select_atoms("name CA").resids)
    if last_res is None:
        last_res = biggest_protein_res

    if last_res > biggest_protein_res:
        warning_message = f"""
            You stated your last residue was residue number: {last_res}.
            I found the last protein residue to be: {biggest_protein_res}. \n
            This program is primarily designed to analyse protein-protein interactions, but can be used
            on ligands, or even water molecules \n
            If the difference between the {last_res} and {biggest_protein_res} is very large, then check you have not
            included all water molecules in your calculation. Doing so will make this calculation a lot slower. \n

            If the difference is small, you can safely ignore this warning message as it is probably the case you just
            have a ligand or two in your input file.
        """
        warnings.warn(warning_message, stacklevel=2)

        biggest_res = last_res
    else:
        biggest_res = biggest_protein_res

    trajectory_length = len(universe.trajectory)
    trajectory_of_zeros = np.zeros(trajectory_length)

    all_heavy_atoms_sele = "not name H* and resid 1" + "-" + str(biggest_res)
    all_heavy_atoms = universe.select_atoms(all_heavy_atoms_sele)
    residue_names = [names.capitalize() for names in list(universe.residues.resnames)]

    # determine which residue each heavy atom belongs to.
    residue_ranges = {}
    for res_numb in range(1, biggest_res + 1):
        residue_range = np.where(all_heavy_atoms.atoms.resids == res_numb)
        residue_ranges[res_numb] = residue_range

    print("setup complete, analysing contacts now...")
    hbond_pairs = _determine_hbond_pairs(universe=universe)

    # Now go through each frame.
    all_contact_scores = {}
    for idx, _ in enumerate(universe.trajectory):  # each step is new frame.
        # calculate all heavy atom distances for this trajectory
        heavy_atom_dists = distances.distance_array(
            all_heavy_atoms.positions,
            all_heavy_atoms.positions,
        )

        for res1 in range(first_res, last_res + 1):
            res_dists = heavy_atom_dists[residue_ranges[res1]]

            # +3 here as neighbouring residues not interesting.
            for res2 in range(res1 + 3, biggest_res + 1):
                res_res_dists = res_dists[:, residue_ranges[res2]]

                # score would be 0 if true.
                if res_res_dists.min() > MAX_HEAVY_DIST:
                    continue

                contact_score = _score_residue_contact(res_res_dists)
                if (res1, res2) not in all_contact_scores:
                    # create empty array of size trajectory for it...
                    all_contact_scores[(res1, res2)] = trajectory_of_zeros.copy()
                # now update score for this frame.
                all_contact_scores[(res1, res2)][idx] = contact_score

    contact_labels_scores = {}
    for res_pair, contact_scores in all_contact_scores.items():
        # save contact only if non-negligible interaction present
        avg_contact_score = sum(contact_scores) / trajectory_length
        if avg_contact_score < 0.1:
            continue

        res1, res2 = res_pair
        # -1 as 0 indexed...
        res1_name = residue_names[res1 - 1]
        res2_name = residue_names[res2 - 1]

        interaction_type = _determine_interaction_type(
            res1_id=res1,
            res2_id=res2,
            hbond_pairs=hbond_pairs,
            universe=universe,
        )

        contact_label = str(res1) + res1_name + " " + str(res2) + res2_name + " " + interaction_type
        contact_labels_scores.update({contact_label: contact_scores})

    # reorders column names, to be like the old format.
    sorted_dict = dict(
        sorted(
            contact_labels_scores.items(),
            key=lambda item: (int("".join(filter(str.isdigit, item[0].split(":")[0])))),
        )
    )

    df_contact_scores = pd.DataFrame(sorted_dict)
    df_contact_scores.to_csv(out_file, index=False)

    if report_timings:
        end_time = time.monotonic()
        delta_time = timedelta(seconds=end_time - start_time)
        print(f"Time taken: {delta_time}")


# helper functions below.
def _atom_num_to_res_info(atom_num: int, universe: Universe) -> Tuple[str, int]:
    """
    From an MDAnalysis atom number and universe, obtain the residue number
    and residue name.

    Parameters
    ----------
    atom_num: int
        Atom id to get residue info from.

    universe: Universe
        MDAnalysis universe object

    Returns
    -------
    Tuple[str, int]
        string is resname, int is the resid.
    """
    atom_num = int(atom_num)  # in case float passed.
    donor_info = str(universe.atoms[atom_num].residue)
    donor_parts = donor_info.replace(", ", " ").replace(">", "").split(" ")

    res_name, resid = donor_parts[1], int(donor_parts[2])
    return res_name, resid


def _determine_hbond_pairs(universe: Universe) -> List[tuple]:
    """
    Run hydrogen bonding analysis on the trajectory to figure out
    which interacting residue pairs contain hydrogen bonds.

    Parameters
    ----------
    universe: Universe
        MDAnalysis universe object

    Returns
    -------
    List[tuple[int]]
        List of hydrogen bonding pairs,
        tuples are of size 2: residue 1, residue 2.
    """
    hbonds = HBA(universe=universe)
    hbonds.hydrogens_sel = hbonds.guess_hydrogens("protein")
    hbonds.acceptors_sel = hbonds.guess_acceptors("protein")
    hbonds.run()
    # MDAnalysis.exceptions.NoDataError will occur if topology has no charge info.

    # hbond_results is a nested array.
    # outer layer is a hbond found.
    # inner layer is: frame, donor, hydrogen, acceptor index, distance, angle
    hbond_results = hbonds.results.hbonds

    # Make list of all hbond residue pairs.
    hbond_pairs = []
    for observation in hbond_results:
        donor_atom, acceptor_atom = observation[1], observation[3]

        donor_resid = _atom_num_to_res_info(atom_num=donor_atom, universe=universe)[1]
        acceptor_resid = _atom_num_to_res_info(atom_num=acceptor_atom, universe=universe)[1]

        if (donor_resid, acceptor_resid) not in hbond_pairs:
            hbond_pairs.append((donor_resid, acceptor_resid))

    return hbond_pairs


def _determine_interaction_type(
    res1_id: int,
    res2_id: int,
    hbond_pairs: List[tuple],
    universe: Universe,
) -> str:
    """
    Determine the interaction type for a residue pair. Options are:
    hydrogen bond, salt bridge, hydrophobic and VdW's interaction.

    Parameters
    ----------
    res1_id:int
        Number of first residue

    res2_id:int
        Number of second residue

    hbond_pairs: List[tuple]
        list of residue pairs that form hydrogen bonds to one another.

    universe: Universe
        MDAnalysis universe object

    Returns
    -------
    str
        The determined interaction type for the residue pair.
    """
    # -1 as 0 indexed...
    res1_name = universe.residues[res1_id - 1].resname
    res2_name = universe.residues[res2_id - 1].resname

    # salt bridge recongition.
    if (res1_name in POSITIVE_SB_RESIDUES) and (res2_name in NEGATIVE_SB_RESIDUES):
        return "Saltbr"
    if (res1_name in NEGATIVE_SB_RESIDUES) and (res2_name in POSITIVE_SB_RESIDUES):
        return "Saltbr"

    # hydrogen bond recognition.
    if ((res1_id, res2_id) in hbond_pairs) or ((res2_id, res1_id) in hbond_pairs):
        return "Hbond"

    # Hydrophobic recognition.
    if (res1_name in HYDROPHOBIC_RESIDUES) and (res2_name in HYDROPHOBIC_RESIDUES):
        return "Hydrophobic"

    # if not any of the above, then VdW contact.
    return "Other"


def _score_residue_contact(res_res_dists: np.ndarray, dist_cut: float = 6.0) -> float:
    """
    Score the "strength" of a pair of residues based on their atomic distances.
    Same implementation as in pycontact: https://github.com/maxscheurer/pycontact

    Parameters
    ----------
    res_res_dists: np.ndarray[float]
        Distance matrix of size:
        number of atoms in residue 1 x number of atoms in residue 2.

    dist_cut: float=6.0
        Max distance before atom-atom contact not included in scoring.
        Values much larger don't notably affect the score.

    Returns
    -------
    float
        Calculated contact score.
    """
    contact_score = 0
    for dist in res_res_dists.flatten():
        if dist <= dist_cut:
            contact_score += 1.0 / (1.0 + np.exp(5.0 * (dist - 4.0)))

    return round(contact_score, 4)
