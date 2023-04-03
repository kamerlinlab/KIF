"""
This helper script performs contact analysis on a single trajectory.

After performing this you'll be able to use the outputted file CSV with KIF.

TODO - Add additional documentation.
TODO - Add arg parser
TODO - Add optional cut-off on hbonds as to if counted as a hbond.
"""
from typing import Tuple, List

import pandas as pd
import numpy as np

import MDAnalysis
from MDAnalysis import Universe
from MDAnalysis.tests.datafiles import PSF, DCD, GRO, XTC  # test trajectory
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.analysis import contacts


# Adjustable parameters, please edit.
PARM_FILE = PSF  # r"../Trajectories/Stripped.PDZ_Apo.prmtop"
TRAJ_FILE = DCD  # r"../Trajectories/PDZ_Bound_plus_Apo_Pycontact_Ready_10frames.nc"
OUT_FILE = "Contact_Scores.csv"
FIRST_RES = 55
LAST_RES = 60
REPORT_TIMINGS = True


# Non adjustable parameters - do not need to be touched.
POSITIVE_SB_RESIDUES = ("LYS", "ARG")
NEGATIVE_SB_RESIDUES = ("GLU", "ASP")
HYDROPHOBIC_RESIDUES = ("ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "Cys")


def atom_num_to_res_info(atom_num: int,
                         universe: MDAnalysis.core.universe.Universe) -> Tuple[str, int]:
    """
    From an MDAnalysis atom number and universe, obtain the residue number
    and residue name.

    Parameters
    ----------
    atom_num: int
        Atom id to get residue info from.

    universe: MDAnalysis.core.universe.Universe
        mdanalysis universe object

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


def determine_hbond_pairs(universe: MDAnalysis.core.universe.Universe) -> List[tuple]:
    """
    Run hydrogen bonding analysis on the trajectory to figure out
    which interacting residue pairs contain hydrogen bonds.

    TODO - could add some kind of cut-off on %observations to stop outliers?
    cut-off would need to be an optional param.

    Parameters
    ----------
    universe: MDAnalysis.core.universe.Universe
        mdanalysis universe object

    Returns
    -------
    List[tuple[int]]
        List of hydrogen bonding pairs,
        tuples are of size 2, residue 1, residue 2.
    """
    hbonds = HBA(universe=universe)
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

        donor_resid = atom_num_to_res_info(
            atom_num=donor_atom, universe=universe)[1]
        acceptor_resid = atom_num_to_res_info(
            atom_num=acceptor_atom, universe=universe)[1]

        if (donor_resid, acceptor_resid) not in hbond_pairs:
            hbond_pairs.append((donor_resid, acceptor_resid))

    return hbond_pairs


def determine_interaction_type(res1_id: int,
                               res2_id: int,
                               hbond_pairs: List[tuple],
                               universe: MDAnalysis.core.universe.Universe) -> str:
    """
    Determine the interaction type for a residue pair. Options are:
    hydrogen bond, salt bridge, hydrophobic and vdW's interaction.

    Parameters
    ----------
    res1_id:int
        Number of first residue

    res2_id:int
        Number of second residue

    hbond_pairs: List[tuple]
        list of residue pairs that form hydrogen bonds to one another.

    universe: MDAnalysis.core.universe.Universe
        mdanalysis universe object

    Returns
    -------
    str
        The determined interaction type for the residue pair.
    """
    res1_name = universe.residues[res1_id - 1].resname  # -1 as 0 indexed...
    res2_name = universe.residues[res2_id - 1].resname  # -1 as 0 indexed...

    # salt bridge recongition.
    if (res1_name in POSITIVE_SB_RESIDUES) and (res2_name in NEGATIVE_SB_RESIDUES):
        return "Salt Bridge"
    if (res1_name in NEGATIVE_SB_RESIDUES) and (res2_name in POSITIVE_SB_RESIDUES):
        return "Salt Bridge"

    # hydrogen bond recognition.
    if ((res1_id, res2_id) in hbond_pairs) or ((res2_id, res1_id) in hbond_pairs):
        return "Hydrogen Bond"

    # Hydrophobic recognition.
    if (res1_name in HYDROPHOBIC_RESIDUES) and (res2_name in HYDROPHOBIC_RESIDUES):
        return "Hydrophobic"

    # if not others, then VdW contact.
    return "VdW"


def score_residue_contact(res_res_dists: np.ndarray, dist_cut: float = 6.0) -> float:
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


def main():
    """Perform contact analysis. Saves a .csv file with the results. """
    universe = Universe(PARM_FILE, TRAJ_FILE)
    hbond_pairs = determine_hbond_pairs(universe=universe)

    all_contact_scores = {}
    for res1 in range(FIRST_RES, LAST_RES + 1):
        res1_sele = "not name H* and resid " + str(res1)
        res1_atoms = universe.select_atoms(res1_sele)

        # symmetrical matrix along diagonal, hence loop style.
        for res2 in range(res1, len(universe.residues) + 1):

            res_delta = abs(res1 - res2)
            if res_delta <= 2:  # neighbour residues should have 0 score
                continue

            res2_sele = "not name H* and resid " + str(res2)
            res2_atoms = universe.select_atoms(res2_sele)

            contact_scores = []
            for timestep in universe.trajectory:
                res_res_dists = contacts.distance_array(
                    res1_atoms.positions, res2_atoms.positions)
                contact_score = score_residue_contact(
                    res_res_dists=res_res_dists)
                contact_scores.append(contact_score)

            # save contact only if non-negligible interaction present
            avg_contact_score = sum(contact_scores) / len(universe.trajectory)
            if avg_contact_score > 0.1:

                # -1 as 0 indexed...
                res1_name = universe.residues[res1 - 1].resname.capitalize()
                # -1 as 0 indexed...
                res2_name = universe.residues[res2 - 1].resname.capitalize()

                interaction_type = determine_interaction_type(
                    res1_id=res1, res2_id=res2,
                    hbond_pairs=hbond_pairs, universe=universe
                )

                contact_label = str(res1) + res1_name + " " + \
                    str(res2) + res2_name + " " + interaction_type
                all_contact_scores.update({contact_label: contact_scores})

    df_contact_scores = pd.DataFrame(all_contact_scores)
    df_contact_scores.to_csv(OUT_FILE, index=False)


if __name__ == "__main__":
    if REPORT_TIMINGS:
        import time
        from datetime import timedelta
        start_time = time.monotonic()

    main()

    if REPORT_TIMINGS:
        end_time = time.monotonic()
        delta_time = timedelta(seconds=end_time - start_time)
        print(f"Time taken: {delta_time}")
