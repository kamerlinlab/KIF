"""
Below script will:
(1) Run PyContact analysis on your trajectory.
(2) Output 2 .csv files:
    1) Per frame interactions scores for each contact.
    2) Summary stats for each interaction.

This script deals with the issue that for the current version of PyContact you...
need to save the results to a session file and then open them up with the GUI...
in order to save the results. (The built-in method: writeContactDataToFile()....
to save a results file automatically does not seem to work.)

Further, with the below approach you can measure the relative contributions...
from the side chain as compared to the main chain for each interaction...
(this info is lost when you export using the PyContact GUI).
"""
import numpy as np
import pandas as pd
from PyContact.core.Scripting import PyContactJob, JobConfig


def assign_contact_type(res1_contrib, res2_contrib):
    """
    Simple logic to assign whether the backbone (bb) or sidechain (sc)
    of each residue is mainly responsible for the interaction.
    Takes as input two dictionaries and returns a string.
    """
    if (res1_contrib["res1_bb"] == 0.0) and (res2_contrib["res2_bb"] == 0.0):
        return "sc-sc"
    if (res1_contrib["res1_sc"] == 0.0) and (res2_contrib["res2_sc"] == 0.0):
        return "bb-bb"

    # if mix of sc and bb contributions.
    if res1_contrib["res1_bb"] >= res1_contrib["res1_sc"]:
        part1 = "bb"
    else:
        part1 = "sc"
    if res2_contrib["res2_bb"] >= res2_contrib["res2_sc"]:
        part2 = "bb"
    else:
        part2 = "sc"
    return part1 + "-" + part2


def main():
    """Main function, you'll want to edit the PyContactJob section for sure!"""
    # define input files and parameters
    job = PyContactJob(
        "[your_topology_file_path]",
        "[your_trajectory_file_path]",
        "[your_protein_name]",
        JobConfig(
            5.0, 2.5, 120,
            [0, 0, 1, 1, 0],  # if you change me you'll need to change main()
            [0, 0, 1, 1, 0],  # if you change me you'll need to change main()
            "protein", "self")
    )
    # Run job on 4 cores.
    job.runJob(4)

    # Comment me in if you want to write a session file to visulise results in the GUI.
    # job.writeSessionToFile()

    # For mapping the determine_ctype() function output to a specific contact type.
    contact_type_map = {
        0: "Saltbr",
        1: "Hydrophobic",
        2: "Hbond",
        3: "Other"
    }

    # Iterate through each contact and store/determine all desired terms:
    contact_names = []
    per_frame_scores = []
    avg_scores = []
    occupancies = []
    numb_interactions = len(job.analyzer.finalAccumulatedContacts)

    for idx in range(0, numb_interactions):
        res1_contrib, res2_contrib = {}, {}

        # residue info.
        res1_numb = job.analyzer.finalAccumulatedContacts[idx].key1[2]
        res1_name = (
            job.analyzer.finalAccumulatedContacts[idx].key1[3]).capitalize()
        res2_numb = job.analyzer.finalAccumulatedContacts[idx].key2[2]
        res2_name = (
            job.analyzer.finalAccumulatedContacts[idx].key2[3]).capitalize()

        # Is it a hbond? vdw? etc...
        interaction_type = contact_type_map[
            job.analyzer.finalAccumulatedContacts[idx].determine_ctype()
        ]

        # Determine whether key contributor is the backbone (bb) or sidechain (sc) for each residue.
        res1_contrib["res1_bb"] = float(
            job.analyzer.finalAccumulatedContacts[idx].bb1)
        res1_contrib["res1_sc"] = float(
            job.analyzer.finalAccumulatedContacts[idx].sc1)
        res2_contrib["res2_bb"] = float(
            job.analyzer.finalAccumulatedContacts[idx].bb2)
        res2_contrib["res2_sc"] = float(
            job.analyzer.finalAccumulatedContacts[idx].sc2)
        contact_type = assign_contact_type(res1_contrib, res2_contrib)

        # build a residue name using all the above info.
        contact_names.append(
            res1_numb + res1_name + " " +
            res2_numb + res2_name + " " +
            interaction_type + " " +
            contact_type
        )

        # Get per frame scores alongside some summary stats.
        scores = job.analyzer.finalAccumulatedContacts[idx].scoreArray
        per_frame_scores.append(np.round(scores, 5))

        avg_score = np.round(np.average(scores), 5)
        avg_scores.append(avg_score)

        nonzero_frames = np.count_nonzero(scores)
        tot_frames = len(scores)
        occupancy = np.round((nonzero_frames/tot_frames) * 100, 1)
        occupancies.append(occupancy)

    # df storing per frame interaction scores.
    df_per_frame = pd.DataFrame.from_records(per_frame_scores).T
    df_per_frame.columns = contact_names
    df_per_frame.to_csv("PyContact_Per_Frame_Interactions.csv", index=False)

    # df of average scores and %occupancies.
    df_summary = pd.DataFrame()
    df_summary["Interaction"] = contact_names
    df_summary["Avg. Strength"] = avg_scores
    df_summary["Occupancy (%)"] = occupancies
    df_summary.to_csv("PyContact_Summary_Stats.csv", index=False)


if __name__ == "__main__":
    main()
