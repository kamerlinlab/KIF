"""
Example approaches to identify all the non-covalent interactions present in your simulations.

We just need to call the function contact_identification() which has 6 parameters:
parm_file: str
    Topology file of your protein. All MDAnalysis allowed topologies can be used.
    Please do not use a PDB file for this, use something with charge information.
    This is important for the hydrogen bonding part of the calculation to work.

traj_file: str
    Trajectory file. All MDAnalysis allowed trajectory file types can be used.

out_file: str
    Path to write the csv output file too.

first_res: Optional[int]
    First residue to analyse (if you want to break the analysis into blocks).
    If not provided, the first protein residue in the trajectory will be used.

last_res: Optional[int]
    Last residue to analyse (if you want to break the analysis into blocks).
    If not provided, the last protein residue in the trajectory will be used.

report_timings: bool = True
    Choose whether to print to the console how long the job took to run.
    Optional, default is True.
"""
from MDAnalysis.tests.datafiles import PSF, DCD  # test trajectory
from key_interactions_finder import contact_identification

# Here we will use a test trajectory from MDAnalysis to showcase the function.
# In your case you would just replace these two items with the file paths to your simulations.
# A recommendation is to not use more than 10,000 frames for this analysis, you may need much less.

# Version 1. - All residues in the trajectory will be analysed.
# First calculation took 26 seconds to run on my laptop.
contact_identification.calculate_contacts(
    parm_file=PSF,
    traj_file=DCD,
    out_file="contacts_all_res.csv",
    report_timings=True,  # optional
)


# Version 2. - Determine the interaction network for a subset of residues.
# This calculation took 16 seconds to run on my laptop.
# In prior versions of KIF this approach was recommended for large systems.
# However, due to a rewrite of this code, it is no longer recommended
# as it will take longer, overall to run.
# I.e., run the analysis as in version1.
contact_identification.calculate_contacts(
    parm_file=PSF,
    traj_file=DCD,
    out_file="contacts_res_1_to_50.csv",
    first_res=1,  # optional
    last_res=50,  # optional
    report_timings=True,  # optional
)
