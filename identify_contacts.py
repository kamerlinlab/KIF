from key_interactions_finder import contact_identification
from MDAnalysis.tests.datafiles import PSF, DCD, GRO, XTC  # test trajectory

contact_identification.calculate_contacts(
    parm_file=PSF,
    traj_file=DCD,
    out_file="test_contacts.csv",
    first_res=55,
    last_res=60,
    report_timings=True
)
