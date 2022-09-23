# Example of how to prepare and run a WISP calculation with Bio3D, as well as write out the necessary output files
# so you can make a PyMOL visualisation of your WISP results with the script titled: gen_pymol_visuals.py

# load libraries required.
library(bio3d)
library(igraph)

# read X-ray structure in.
mypdbfile <- "WISP_Inputs/WT_PTP1B_CSP_Bound_Closed_All_Heavy.pdb"
pdb <- read.pdb(mypdbfile)

# read in correlation matrix and heavy atom contact map (6 Å cut-off used) already made.
cij <- as.matrix(read.table("WISP_Inputs/per_res_matrix.csv", sep=" "))
pdb_cm6 <- as.matrix(read.table("WISP_Inputs/heavy_atom_contact_map_MultiPDB.csv", sep=" "))

# network analysis.
net <- cna(cij, cutoff.cij = 0.25)

# Filter correlation matrix with the pdb generated contact map
net.cut <- cna(cij, cm = pdb_cm6)


############# E200 site to Asp181. #############
# This pdb is missing the first residue, hence numbering of the source and sink is 199 and 180 respectively.
PathGlu200Site <- cnapath(net.cut, from=199, to=180, k=300)
print(PathGlu200Site)

# Save Summary WISP results
sink("WISP_Results/PathGlu200Site_Summary.txt")
print(PathGlu200Site)
sink()

# Save all path lengths and paths used.
cat(PathGlu200Site$dist, "\n", file="WISP_Results/PathGlu200Site_All_Path_Lengths.txt")

sink("WISP_Results/PathGlu200Site_All_Paths.txt")
print(PathGlu200Site$path)
sink()

# Write a VMD session file.
vmd.cnapath(PathGlu200Site, pdb=pdb, spline=TRUE, col='cyan', type = "session", out.prefix = "WISP_Results/PathGlu200Site")


