# key-interactions-finder
 A python package that enables users to easily analyse non-covalent interaction datasets from molecular dynamics simulations.   
 
 **In short, this package allows you to:**
 1. Identify important non-covalent interactions and residues that modulate a descriptor of interest to you and your system. This can be done with either:
     * Supervised machine learning (both classification and regression are supported). (There is also limited support for unsuperivsed learning. 
     
     * Statistical analysis to identify how different/similar each non-covalent interaction is against a target variable. For binary classification, the metrics available are the mutual information and the Jensen-Shannon distance. For a contionous variable, the metrics available are the linear correlation and the mutual information. 

Furthermore, the data generated above can be readily visualised on your protein structure with [PyMOL](https://pymol.org/2/) 

 2. Generate per residue correlation and distance matrices that can be easily applied to the many graph theory methods available in order to study protein interaction networks and allostery.   
 

 
## Description/How it Works

Please note that a full description of how this package works, a more complete description of a generic workflow, and several example applications to different biomolecular systems can be found in our preprint: TODO. 

#### The approximate workflow for this package is as follows: 
1. Run MD simulations on your system(s) of interest with whichever MD engine you want. 
2. (Both can be done simultanesouly)
    1. Analyse the trajectory frames with [PyContact](https://github.com/maxscheurer/pycontact) (NOTE: this program is not made by us), to determine all the non-covalent contacts in your trajectory. [PyContact](https://github.com/maxscheurer/pycontact)  is a well established program and is MD engine agnostic.
    2. (Optional) Calculate the value of a target variable for each frame you analyse with [PyContact](https://github.com/maxscheurer/pycontact)  using whatever approach you see fit. 
3. Load your non-covalent interactions data and optionally generated target data into our program and perform some combination of machine learning, statistical analysis or network analysis
4. Output the results for analysis. This includes visulisation scripts which are compatable with [PyMOL](https://pymol.org/2/) so you can see the results on a 3D protein structure. 


## Dependancies and Install 
Python3 
Python 3.7 - because using dataclasses. 
TODO 


## Getting Started and Tutorials
This repostory contains several juputer notebooks that showcase the major things you can do with this package. 

It would also be a good idea to check out our pre-print (TODO - link) if you have not yet as the pre-print presents and interprets the results generated from these tutorials.  

#### Tutorials Available
(All tutorials include the setup and post-processing steps used on each system.)

1. **[Title of notebook wih link]**  - Binary classification machine learning on simulations of the enzyme PTP1B.  

2. **[Title of notebook wih link]** - Binary classification statistical analysis on simulations of the PDZ3 domain. 

3. **[Title of notebook wih link]** - Regression machine learning on simulations of a Retro-aldol lase enzyme 

4. **[Title of notebook wih link]** - Regression statistical analysis  on simulations of a Retro-aldol lase enzyme 

5. **[Title of notebook wih link - should be a folder this one]** - Preperation of PTP1B inputs required for graph theory based calculations. 
(Included in this tutorial is the R script used to perform WISP and a python script I used to convert the VMD comptabile WISP output to a PyMOL style output.)



## Choosing an Appropriate Target variable.  

To perform either the machine learning or statistical analysis methods available in this package you will most likely want to calculate a target variable to go alongside the features (non-covalent interactions generated with [PyContact](https://github.com/maxscheurer/pycontact) ). 

Below are some examples of what could work for you. Of course, this is use case specific so feel free to select what makes sense for your problem.

**Enzyme catalysis** - A per frame reacting atom distance for regression or define a max reacting atom distance cut-off to classify each frame as either "active" or "inactive". 

**Conformational Changes** - Calculate the RMSD for each frame in your simulation to each conformational state. In this case of 2 different states, classify each frame as either "State 1", "State 2" or "neither" based on your RMSD metrics. 

**For example:**
* if RMSD to "State 1" <= 1.5 Å and an RMSD to "State 2" >= 1.5 Å --> assign frame as "State 1".
* if RMSD to "State 1" => 1.5 Å and an RMSD to "State 2" <= 1.5 Å --> assign frame as "State 2".
* else --> assign frame as "neither".
You can also consider dropping the frames with state "neither" from your analysis to make the calculation cleaner (i.e., turn it into binary classification).
This is the approach we took for the enzyme PTP1B, which you can find described in our preprint. 

 

## Citing this work
If you make use of this package please cite our preprint: 

When accepted/published the above link will be updated. 

### TODOs ### 
Add a TOC like figure demonstrating all the things one can do... 


## Issues/Questions/Contributions
Please feel free to open an issue or submit a pull request as necessary. 

If you have a specific feature request feel free to open an issue. 
