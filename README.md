# key-interactions-finder
 A python package that enables users to easily analyse non-covalent interaction datasets from molecular dynamics simulations.   
 
 **In short, this package allows you to:**
 1. Identify important non-covalent interactions and residues that modulate a descriptor of interest to you and your system. This can be done with either:
     * Supervised machine learning (both classification and regression are supported). (There is limited support for unsuperivsed learning, see section TODO below). 
     * Statistical analysis to identify how different/similar each non-covalent interaction is in 2 states. The metrics available are the mutual information and the Jensen-Shannon distance.  

 2. Generate per residue correlation and distance matrices that can be easily applied to graph theory methods in order to study protein contact networks/allostery.   
 
 Further, all data generated can be readily visualised on your protein structure with [PyMOL](https://pymol.org/2/)



## Description/ How it Works

This package is designed to 

Pycontact and is MD engine agnositc. 



## Dependancies and Install 
Python3
Fill in. 
Python 3.7 - because using dataclasses. 

## Getting Started and Tutorials
This repostory contains several juputer notebooks that showcase the major things you can do with this package. 

It would also be a good idea to check out our pre-print (TODO, add) on the package 

Before beginning you'll need to 


## Choosing an Appropriate Target variable.  

To perform either the machine learning or statistical analysis methods available in this package you will most likely need to calculate a target varaible to go alongside the input features (non-covalent interactions generated with PyContact). 

Most of the analysis possible with this package uses some kind of target feature to determine 

Some examples:

**Enzyme catalysis** - A per frame reacting atom distance for regression or define a max reacting atom distance cut-off to classify the frame as either "active" or "inactive". 

**Conformational Changes** - Calculate the RMSD for each frame in your simulation to each conformational state. In this case of 2 different states, classify each frame as either "State 1", "State 2" or "neither" based on your RMSD metrics. 

**For example:**
* if RMSD to "State 1" <= 1.5 Å and an RMSD to "State 2" >= 1.5 Å --> assign frame as "State 1".
* if RMSD to "State 1" => 1.5 Å and an RMSD to "State 2" <= 1.5 Å --> assign frame as "State 2".
* else --> assign frame as "neither".

You can also consider dropping the frames with state "neither" from your analysis to make the calculation cleaner. 

Of course, the above are suggestions, feel free select what makes sense for your problem. 

## Citing this work
If you make use of this package please cite the following manuscript: 

When accepted/published the above link will be updated. 

### TODO ### 
Add a TOC like figure demonstrating all the things one can do... 


## Issues/Questions/Contributions
Please feel free to open an issue or submit a pull request as necessary. 

If you have a specific feature request feel free to open an issue. (Of coure, I can't promise I can write/add it)
