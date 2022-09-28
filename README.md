# KIF - Key Interactions Finder
 A python package to identify the key molecular interactions that regulate any conformational change.
 
 ### TODOs ### 
Add a TOC like figure demonstrating all the things one can do... 
 
## In short, this package allows you to:
 - Identify important non-covalent interactions that regulate any conformational you are interested in (as long as you can describe the descriptor and sample the descriptor in your MD simulations). The interactions can be ranked in terms of their importance and used to generate per residue importances as well. 
 - Generate [PyMOL](https://pymol.org/2/) output scripts that enable you to visulaise your results on 3D structures. 
 - Generate per residue correlation and distance matrices that can be easily applied to the many graph theory methods available in order to study protein interaction networks and allostery within your system (no descriptor/target variable required for this). 

Note that how define the descriptor is up to you, and you can use both a continous variable or a categorical variable (some tips on how to decide what to use will be given below). 

**More Detail Please!**
For a more complete description of KIF, please refer to our preprint: TODO. Included in the preprint is a description of some of the generic workflows possible alongside the application of KIF to several different biomolecular systems. 

There are also tutorials available (discussed below). 

## The approximate workflow for this package is as follows: 
1. Run MD simulations on your system(s) of interest with whichever MD engine you want. 
2. (Both can be done simultanesouly)
    1. Analyse the trajectory frames with [PyContact](https://github.com/maxscheurer/pycontact) (NOTE: this program is not made by us), to determine all the non-covalent contacts in your trajectory. [PyContact](https://github.com/maxscheurer/pycontact) is a well established program and is MD engine agnostic.
    2. (Optional) Calculate the value of a target variable for each frame you analyse with [PyContact](https://github.com/maxscheurer/pycontact) using whatever approach you see fit. 
3. Load your non-covalent interactions data and optionally generated target data into KIF and perform some combination of machine learning, statistical analysis or network analysis. 
4. Output the results for analysis. This includes visulisation scripts which are compatable with [PyMOL](https://pymol.org/2/) so you can see the results on a 3D protein structure. 


## Dependancies and Install 
Python 3.7 - because using dataclasses. 
TODO 

## Running PyContact 
Prior to using our package you'll need to have analysed your simulation(s) with [PyContact](https://github.com/maxscheurer/pycontact). For this, [we have a provided a script in this repo to do this](https://github.com/kamerlinlab/key-interactions-finder/blob/main/key_interactions_finder/run_pycontact.py), from which you'll obtain two outputs (one with summary stats - not needed for KIF, and the other with per frame interaction scores). This script helps to standardise the output from PyContact making it easier for KIF to handle the data. If you instead wish to process the data through the PyContact GUI, please refer to tutorial 3 for how to go about this. All other tutorials used datasets generated from the custom script described above. 

For a large number of frames and/or a large system, you will likely need to break up your PyContact calculation into blocks (to prevent running out of memory). We did this by making a single trajectory (of all frames we want to analyse) and submitting several (between 10-20) PyContact jobs on different residue ranges. Merging these results files back together again can be done with KIF - see the tutorials (1, 2 or 4).  


## Choosing an Appropriate Target variable.  
To perform either the machine learning or statistical analysis methods available in this package you will want to calculate a target variable to go alongside the features (non-covalent interactions generated with [PyContact](https://github.com/maxscheurer/pycontact) ). This target variable should as cleanly as possible describe your process of interest. 

Below are some examples of what could work for you. Of course, this is use case specific so feel free to select what makes sense for your problem.

**Enzyme catalysis** - A per frame reacting atom distance for regression or define a max reacting atom distance cut-off to classify each frame as either "active" or "inactive". 

**Conformational Changes** - Calculate the RMSD for each frame in your simulation to each conformational state. In this case of 2 different states, classify each frame as either "State 1", "State 2" or "neither" based on your RMSD metrics.

**For example:**
* if RMSD to "State 1" <= 1.5 Å and an RMSD to "State 2" >= 1.5 Å --> assign frame as "State 1".
* if RMSD to "State 1" => 1.5 Å and an RMSD to "State 2" <= 1.5 Å --> assign frame as "State 2".
* else --> assign frame as "neither".
You can also consider dropping the frames with state "neither" from your analysis to make the calculation cleaner (i.e., turn it into binary classification).
This is the approach we took for the enzyme PTP1B, which you can find described in our preprint. 


## Tutorials Available
All tutorials include the setup and post-processing steps used for each system. All tutorials used datasets we analysed in our [preprint](TODO)

1. **[Tutorial_PTP1B_Classification_ML_Stats.ipynb](LINK HERE TODO)**  - Perform binary classification ML and statistical analysis on simulations of PTP1B. Used to describe the differences in the closed and open WPD-loop states of PTP1B.   

2. **[Tutorial_KE07_Regression_ML_Stats.ipynb](LINK HERE TODO)** - Perform regression ML and statistical analysis on a kemp eliminse enzyme. Here the target value is the side chain dihedral of W50. 

3. **[Tutorial_Process_PyContact_GUI_Input.ipynb](LINK HERE TODO)** - This tutorial will provide a short example of how to use the "pycontact_processing.py" module to load in a PyContact dataset generated via the PyContact GUI. Please note it is recommended to use the ["run_pycontact.py"](https://github.com/kamerlinlab/key-interactions-finder/blob/main/key_interactions_finder/run_pycontact.py) script provided in this repo instead - see section: "Running PyContact" below. 

4. **[network_analysis_tutorial](LINK HERE TODO)** - Preperation of PTP1B inputs required for graph theory based calculations. This tutorial is in its own folder, as two additional scripts are provided: 
      - A .R scipt (which uses [BIO3D](http://thegrantlab.org/bio3d_v2/)) to perform [WISP](https://pubs.acs.org/doi/10.1021/ct4008603)  
      - A python script to generate PyMOL comptabile figures depicting the results from the WISP calculation (The .R script will only generate VMD compataible ones.) 


## Citing this work
If you make use of this package please cite our preprint: TODO. 

## Issues/Questions/Contributions
All welcome. Please feel free to open an issue or submit a pull request as necessary. Feature requests are welcome too. 
You can also reach me at: rory.crean [at] kemi.uu.se
