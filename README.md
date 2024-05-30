# KIF - Key Interactions Finder
 A python package to identify the key molecular interactions that regulate any conformational change.

### New in Version 0.4.0 - (30/05/2024)

Now KIF results can be projected onto a given 3D structure using [ChimeraX](https://www.cgl.ucsf.edu/chimerax/). Previously only [PyMOL](https://pymol.org/) was possible. Thanks to @darianyang for adding this functionality. 

Check out the tutorial to see it in action: 


In order to make use of this you'll need to update your install of KIF to at least 0.4.0 or greater:
```
pip install --upgrade KIF
```

**Keep in mind** that your trajectory(s) must be [imaged](http://ambermd.org/Questions/periodic2.html) before using KIF, to remove periodic effects. 

![KIF_ReadMe_Pic](https://user-images.githubusercontent.com/49672044/207597051-7dcde86a-62bd-4f69-96aa-326cad938a65.png)


## In short, this package allows you to:
 - Identify important non-covalent interactions that are associated with any conformational change you are interested in (as long as you can describe the descriptor and sample the descriptor in your MD simulations). The non-covalent interactions are scored according to their association/importance to the conformational change and you can easily convert the per interaction/feature scores to per residue scores as well.
 - Generate [PyMOL](https://pymol.org/2/) output scripts that enable you to visualise your results on 3D structures.
 - Generate per residue correlation and distance matrices that can be easily applied to the many graph theory methods available in order to study protein interaction networks and allostery within your system (no descriptor/target variable required for this).

Note that how you define the descriptor is up to you, and you can use either a continuous variable or a categorical variable (some tips on how to decide what to use will be given below).

**More Detail Please!**
For a more complete description of KIF, [please refer to our article](https://aip.scitation.org/doi/10.1063/5.0140882). Included in the article is a description of some of the generic workflows possible alongside the application of KIF to several different biomolecular systems.

There are also tutorials available (discussed below).

## The approximate workflow for this package is as follows:
1. Run MD simulations on your system(s) of interest with whichever MD engine you want.
2. (Both can be done simultaneously)
    1. Analyse the trajectory frames with KIF to determine all non-covalent interactions.* See tutorial 0 for how to do this.
    2. (Optional) Calculate the value of a target variable for each frame using whatever approach you see fit (tips provided below).
3. Load your non-covalent interactions data and optionally generated target data into KIF and perform some combination of machine learning, statistical analysis or network analysis.
4. Output the results for analysis. This includes visualisation scripts which are compatible with [PyMOL](https://pymol.org/2/) so you can see the results on a 3D protein structure.


*The previous implementation of KIF used [PyContact](https://github.com/maxscheurer/pycontact) (note that this program is not made by us), to determine all the non-covalent interactions in your trajectory. You can still use this approach if desired, see the legacy tutorials section below.


## Dependencies and Install
- Python 3.7 or higher is required as this package uses dataclasses. We recommend python 3.10 as 3.7 is rather old.

**Option 1: Install with pip**
```
pip install KIF
```

**Option 2: Clone/Download Repo first and then run setup.py :**

```
cd KIF-main
python setup.py install
```

Note that in prior versions of KIF (less than 0.2.0) you would have needed to also install [PyContact (see the repo for how to do this)](https://github.com/maxscheurer/pycontact). This is no longer the case.

## Choosing an Appropriate Target variable.
To perform either the machine learning or statistical analysis methods available in this package you will want to calculate a target variable to go alongside the features (non-covalent interactions identified by KIF). This target variable should as cleanly as possible describe your process of interest.

Below are some examples of what could work for you. Of course, this is use case specific so feel free to select what makes sense for your problem.

**Enzyme catalysis** - A per frame reacting atom distance for regression or define a max reacting atom distance cut-off to classify each frame as either "active" or "inactive".

**Conformational Changes** - Calculate the RMSD for each frame in your simulation to each conformational state. In this case of 2 different states, classify each frame as either "State 1", "State 2" or "neither" based on your RMSD metrics.

**For example:**
* if RMSD to "State 1" <= 1.5 Å and an RMSD to "State 2" >= 1.5 Å --> assign frame as "State 1".
* if RMSD to "State 1" => 1.5 Å and an RMSD to "State 2" <= 1.5 Å --> assign frame as "State 2".
* else --> assign frame as "neither".
You can also consider dropping the frames with state "neither" from your analysis to make the calculation cleaner (i.e., turn it into binary classification).
This is the approach we took for the enzyme PTP1B, which you can find described in our manuscript.

## Tutorials Available
All tutorials include the setup and post-processing steps used for each system. All tutorials used datasets we analyzed in our [manuscript](https://aip.scitation.org/doi/10.1063/5.0140882)

0. **[identify_contacts.py](https://github.com/kamerlinlab/KIF/blob/main/tutorials/identify_contacts.py)** - Example script showing how you can use KIF to identify all the non-covalent interactions present in your MD simulations. This needs to be determined before using the rest of KIF.

1. **[Tutorial_PTP1B_Classification_ML_Stats.ipynb](https://github.com/kamerlinlab/KIF/blob/main/tutorials/Tutorial_PTP1B_Classification_ML_Stats.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kamerlinlab/KIF/blob/main/tutorials/Tutorial_PTP1B_Classification_ML_Stats.ipynb)**  - Perform binary classification ML and statistical analysis on simulations of PTP1B. Used to describe the differences in the closed and open WPD-loop states of PTP1B.

2. **[Tutorial_KE07_Regression_ML_Stats.ipynb](https://github.com/kamerlinlab/KIF/blob/main/tutorials/Tutorial_KE07_Regression_ML_Stats.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kamerlinlab/KIF/blob/main/tutorials/Tutorial_KE07_Regression_ML_Stats.ipynb)** - Perform regression ML and statistical analysis on a kemp eliminase enzyme. Here the target value is the side chain dihedral of W50.


3. **[network_analysis_tutorial](https://github.com/kamerlinlab/KIF/tree/main/tutorials/network_analysis_tutorial) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kamerlinlab/KIF/blob/main/tutorials/network_analysis_tutorial/Step1_Tutorial_PTP1B_Network_Analysis.ipynb)** - Preparation of PTP1B inputs required for graph theory based calculations. This tutorial is in its own folder, as two additional scripts are provided:
      - A .R script (which uses [BIO3D](http://thegrantlab.org/bio3d_v2/)) to perform [WISP](https://pubs.acs.org/doi/10.1021/ct4008603)
      - A python script to generate PyMOL compatible figures depicting the results from the WISP calculation (The .R script will only generate VMD compatible ones.)


4. **[Workup_PDZ_Bootstrapping.ipynb](https://github.com/kamerlinlab/KIF/blob/main/tutorials/Workup_PDZ_Bootstrapping.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kamerlinlab/KIF/blob/main/tutorials/Workup_PDZ_Bootstrapping.ipynb)** - Workup the bootstrapping calculations performed on the PDZ3 domain. As described in the notebook, the bootstrapping itself was computationally intensive to run so only the workup is included in this notebook. We have however included the [scripts that we used to perform bootstrapping for both PTP1B and the PDZ3 domain on Zenodo](https://zenodo.org/record/7104965#.Y5meLXbMKUk)


There are also [several legacy tutorials which can be found here](https://github.com/kamerlinlab/KIF/blob/main/tutorials/legacy_tutorials). These tutorials are almost identical to the equivalent ones found above, but instead use non-covalent interaction data identified with PyContact instead of KIF.


## License and Disclaimer

This software is published under a GNU General Public License v2.0.

As the principal investigator behind this software is employed by Georgia Tech University we must also clarify: “The software is provided “as is.” Neither the Georgia Institute of Technology nor any of its units or its employees, nor the software developers of KIF or any other person affiliated with the creation, implementation, and upkeep of the software’s code base, knowledge base, and servers (collectively, the “Entities”) shall be held liable for your use of the platform or any data that you enter. The Entities do not warrant or make any representations of any kind or nature with respect to the System, and the Entities do not assume or have any responsibility or liability for any claims, damages, or losses resulting from your use of the platform. None of the Entities shall have any liability to you for use charges related to any device that you use to access the platform or use and receive the platform, including, without limitation, charges for Internet data packages and Personal Computers. THE ENTITIES DISCLAIM ALL WARRANTIES WITH REGARD TO THE SERVICE,INCLUDING WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE TO THE FULLEST EXTENT ALLOWED BY LAW.”



## Citing this work
If you make use of this package [please cite our manuscript:](https://aip.scitation.org/doi/10.1063/5.0140882).

KIF – Key Interactions Finder: A Program to Identify the Key Molecular Interactions that Regulate Protein Conformational Changes

Authors: Rory M. Crean, Joanna S. G. Slusky, Peter M. Kasson and Shina Caroline Lynn Kamerlin

DOI: 10.1063/5.0140882

## Issues/Questions/Contributions
All welcome. Please feel free to open an issue or submit a pull request as necessary. Feature requests are welcome too.
You can also reach me at: rory.crean [at] kemi.uu.se
