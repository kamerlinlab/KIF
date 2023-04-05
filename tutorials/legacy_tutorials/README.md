#### Legacy Tutorials

**These tutorials are relevant for versions of KIF < 0.2.0. After this point KIF was updated to include its own contact analyzer, which is the recommended approach. If you have not yet run simulations please check your version of KIF is up to date and refer to the [more up to date tutorials](https://github.com/kamerlinlab/KIF/tree/main/tutorials) **

These tutorials used a program called [PyContact](https://github.com/maxscheurer/pycontact) (not made by us) to generate the non-covalent interaction data. This required another install and dependency. If you have used PyContact to generate your input files then you can follow the tutorials provided here to analyse the results.

One final thing to note is that workflows in these tutorials are identical to the current tutorials after step 1.


### Some Information on Preparing to run PyContact
To run calculations with PyContact we have provided a script in this folder to do this, called "run_pycontact.py". From this script you'll obtain two outputs (one with summary stats - not needed for KIF, and the other with per frame interaction scores). This script helps to standardize the output from PyContact making it easier for KIF to handle the data. If you instead wish to process the data through the PyContact GUI, please refer to the tutorial titled: "Tutorial_Process_PyContact_GUI_Input.ipynb" for how to go about this. All other tutorials used datasets generated from the custom script described above.

For a large number of frames and/or a large system, you will likely need to break up your PyContact calculation into blocks (to prevent running out of memory). We did this by making a single trajectory (of all frames we wanted to analyse) and submitting several (between 10-20) PyContact jobs on different residue ranges. Merging these results files back together again can be done with KIF - see tutorials (1 or 2).


### Legacy Tutorials Available:

1. **[Tutorial_PTP1B_Classification_ML_Stats.ipynb](https://github.com/kamerlinlab/KIF/blob/main/tutorials/Tutorial_PTP1B_Classification_ML_Stats.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kamerlinlab/KIF/blob/main/tutorials/legacy_tutorials/Tutorial_PTP1B_Classification_ML_Stats.ipynb)**  - Perform binary classification ML and statistical analysis on simulations of PTP1B. Used to describe the differences in the closed and open WPD-loop states of PTP1B.

2. **[Tutorial_KE07_Regression_ML_Stats.ipynb](https://github.com/kamerlinlab/KIF/blob/main/tutorials/Tutorial_KE07_Regression_ML_Stats.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kamerlinlab/KIF/blob/main/tutorials/legacy_tutorials/Tutorial_KE07_Regression_ML_Stats.ipynb)** - Perform regression ML and statistical analysis on a kemp eliminase enzyme. Here the target value is the side chain dihedral of W50.

3. **[Tutorial_Process_PyContact_GUI_Input.ipynb](https://github.com/kamerlinlab/KIF/blob/main/tutorials/Tutorial_Process_PyContact_GUI_Input.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kamerlinlab/KIF/blob/main/tutorials/legacy_tutorials/Tutorial_Process_PyContact_GUI_Input.ipynb)** - This tutorial will provide a short example of how to use the "pycontact_processing.py" module to load in a PyContact dataset generated via the PyContact GUI. Please note it is recommended to use the ["run_pycontact.py"](https://github.com/kamerlinlab/key-interactions-finder/blob/main/key_interactions_finder/run_pycontact.py) script provided in this repo instead - see section: "Running PyContact" below.



