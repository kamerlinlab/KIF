## Synthetic Data Test 

In response to a reviewer comment (at the time of writing this the manuscript is currently under revisions) we have generated a synthetic dataset with known correlations 
to a defined target variable. This has been used to validate the methods contained within KIF work correctly/as expected.  

**The reviewer comment was as follows:** "Construct a model system with synthetic data where correlated features are known a priori, and show the method works. I would especially like to see a test where duplicate copies of some features are added. Does the method still work?". 


**To address this, this section contains 2 notebooks:**
- [1_Gen_Synth_Datasets.ipynb](https://github.com/kamerlinlab/KIF/blob/main/tutorials/synthetic_data_test/1_Gen_Synth_Datasets.ipynb) -
The synthetic data required to test the methods within KIF will be generated and analysed to ensure they reasonably represent a typical dataset.

- [2_Run_Synth_Datasets.ipynb](https://github.com/kamerlinlab/KIF/blob/main/tutorials/synthetic_data_test/2_Run_Synth_Datasets.ipynb) - 
The synthetic data generated in the first notebook will be put through KIF and the results obtained (per feature/interaction scocres) will be analysed against the known correlations. 


#### Summary of the results:
The results generated from this analysis are being written up now to be included in the next version of the manuscript (they will be in supplementary information, Section S3). 

**They can be summarized as follows though:**
- All methods available within KIF work as expected, given the nature of the dataset (many features that were highly correlated to the target), the stats methods were better suited for the problem. 
- Multi-collinearity and duplicated features were not a problem for the stats methods within KIF. As expected, this caused issues for the ML methods. 
- This dataset again highlights that the methods from the stats modules are better suited to the typical problems KIF is designed to solve - Identify and score all the important interactions within a protein that regulates a conformational change of interest.
