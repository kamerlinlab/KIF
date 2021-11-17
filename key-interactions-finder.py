from key_interactions_finder import pycontact_processing #as pycontact_processing

from key_interactions_finder import data_preperation
import pandas as pd 
import numpy as np
import re 

# Main module used to run 







############################### pycontact_processing.py ###############################
# # Multi-Files 
# dataset = pycontact_processing.PyContactInitializer()

# in_dir = "example_data"
# out_dir = "example_data"
# pycontact_files = ["PTP1B_CSP_Part1_Contacts.txt", "PTP1B_CSP_Part2_Contacts.txt"]
# base_name = "PTP1B_CSP"

# df = dataset.process_pycontact_files(pycontact_files, base_name, in_dir, out_dir, multiple_files=True)
# df.to_csv("example_data/PTP1B_CSP_Test.csv", sep=',')
# print(df)



# # Single-File 
# # dataset = pycontact_processing.PyContactInitializer()

# # in_dir = "example_data"
# # out_dir = "example_data"
# # pycontact_files = "PTP1B_CSP_Part1_Contacts.txt"
# # base_name = "PTP1B_CSP"

# # df = dataset.process_pycontact_files(pycontact_files, base_name, in_dir, out_dir, multiple_files=False)
# # print(df)
############################### pycontact_processing.py ###############################





############################### data_preperation.py ###############################
# UnsupervisedFeatureData
unsupervised_data = data_preperation.UnsupervisedFeautureData()

min_occupancy = 25
df = pd.read_csv("example_data/PTP1B_CSP_Test.csv", sep=',') # in future will use just the step aboves DF!
print(unsupervised_data.filter_features(df, min_occupancy))
print(df)

# SupervisedFeatureData
# supervised_data = data_preperation.UnsupervisedFeautureData()

############################### data_preperation.py ###############################






