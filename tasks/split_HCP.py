import logging

import pandas as pd
from cv_utils.utils.split_data import split_data

logger = logging.getLogger(__name__)
FORMAT = '%(asctime)s - %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

data = pd.read_csv("./code/cv_utils/labels/HCP_data.csv")
# Subject,Age_in_Yrs
mandatory_columns = ["Subject","Age_in_Yrs"] # ,"Ethnicity", "BMI", "Opiates"] # ,"Gender" # list(balance_columns.keys())
# filter_columns = {"SSAGA_Alc_D4_Ab_Dx": "CN"}
label = {"Age_in_Yrs": "numerical"}
id = "Subject"
balance_columns = {"Age_in_Yrs": "cr(Age_in_Yrs, df=4)"} #, "Ethnicity": "C(Ethnicity)", "BMI": "cr(BMI, df=3)", "Opiates": "C(Opiates)"} # "Gender": "C(Gender)", 
missing_values = "fill"
save_folder = "./code/cv_utils/labels/HCP_age/"

datasets = split_data(data, mandatory_columns, label=label, id=id, balance_columns=balance_columns, filter_columns=None, missing_values=missing_values, method="pad", seed=42, save_folder=save_folder)

train, val, test = datasets

print(len(datasets))
