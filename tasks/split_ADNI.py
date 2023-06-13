import pandas as pd
import logging
from cv_utils.utils.split_data import split_data

logger = logging.getLogger(__name__)
FORMAT = '%(asctime)s - %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

data = pd.read_csv("./code/cv_utils/labels/ADNI_data.csv")
mandatory_columns = ["PTID","VISCODE","RID","IMAGEUID","Month","AGE","Years_bl","PTGENDER","DX"]
filter_columns = {"DX": "CN"}
label = {"AGE": "numerical"}
id = "PTID"
balance_columns = {"AGE": "cr(AGE, df=4)", "PTGENDER": "C(PTGENDER)"}
missing_values = "fill"
save_folder = "./code/cv_utils/labels/adult_age/"

datasets = split_data(data, mandatory_columns, label=label, id=id, balance_columns=balance_columns, filter_columns=filter_columns, missing_values=missing_values, method="pad", seed=42, save_folder=save_folder)

train, val, test = datasets

print(len(datasets))
