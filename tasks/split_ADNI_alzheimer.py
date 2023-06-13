import pandas as pd
import logging
from cv_utils.utils.split_data import split_data

logger = logging.getLogger(__name__)
FORMAT = '%(asctime)s - %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

data = pd.read_csv("./code/cv_utils/labels/ADNI_data.csv")
mandatory_columns = ["PTID","VISCODE","RID","IMAGEUID","Month","AGE","Years_bl","PTGENDER","DX"]
split_ratio = [0.6, 0.2, 0.2]
filter_columns = None #{"DX": "CN"}
label = {"DX": "categorical"}
id = "IMAGEUID"
balance_columns = {"AGE": "cr(AGE, df=4)", "PTGENDER": "C(PTGENDER)", "DX": "C(DX)"}
missing_values = "fill"
oversampling = True
undersampling = False
save_folder = "./code/cv_utils/labels/alzheimer_oversampled_train/"

datasets = split_data(data, mandatory_columns, split_ratio=split_ratio, label=label, id=id, balance_columns=balance_columns, filter_columns=filter_columns, missing_values=missing_values, method="pad", seed=42, oversampling=oversampling, undersampling=undersampling, save_folder=save_folder)

train, val, test = datasets

print(len(datasets))
