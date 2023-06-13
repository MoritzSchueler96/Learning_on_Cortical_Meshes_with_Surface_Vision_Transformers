import logging
import os

import pandas as pd
from cv_utils.utils.split_data import split_data

logger = logging.getLogger(__name__)
FORMAT = '%(asctime)s - %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

data = pd.read_csv("./code/cv_utils/labels/dHCP_gDL_demographic_data.csv")
# Subject ID ,Session ID,GA at birth (weeks),PMA at scan (weeks),"Sex (M=1,F=2)",Birthweight (kg),Head circumference at scan (cm)

name_mapping = {"GA at birth (weeks)": "birth_age", "PMA at scan (weeks)": "scan_age", "Sex (M=1,F=2)": "Sex"}
data = data.rename(columns=name_mapping)

mandatory_columns = ["Subject ID","Session ID","birth_age","scan_age"]
label = {"birth_age": "numerical"}
balance_columns = {"birth_age": "cr(birth_age, df=4)", "Sex": "C(Sex)"}
id = "Subject ID"
missing_values = "fill"
datasets = split_data(data, mandatory_columns, label=label, id=id, balance_columns=balance_columns, filter_columns=None, missing_values=missing_values, method="pad", seed=42)

inverse_name_mapping = {y: x for x, y in name_mapping.items()}
# inverse_name_mapping = dict(zip(name_mapping.values(), name_mapping.keys()))

save_folder="./code/cv_utils/labels/birth_age/"

splits = ["train", "validation", "test"]
for split, data in zip(splits, datasets):
    data = data.rename(columns=inverse_name_mapping)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    filename = os.path.join(save_folder, f"{split}.csv")
    data.to_csv(filename)

print(len(datasets))
