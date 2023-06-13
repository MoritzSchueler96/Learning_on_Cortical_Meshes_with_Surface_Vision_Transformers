import os
from dotenv import load_dotenv

import nibabel as nb
import pandas as pd
from matplotlib import pyplot as plt

load_dotenv()

BASE_REPO_FOLDER = os.getenv("BASE_REPO_FOLDER")
BASE_DATA_FOLDER = os.getenv("BASE_DATA_FOLDER")

def get_config():
    config = {}
    config["data"] = {
        "data_path": "data/ADNI_CSR/FS72/",
        "label_path": "./code/cv_utils/labels/",
        "task": "alzheimer",
        "configuration": "template", #template #native
        "split": "train", #train #validation #test
    }

    return config

def get_data(config):
    split = config['data']['split']
    task = config['data']['task']

    label_path = config['data']['label_path']
    label_path = os.path.join(BASE_REPO_FOLDER, label_path)

    #### Load data
    data = pd.read_csv(os.path.join(label_path, '{}/{}.csv'.format(task,split)))
    data["ids"] = data["IMAGEUID"].astype(str)

    return data

def get_ids(data):
    healthy = data[data["DX"] == "CN"].reset_index()
    id_healthy = healthy[healthy["AGE"] == 87.4]["ids"].item()
    mild = data[data["DX"] == "MCI"].reset_index()
    id_mild = mild[mild["AGE"] == 87.5]["ids"].item()
    ill = data[data["DX"] == "Dementia"].reset_index()
    id_ill = ill[ill["AGE"] == 87.7]["ids"].item()
    ids = {"Healthy": id_healthy, "MCI": id_mild, "Dementia": id_ill}

    return ids

def plot_single_slice(config, ids, slice=85, condition="Dementia"):
    data_path = config["data"]["data_path"]
    path_to_data = os.path.join(BASE_REPO_FOLDER, data_path)

    # fig = plt.figure()
    fig, axs = plt.subplots(1, 1) #, figsize=(10, 6)) #, constrained_layout=True)

    plt.gray()

    file = "mri.nii.gz"
    for item in ids.items():
        if item[0] != condition: continue
        path = os.path.join(path_to_data, str(item[1]))
        data = nb.load(os.path.join(path, file)).get_fdata()        
        axs.set_ylabel("X")
        axs.set_xticks([], [])
        axs.set_yticks([], [])
        axs.imshow(data[:,:,slice].squeeze(), interpolation='nearest')

    plt.show()

    return fig, axs

def plot_specific_slices(config, ids, num_slices=4, equally_spaced=False, slices=[100, 95, 85, 75]):
    data_path = config["data"]["data_path"]
    path_to_data = os.path.join(BASE_REPO_FOLDER, data_path)

    if equally_spaced:
        slc = data.shape[2] // (num_slices+1)
        slices = list(range(slc, data.shape[2], slc))[:num_slices]
        # slices = [130, 110, 100, 95, 85, 75, 45]

    fig, axs = plt.subplots(len(ids.keys()), len(slices)) #, figsize=(10, 6)) #, constrained_layout=True)
    plt.gray()
    
    file = "mri.nii.gz"
    for row, item in zip(axs, ids.items()):
        print(item)
        path = os.path.join(path_to_data, str(item[1]))
        data = nb.load(os.path.join(path, file)).get_fdata()        
        row[0].set_ylabel(item[0])
        for col, slice in zip(row, slices):
            if item[0] == "Healthy":
                col.set_title(f"Slice at {slice}")
            col.set_xticks([], [])
            col.set_yticks([], [])
            col.imshow(data[:,:,slice].squeeze(), interpolation='nearest')

    plt.show()

    return fig, axs

def plot_animated_image(config, ids):
    data_path = config["data"]["data_path"]
    path_to_data = os.path.join(BASE_REPO_FOLDER, data_path)

    for item in ids.items():
        print(item)
        path = os.path.join(path_to_data, str(item[1]))
        file = "mri.nii.gz"
        data = nb.load(os.path.join(path, file)).get_fdata()
        nb.viewers.OrthoSlicer3D(data).show()

if __name__ == "__main__":
    
    config = get_config()
    data = get_data(config)
    ids = get_ids(data)
    print(ids)

    # plot_specific_slices(config, ids, equally_spaced=False, num_slices=4, slices=[100, 95, 85, 75])
    plot_single_slice(config, ids, 85, "Dementia")
    # plot_animated_image(config, ids)    


    print("Done")
