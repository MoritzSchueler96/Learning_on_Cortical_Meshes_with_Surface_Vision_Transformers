# -*- coding: utf-8 -*-
# @Author: Simon Dahan
# @Last Modified time: 2022-01-12 15:37:28

'''
This file is used to preprocess data surface metrics into triangular patches
filling the entire surface. 
inputs: (N,C) - N subjects. C channels
outputs: (L,V,C) - L sequence lenght, V number of vertices per patch, C channels
'''

import argparse
import os
from dotenv import load_dotenv
import nibabel.freesurfer.io as frio

import nibabel as nb
import numpy as np
import pandas as pd
import yaml

load_dotenv()

BASE_REPO_FOLDER = os.getenv("BASE_REPO_FOLDER")
BASE_DATA_FOLDER = os.getenv("BASE_DATA_FOLDER")


def check_data_split(config):
    #### Check if data already split
    label_path = config['data']['label_path']
    label_path = os.path.join(BASE_REPO_FOLDER, label_path)
    split = config['data']['split']
    task = config['data']['task']

    FILE = os.path.join(label_path, '{}/{}.csv'.format(task,split))
    
    if not os.path.exists(FILE):
        print('')
        raise FileNotFoundError(f'{split}.csv file not found. Run data_split script to generate.')

def load_labels(config):
    configuration = config['data']['configuration']
    split = config['data']['split']
    task = config['data']['task']

    label_path = config['data']['label_path']
    label_path = os.path.join(BASE_REPO_FOLDER, label_path)

    print('')
    print('Task: {} - Split: {} - Data: {}'.format(task,split,configuration))
    print('')

    #### Load data
    data = pd.read_csv(os.path.join(label_path, '{}/{}.csv'.format(task,split)))

    return data

def normalize_data(config, data):
    ## data normalization 
    configuration = config['data']['configuration']
    split = config['data']['split']
    task = config['data']['task']

    label_path = config['data']['label_path']
    num_channels = data.shape[1]

    MEANS_FILE = os.path.join(label_path,'{}/{}/means.npy'.format(task,configuration))
    STDS_FILE = os.path.join(label_path,'{}/{}/stds.npy'.format(task,configuration))

    if not os.path.exists(MEANS_FILE) or not os.path.exists(STDS_FILE):
        if split == "train":
            means  = np.mean(np.mean(data,axis=2),axis=0)
            stds  = np.std(np.std(data,axis=2),axis=0)

            try:
                os.makedirs(os.path.join(config['data']['label_path'],'{}/{}/'.format(task,configuration)))
            except:
                print("Folder already exists.")

            np.save(MEANS_FILE, means)
            np.save(STDS_FILE, stds)
        else:
            raise FileNotFoundError("Mean and std files not available, please run this script for train split first.")
    
    means = np.load(MEANS_FILE)
    stds = np.load(STDS_FILE)

    normalized_data = (data - means.reshape(1,num_channels,1))/stds.reshape(1,num_channels,1)
    return normalized_data

def rearrange_data(config, normalized_data, ids):
    ico = config['resolution']['ico']
    sub_ico = config['resolution']['sub_ico']

    num_subjects = ids.shape[0]
    num_channels = normalized_data.shape[1]
    num_vertices = config['sub_ico_{}'.format(sub_ico)]['num_vertices']
    num_patches = config['sub_ico_{}'.format(sub_ico)]['num_patches']

    trimesh_file = os.path.join(BASE_REPO_FOLDER, './code/cv_utils/utils/triangle_indices_ico_{}_sub_ico_{}.csv'.format(ico,sub_ico))
    indices_mesh_triangles = pd.read_csv(trimesh_file)

    #shape of the data is num_subjects * 2, channels, num_triangles, num_vertices_per_triangle
    data = np.zeros((num_subjects*2, num_channels, num_patches, num_vertices))

    for i, idx in enumerate(ids):
        # print(idx)
        for j in range(num_patches):
            indices_to_extract = indices_mesh_triangles[str(j)].to_numpy()
            data[i,:,j,:] = normalized_data[2*i][:,indices_to_extract]
            data[i+num_subjects,:,j,:] = normalized_data[2*i+1][:,indices_to_extract]

    return data

def save_data(config, data, labels, ids):
    configuration = config['data']['configuration']
    split = config['data']['split']
    task = config['data']['task']

    output_folder = config['output']['folder'].format(task,configuration)
    output_folder = os.path.join(BASE_REPO_FOLDER, output_folder)

    print('')
    print('#'*30)
    print('#Saving: {} {} data'.format(split,configuration))
    print('#'*30)
    print('')

    try:
        os.makedirs(output_folder,exist_ok=False)
        print('Creating folder: {}'.format(output_folder))
    except OSError:
        print('folder already exist: {}'.format(output_folder))
    
    filename = os.path.join(output_folder,'{}_data.npy'.format(split,configuration))
    np.save(filename,data)
    filename = os.path.join(output_folder,'{}_labels.npy'.format(split,configuration))

    if split == "validation":
        labels = np.stack((ids, labels), axis=1)
    labels = np.concatenate((labels,labels))
    np.save(filename,labels)

    filename = os.path.join(output_folder,'{}_ids.npy'.format(split,configuration))
    np.save(filename,ids)

    print('')
    print(data.shape,labels.shape)

def rename_dHCP(data, task):
    data = data.rename(columns={"GA at birth (weeks)": "birth_age", "PMA at scan (weeks)": "scan_age"})
    data["ids"] = data["Subject ID"] + "_" + data["Session ID"].astype(str)
    data = data[["ids", task]]
    data = data.rename(columns={task: "labels"})
    return data

def load_data_dHCP(config, ids):
    
    configuration = config['data']['configuration']
    path_to_data = config['data']['data_path_gDL']
    path_to_data = os.path.join(BASE_DATA_FOLDER, path_to_data)

    data = []

    for idx in ids:
        # print(idx)
        filename = os.path.join(path_to_data, "{}_Space".format(configuration.title()), 'regression_{}_space_features'.format(configuration),'sub-{}_ses-{}_L.shape.gii'.format(idx.split('_')[0],idx.split('_')[1]))
        data.append(np.array(nb.load(filename).agg_data())[:,:])
        filename = os.path.join(path_to_data, "{}_Space".format(configuration.title()), 'regression_{}_space_features'.format(configuration),'sub-{}_ses-{}_R.shape.gii'.format(idx.split('_')[0],idx.split('_')[1]))
        data.append(np.array(nb.load(filename).agg_data())[:,:])
        
    data = np.asarray(data)

    return data, []

def get_data(path, configuration):
    data_left = []
    data_right = []

    files = os.listdir(path)

    for file in files:
        # print(file)
        # decide whether regular / template space or native space
        if ".reg." in file and configuration == "template":
            continue
        elif not ".reg." in file and configuration == "native":
            continue
        elif configuration not in ["template", "native"]:
            raise AssertionError

        # process files of left and right side
        if "lh." in file:
            data_left.append(frio.read_morph_data(os.path.join(path, file)))
        else:
            data_right.append(frio.read_morph_data(os.path.join(path, file)))
    
    return data_left, data_right

def rename_ADNI(data, task):
    data = data.rename(columns={"AGE": "adult_age", "DX": "alzheimer"})
    data["ids"] = data["IMAGEUID"].astype(str)
    if "alzheimer" in task or "baseline" in task:
        task_name = "alzheimer"
    elif "adult_age" in task:
        task_name = "adult_age"
    else:
        raise ValueError("Task not known.")
    data = data[["ids", task_name]]
    data = data.rename(columns={task_name: "labels"})
    return data

def load_data_ADNI(config, ids):

    configuration = config['data']['configuration']
    path_to_data = config['data']['data_path_ADNI']
    path_to_data = os.path.join(BASE_DATA_FOLDER, path_to_data)

    data = []
    ids_to_remove = []

    for i, idx in enumerate(ids):
        # print(id)
        path = os.path.join(path_to_data, str(idx))
        data_l, data_r = get_data(path, configuration)

        if np.asarray(data_l).shape != (7, 40962) or np.asarray(data_r).shape != (7, 40962):
            ids_to_remove.append(i)
            continue

        data.append(data_l)
        data.append(data_r)
        
    data = np.asarray(data)

    return data, ids_to_remove

def rename_HCP(data, task):
    data = data.rename(columns={"Age_in_Yrs": "HCP_age"})
    data["ids"] = data["Subject"].astype(str)
    data = data[["ids", task]]
    data = data.rename(columns={task: "labels"})
    return data

def load_data_HCP(config, ids):
    configuration = config['data']['configuration']
    path_to_data = config['data']['data_path_HCP']
    path_to_data = os.path.join(BASE_DATA_FOLDER, path_to_data)

    data = []
    ids_to_remove = []

    for i, idx in enumerate(ids):
        # print(id)
        path = os.path.join(path_to_data, str(idx))
        try:
            data_l, data_r = get_data(path, configuration)
            data.append(data_l)
            data.append(data_r)
        except:
            ids_to_remove.append(i)

    data = np.asarray(data)

    return data, ids_to_remove

def load_data_baseline(config, ids):
    num_slices = config["baseline"]["num_slices"]
    path_to_data = config['data']['data_path_ADNI_baseline']
    path_to_data = os.path.join(BASE_DATA_FOLDER, path_to_data)

    lst = np.empty((num_slices, 182, 218))
    ids_to_remove = []

    # 2D slices aus 3D Daten rausschneiden -> welche Slice
    for i, idx in enumerate(ids):
        # print(id)
        path = os.path.join(path_to_data, str(idx))
        file = "mri.nii.gz"
        data = nb.load(os.path.join(path, file)).get_fdata()
        
        slc = data.shape[2] // (num_slices+1)
        slices = list(range(slc, data.shape[2], slc))[:num_slices]
        data = data[:,:,slices]

        # nb.viewers.OrthoSlicer3D(data).show()
        # from matplotlib import pyplot as plt
        # for n in range(num_slices):
        #     plt.imshow(data[:,:,n], interpolation='nearest')
        #     plt.imshow(data[:,:,n].squeeze(), interpolation='nearest') -> grayscale
        #     plt.gray() -> alternative for grayscale
        #     plt.show()

        if np.asarray(data).shape != (182, 218) and np.asarray(data).shape != (182, 218, num_slices):
            ids_to_remove.append(i)
            continue
        
        data = np.moveaxis(data, -1, 0)
        lst = np.concatenate((lst, data), axis=0)
        
    data = np.asarray(lst)
    data = data[num_slices:,:,:]

    return data, ids_to_remove

def normalize_data_baseline(config, data):
    ## data normalization 
    configuration = config['data']['configuration']
    split = config['data']['split']
    task = config['data']['task']

    label_path = config['data']['label_path']

    MEANS_FILE = os.path.join(label_path,'{}/{}/means.npy'.format(task,configuration))
    STDS_FILE = os.path.join(label_path,'{}/{}/stds.npy'.format(task,configuration))

    if not os.path.exists(MEANS_FILE) or not os.path.exists(STDS_FILE):
        if split == "train":
            means  = np.mean(data)
            stds  = np.std(data)

            try:
                os.makedirs(os.path.join(config['data']['label_path'],'{}/{}/'.format(task,configuration)))
            except:
                print("Folder already exists.")

            np.save(MEANS_FILE, means)
            np.save(STDS_FILE, stds)
        else:
            raise FileNotFoundError("Mean and std files not available, please run this script for train split first.")
    
    means = np.load(MEANS_FILE)
    stds = np.load(STDS_FILE)

    normalized_data = (data - means)/stds
    return normalized_data

def rearrange_data_baseline(config, normalized_data, ids):
    ico = config['resolution']['ico']
    sub_ico = config['resolution']['sub_ico']

    num_subjects = ids.shape[0]
    num_channels = normalized_data.shape[1] # TODO: probieren slices als channel benutzen
    num_vertices = config['sub_ico_{}'.format(sub_ico)]['num_vertices']
    num_patches = config['sub_ico_{}'.format(sub_ico)]['num_patches']

    trimesh_file = os.path.join(BASE_REPO_FOLDER, './code/cv_utils/utils/triangle_indices_ico_{}_sub_ico_{}.csv'.format(ico,sub_ico))
    indices_mesh_triangles = pd.read_csv(trimesh_file)

    # für baseline
    #shape of the data is num_subjects * num_slices, channel/slices,  dim2D, andereDim2D 
    #shape of the data is num_subjects * 2, channels, num_triangles, num_vertices_per_triangle
    data = np.zeros((num_subjects*2, num_channels, num_patches, num_vertices))

    for i, idx in enumerate(ids):
        # print(idx)
        for j in range(num_patches):
            indices_to_extract = indices_mesh_triangles[str(j)].to_numpy()
            data[i,:,j,:] = normalized_data[2*i][:,indices_to_extract]
            data[i+num_subjects,:,j,:] = normalized_data[2*i+1][:,indices_to_extract]

    return data

def preprocess_baseline(config):
    check_data_split(config)

    print('')
    print('#'*30)
    print('Starting: preprocessing script')
    print('#'*30)

    labels = load_labels(config)
    labels = rename_ADNI(labels, config["data"]["task"])
    
    ids = labels['ids'].to_numpy().reshape(-1)
    labels = labels['labels'].to_numpy().reshape(-1)
    
    data, ids_to_remove = load_data_baseline(config, ids)

    ids = np.delete(ids, ids_to_remove)
    labels = np.delete(labels, ids_to_remove)

    normalized_data = normalize_data_baseline(config, data)
    data = rearrange_data_baseline(config, normalized_data, ids)
    save_data(config, data, labels, ids)

    print('#'*30)
    print('Finished: preprocessing script')
    print('#'*30)
    print('')

def preprocess_data(config, rename, load_data):

    check_data_split(config)

    print('')
    print('#'*30)
    print('Starting: preprocessing script')
    print('#'*30)

    labels = load_labels(config)
    labels = rename(labels, config["data"]["task"])
    
    ids = labels['ids'].to_numpy().reshape(-1)
    labels = labels['labels'].to_numpy().reshape(-1)
    
    data, ids_to_remove = load_data(config, ids)

    ids = np.delete(ids, ids_to_remove)
    labels = np.delete(labels, ids_to_remove)

    normalized_data = normalize_data(config, data)
    data = rearrange_data(config, normalized_data, ids)
    save_data(config, data, labels, ids)

    print('#'*30)
    print('Finished: preprocessing script')
    print('#'*30)
    print('')


def preprocess(config):
    if config["data"]["task"] in ["birth_age", "scan_age"]:
        preprocess_data(config, rename_dHCP, load_data_dHCP)
    elif config["data"]["task"] in ["adult_age", "alzheimer"] or "alzheimer" in config["data"]["task"]:
        preprocess_data(config, rename_ADNI, load_data_ADNI)
    elif config["data"]["task"] == "HCP_age":
        preprocess_data(config, rename_HCP, load_data_HCP)
    elif config["data"]["task"] == "baseline":
        preprocess_data(config, rename_ADNI, load_data_baseline)
    else:
        raise NotImplementedError


if __name__ == '__main__':

    # Set up argument parser
        
    parser = argparse.ArgumentParser(description='preprocessing HCP data for patching')
    
    parser.add_argument(
                        '-config',
                        type=str,
                        default='./code/cv_utils/configs/preprocessing.yml',
                        help='path where the data is stored')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call preprocessing
    preprocess(config)
