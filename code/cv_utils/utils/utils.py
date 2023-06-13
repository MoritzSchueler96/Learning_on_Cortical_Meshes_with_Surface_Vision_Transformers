
""" Loading SLCN challenge data. """

__author__ = "Fabi Bongratz, Moritz Schüler"
__email__ = "fabi.bongratz@gmail.com, moritz.schueler@tum.de"

import os
import re

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from trimesh import Trimesh
import torch
import random
import wandb
import timm

from dotenv import load_dotenv

from matplotlib import pyplot as plt
import pyvista as pv
# from ipyvtklink.viewer import ViewInteractiveWidget
from functools import wraps
from pytorch_lightning.utilities.seed import seed_everything


load_dotenv()

BASE_DATA_FOLDER = os.getenv("BASE_DATA_FOLDER")

# The base folder of the challenge data, clone from
# https://gin.g-node.org/lzjwilliams/geometric-deep-learning-benchmarking

# If you don't change the original data structure, no need to change anything
# here
REGRESSION_FOLDER = os.path.join(BASE_DATA_FOLDER, "geometric-deep-learning-benchmarking", "Data", "Regression")
NATIVE_FEATURES_DIR = os.path.join(
    REGRESSION_FOLDER, "Native_Space", "regression_native_space_features"
)
NATIVE_SURFACES_DIR = os.path.join(
    REGRESSION_FOLDER, "Native_Space", "regression_native_space_surfaces"
)
TEMPLATE_FEATURES_DIR = os.path.join(
    REGRESSION_FOLDER, "Template_Space", "regression_template_space_features"
)
TEMPLATE_SURFACES_DIR = os.path.join(
    REGRESSION_FOLDER, "Template_Space", "regression_template_space_surfaces"
)
TABULAR_FILE = os.path.join(
    BASE_DATA_FOLDER, "dHCP_gDL_demographic_data.csv"
)

def set_seeds(seed: int):
    # For reproducibility
    seed = seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    os.environ["PYTHONHASHSEED"] = f"{seed}"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_weights_imagenet(state_dict,state_dict_imagenet,nb_layers):

    state_dict['mlp_head.0.weight'] = state_dict_imagenet['norm.weight'].data
    state_dict['mlp_head.0.bias'] = state_dict_imagenet['norm.bias'].data

    # transformer blocks
    for i in range(nb_layers):
        state_dict['transformer.layers.{}.0.norm.weight'.format(i)] = state_dict_imagenet['blocks.{}.norm1.weight'.format(i)].data
        state_dict['transformer.layers.{}.0.norm.bias'.format(i)] = state_dict_imagenet['blocks.{}.norm1.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.norm.weight'.format(i)] = state_dict_imagenet['blocks.{}.norm2.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.norm.bias'.format(i)] = state_dict_imagenet['blocks.{}.norm2.bias'.format(i)].data

        state_dict['transformer.layers.{}.0.fn.to_qkv.weight'.format(i)] = state_dict_imagenet['blocks.{}.attn.qkv.weight'.format(i)].data

        state_dict['transformer.layers.{}.0.fn.to_out.0.weight'.format(i)] = state_dict_imagenet['blocks.{}.attn.proj.weight'.format(i)].data
        state_dict['transformer.layers.{}.0.fn.to_out.0.bias'.format(i)] = state_dict_imagenet['blocks.{}.attn.proj.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.fn.net.0.weight'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc1.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.fn.net.0.bias'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc1.bias'.format(i)].data

        state_dict['transformer.layers.{}.1.fn.net.3.weight'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc2.weight'.format(i)].data
        state_dict['transformer.layers.{}.1.fn.net.3.bias'.format(i)] = state_dict_imagenet['blocks.{}.mlp.fc2.bias'.format(i)].data

    return state_dict

def load_model_weights(model, config):
    new_state_dict = None
    if config['setup']['load_weights'] == "wandb" and config['weights']['wandb']:
        print('Loading weights from wandb pretraining')
        art = wandb.run.use_artifact(config["weights"]["wandb"], type='model')
        artifact_dir = art.download()
        checkpoint = torch.load(f"{artifact_dir}/model.ckpt", map_location=model.device)
        new_state_dict = checkpoint["state_dict"]

    if config['setup']['load_weights'] == "imagenet" and config['weights']['imagenet']:
        print('Loading weights from imagenet pretraining')
        model_trained = timm.create_model(config['weights']['imagenet'], pretrained=True)
        new_state_dict = load_weights_imagenet(model.state_dict(),model_trained.state_dict(),config['transformer']['depth'])

    return new_state_dict


def get_sub_ses_ids():
    """ Return a list of all scan ids (a combination of subject and session id) in the dataset.
    """
    extract_id = lambda x: re.sub(
        r"_[RL]\.((white)|(pial))\.shape\.gii", "", x
    )

    ids = sorted(list(set(map(extract_id, os.listdir(NATIVE_SURFACES_DIR)))))

    return ids


def _load_surface_gii_trimesh(path):
    """ Load a gii surface and return Trimesh object
    """
    surface = nib.load(path)
    points, tris = surface.agg_data()

    return Trimesh(points, tris)


def _load_features_gii_np(path):
    """ Load gii features and return as numpy array with shape
    (n_vertices, n_features)
    """
    features = nib.load(path)

    return np.stack(features.agg_data(), axis=1)


def _load_all_surfaces(path, scan_id):
    """ Load all surfaces in a certain directory of one subject
    """
    data = {}

    # Iterate over hemispheres and surfaces
    for s_i in ("L.white", "R.white", "L.pial", "R.pial"):
        data[s_i] = _load_surface_gii_trimesh(os.path.join(path, scan_id + "_" + s_i + ".shape.gii"))

    return data


def _load_all_features(path, scan_id):
    """ Load all surfaces in a certain directory of one subject
    """
    data = {}

    # Iterate over hemispheres and surfaces
    for s_i in ("L", "R"):
        data[s_i] = _load_features_gii_np(os.path.join(path, scan_id + "_" + s_i + ".shape.gii"))

    return data


def load_challenge_data(start=None, end=None):
    """ Load all the challenge data into a dict. This dict contains an entry
    for each subject and for each subject the following data fields:
        - "sub": The subject ID
        - "ses": The session ID
        - "native_surfaces": Surfaces in native space (trimesh meshes)
        - "template_surfaces": Surfaces in template space (trimesh meshes)
        - "native_features": Numpy arrays of shape (40962x4) containing the
        input features per vertex on the 6-icosphere computed in native space
        - "template_features": Numpy arrays of shape (40962x4) containing the
        input features per vertex on the 6-icosphere computed in template space
        - "tabular": Additional tabular data (GA at birth,  PMA at scan, Sex,
        Birthweight, Head circumference) per subject
    """
    data = {}
    ids = get_sub_ses_ids()

    # Confounder data
    tab_data = pd.read_csv(TABULAR_FILE)
    assert len(ids) == tab_data.shape[0]

    if start is None:
        start = 0
    if end is None:
        end = len(ids)
        
    ids = ids[start:end]

    # Iterate over scans
    for i in tqdm(ids, leave=True, position=0, desc="Loading data..."):

        try:
            sub = i.split("_")[0].removeprefix("sub-")
            ses = int(i.split("_")[1].removeprefix("ses-"))
        except:
            sub = remove_prefix(i.split("_")[0], "sub-")
            ses = int(remove_prefix(i.split("_")[1], "ses-"))

        data[i] = {
            "sub": sub,
            "ses": ses,
            "native_surfaces": _load_all_surfaces(NATIVE_SURFACES_DIR, i),
            "template_surfaces": _load_all_surfaces(TEMPLATE_SURFACES_DIR, i),
            "native_features": _load_all_features(NATIVE_FEATURES_DIR, i),
            "template_features": _load_all_features(TEMPLATE_FEATURES_DIR, i),
            "tabular": tab_data.loc[
                (tab_data['Subject ID'] == sub) & (tab_data['Session ID'] == ses)
            ]
        }

    return data

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

class iPlotter(pv.Plotter):
    """Wrapping of PyVista's Plotter to be used interactively in Jupyter."""

    def __init__(self, *args, **kwargs):
        transparent_background = kwargs.pop('transparent_background', pv.rcParams['transparent_background'])
        kwargs["notebook"] = False
        kwargs["off_screen"] = False
        pv.Plotter.__init__(self, *args, **kwargs)
        self.ren_win.SetOffScreenRendering(1)
        self.off_screen = True
        self._widget = ViewInteractiveWidget(self.ren_win, transparent_background=transparent_background)


    @wraps(pv.Plotter.show)
    def show(self, *args, **kwargs):
        kwargs["auto_close"] = False
        _ = pv.Plotter.show(self, *args, **kwargs) # Incase the user sets the cpos or something
        return self.widget

    @property
    def widget(self):
        self._widget.full_render()
        return self._widget


def render_data(file_name: str) -> pv.plotting.Plotter:
    pv.start_xvfb()
    plotter = iPlotter()
    
    s_poly = combine_data(file_name)
    plotter.add_mesh(s_poly, smooth_shading=True)
    # plotter.add_mesh_clip_plane(s_poly)
    
    return plotter

def combine_data(file_name: str):
    for i in ["_L", "_R"]:
        surface_file = file_name + i + ".pial.shape.gii"
        file = os.path.join(NATIVE_SURFACES_DIR, surface_file)
        surface = nib.load(file)

        #################################################
        # Rendering
        #################################################
        s_points, s_tris = surface.agg_data()
        s_faces = np.hstack([np.ones([s_tris.shape[0], 1], dtype=int) * s_tris.shape[1], s_tris])
        s_poly = pv.PolyData(s_points, s_faces)

        return s_poly

def save_gifti(data, filename):
    gifti_file = nib.gifti.gifti.GiftiImage()
    gifti_file.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(data))
    nib.save(gifti_file,filename)


if __name__ == '__main__':

    data = load_challenge_data(start=0, end=5)
    print("Loading Done")

    df = pd.read_csv(TABULAR_FILE)
    subject_id = [key for key in df.keys() if "Subject ID" in key][0]
    session_id = [key for key in df.keys() if "Session ID" in key][0]
    df["filename"] = "sub-" + df[subject_id] + "_ses-" + df[session_id].astype(str)
    filenames = df["filename"].to_list()

    # for filename in filenames:
    #     plotter = render_data(filename)
    #     plotter.show()

    filename = filenames[0]
    poly = combine_data(filename)
    plotter = render_data(filename)
    plotter.show()


    obj = data[filename]
    natives = obj["native_surfaces"]
    tri = natives["L.white"]

    #viewer = SceneViewer()
