from ast import Assert
import streamlit as st
import os
import sys
from custom_blocks import sidebar, crop_and_load_volume, save_data_form
from visualization_helpers import VISUALIZATION
from data_classes import Numpy3D, Numpy2D
from utils import find_files_in_directory, regular_patching_2D
import numpy as np
import time
import torch
import git
from tqdm import tqdm
import re
from scipy.interpolate import interp2d


class CloneProgress(git.RemoteProgress):
    """ This class for tracking the clonning progress
    """
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()

    def update(self, op_code, cur_count, max_count=None, message=''):
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()

def predict_patch(data, dev, patchsize, step):
    patches, starting_indices = regular_patching_2D(data, 
                        patchsize=patchsize,
                        step=step,
                        verbose=True)
    prediction = np.zeros([2*data.shape[0], 2*data.shape[1]])
    for i, pi in enumerate(starting_indices):
        torch_data = torch.from_numpy(np.expand_dims(np.expand_dims(patches[i],axis=0),axis=0)).float()
        if (dev == torch.device("cuda")):
            torch_data = torch_data.cuda()
        print(f"Dim torch_data {torch_data.shape}")
        output = model(torch_data).detach().cpu().numpy().squeeze()
        prediction[2*pi[0]:2*pi[0]+2*patchsize[0], 2*pi[1]:2*pi[1]+2*patchsize[1]] = output
    return prediction

def predict_slice(data, dev):
    torch_data = torch.from_numpy(np.expand_dims(np.expand_dims(data,axis=0),axis=0)).float()
    if (dev == torch.device("cuda")):
        torch_data = torch_data.cuda()
    print(f"Dim torch_data {torch_data.shape}")
    prediction = model(torch_data).detach().cpu().numpy().squeeze()
    return prediction


def resize_array(arr1, arr2):
    # Get the dimensions of the two arrays
    n1, m1 = arr1.shape
    n2, m2 = arr2.shape
    
    # Create a 2D interpolation function for the second array
    f = interp2d(np.arange(m2), np.arange(n2), arr2, kind='linear')
    
    # Evaluate the interpolation function on a grid of points
    xnew = np.linspace(0, m2-1, m1)
    ynew = np.linspace(0, n2-1, n1)
    arr2_interp = f(xnew, ynew)
    
    return arr2_interp

def copy_to_top(file1_path, file2_path):
    marker = "# MERGED CONTENTS - DO NOT DUPLICATE"

    # Read the contents of the first file
    with open(file1_path, 'r') as file1:
        file1_content = file1.read()

    # Read the contents of the second file
    with open(file2_path, 'r') as file2:
        file2_content = file2.read()

    # Check if the files have already been merged
    if marker in file2_content:
        print("Files have already been merged.")
        return

    # Remove the "from model import common" line from the second file
    file2_content = file2_content.replace("from model import common", "")

    # Remove all references to "common."
    file2_content = re.sub(r'common\.', '', file2_content)

    # Write the marker, contents of the first file, and modified contents of the second file
    with open(file2_path, 'w') as file2:
        file2.write(marker + "\n" + file1_content + "\n" + file2_content)

st.markdown("### 🛈 Deep Learning for Simultaneous Seismic Image Super-Resolution and Denoising by Jintao Li (Links: [Paper](https://ieeexplore.ieee.org/abstract/document/9364884), [GitHub](https://github.com/JintaoLee-Roger/SeismicSuperResolution))")

with st.expander("Abstract"):
    st.markdown(""" ### Authors: Jintao Li; Xinming Wu; Zhanxuan Hu

Seismic interpretation is often limited by low resolution and strong noise data. To deal with this issue, we propose to leverage deep convolutional neural network (CNN) to achieve seismic image super-resolution and denoising simultaneously. To train the CNN, we simulate a lot of synthetic seismic images with different resolutions and noise levels to serve as training data sets. To improve the perception quality, we use a loss function that combines the ℓ1 loss and multiscale structural similarity loss. Extensive experimental results on both synthetic and field seismic images demonstrate that the proposed workflow can significantly improve the perception of quality of original data. Compared to conventional methods, the network obtains better performance in enhancing detailed structural and stratigraphic features, such as thin layers and small-scale faults. From the seismic images super-sampled by our CNN method, a fault detection method can compute more accurate fault maps than from the original seismic images.
""")
    
st.markdown("Some of my videos about the subject")
col1, col2 = st.columns(2)
with col1: 
    st.video('https://youtu.be/ohByy_yKV6M')
with col2:
    st.video('https://youtu.be/HV15X26itCA')

st.markdown("## 🔰 Install external dependencies")
with st.expander("GPU version"):
    st.code('''conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch''')
with st.expander("CPU version"):
    st.code('''conda install pytorch torchvision torchaudio cpuonly -c pytorch''')

st.markdown("## 🧬 Clone the SeismicSuperResolution repository")
st.info("Use the button below to clone the repo, but if it's not working as expected then please download the repo from https://github.com/JintaoLee-Roger/SeismicSuperResolution and place it in appdata/geophysics/post_processing/")
is_clone = st.button("Clone")
is_path_exist = os.path.isdir("./appdata/geophysics/post_processing/SeismicSuperResolution/")
if is_clone:
    if is_path_exist == False:
        with st.spinner("Clonning the repo"):
            git.Repo.clone_from("https://github.com/JintaoLee-Roger/SeismicSuperResolution.git", "./appdata/geophysics/post_processing/SeismicSuperResolution", progress=CloneProgress())
            copy_to_top('./appdata/geophysics/post_processing/SeismicSuperResolution/src/model/common.py', 
            './appdata/geophysics/post_processing/SeismicSuperResolution/src/model/unet.py')
            is_path_exist = True
    else: 
        st.warning("The path ./appdata/geophysics/post_processing/SeismicSuperResolution/ already exist.")


st.markdown("## 🧬 Download the pre-trained network")
st.info("Download model_best.pt from the author's Google Drive https://drive.google.com/drive/folders/1DuMdclOdeXDgGBOhsHSlEdTB_LvhIH-X?usp=sharing and place it in appdata/geophysics/post_processing/SeismicSuperResolution/")
is_weight_exist = os.path.exists("./appdata/geophysics/post_processing/SeismicSuperResolution/model_best.pt")

if "seismic" not in st.session_state or st.session_state.seismic_type=="2D":
        st.error("Please import 3D seismic first")
        st.stop()

st.markdown("## ✨ Here we go with the APP")

if is_path_exist == False or is_weight_exist == False:
        st.error("Please clone the repo and download the weights!")
        st.stop()

seismic = st.session_state.seismic
seismic_type = st.session_state.seismic_type

module_name = 'SuperRes'

# Initialize state

if module_name not in st.session_state:
    st.session_state[module_name] = {"numpy_data" : None,
        "numpy_result" : None, "is_predicted" : False, 'cropped_info' : None , \
        "step1_status" : None, "step2_status" : None, "step3_status" : None} 


sys.path.append('./appdata/geophysics/post_processing/SeismicSuperResolution/src/model/')
from appdata.geophysics.post_processing.SeismicSuperResolution.src.model.unet import UNet


if st.session_state.seismic_type=="3D":
    with st.expander("🟢 Step 1 - To work with this APP we need to load seismic into RAM"):
        st.subheader("To save your precious ram, you can now crop the volume here")

        step1_viz = VISUALIZATION(seismic, st.session_state.seismic_type)
        step1_viz.viz_data_3d(seismic, key=10, is_fspect=False)

        st.subheader("Cropping the volume")

        st.session_state[module_name]['numpy_data'], st.session_state[module_name]['cropped_info'] = \
            crop_and_load_volume(seismic, st.session_state[module_name]['numpy_data'], \
            st.session_state[module_name]['cropped_info'])

        if st.session_state[module_name]['numpy_data'] is not None:    
            step1_crop_viz = VISUALIZATION(st.session_state[module_name]['numpy_data'] , st.session_state.seismic_type)
            step1_crop_viz.viz_data_3d(st.session_state[module_name]['numpy_data'] , key = 20, is_fspect=False)

        # if range changes what to do?
        #TODO change corner point on plot

    with st.expander("🟢 Step 2 - Calculation"):
        st.subheader("Here we select the weights for computation")
        
        inference_form = st.form("Inference")
        col1, col2 = inference_form.columns(2)
        compute_size = col1.radio('Compute the whole slice at once or devide into patches', ["Slice", "Patch"], horizontal=True)
        patchsize = col2.selectbox("Define dimentions for the patch. Number of pixels in XY", np.arange(128, 1024, 128), 1)
        direction = inference_form.radio("It's a 2D network, thus you must decide the inference direction", ["Inline", "Xline"], horizontal=True)
        compute = inference_form.radio("Compute", ["GPU", "CPU"], horizontal=True)
        inference_submit = inference_form.form_submit_button("Submit")
        
        if inference_submit:
            #TODO may be two times memory allocation
            np_data = st.session_state[module_name]['numpy_data'].get_cube()
            model = UNet(feature_scale=1, scale=2)
            model.load_state_dict(
                torch.load(r"./appdata/geophysics/post_processing/SeismicSuperResolution/model_best.pt", map_location=f'cpu'))
            model.eval()

            dev = torch.device("cpu") 
            if compute == "GPU":
                dev = torch.device("cuda") if torch.cuda.is_available() else st.warning("NO GPU SUPPORT ESTABLISHED, CONTINUING WITH CPU")
                torch.cuda.empty_cache()

            model.to(dev)

            # data standartization
            np_data_min = float(np.min(np_data))
            np_data_max = float(np.max(np_data))
            np_data = (np_data - np_data_min) / (np_data_max - np_data_min)

            if direction == "Xline":
                np_data = np.swapaxes(np_data, 0, 1)

            my_bar = st.progress(0)
            metric = st.empty()
            t_start = 0
            predict = []
            for i in range(np_data.shape[0]):
                with metric.container():
                    col1, col2 = st.columns(2)
                    col1.metric(label="Slices out of {}".format(np_data.shape[0]), value=i)
                    col2.metric(label="Estimate wait time", value= "--" if t_start == 0 else time.strftime('%H:%M:%S',time.gmtime(((t_end - t_start)*(np_data.shape[0]-i)))))
                my_bar.progress(i/np_data.shape[0])
                t_start = time.time()
                if (compute_size == "Patch"):
                    slice = predict_patch(np_data[i,:,:], dev, patchsize=[patchsize, patchsize], step=[patchsize//2, patchsize//2])
                    predict.append(resize_array(np_data[i,:,:], slice))
                else:
                    slice = predict_slice(np_data[i,:,:], dev)
                    predict.append(resize_array(np_data[i,:,:], slice))
                t_end = time.time()

            if direction == "Xline":
                np_data = np.swapaxes(np_data, 0, 1)
                predict = np.swapaxes(predict, 0, 1)

            print (f"pred Min {np.min(predict)}, Max {np.max(predict)}")
            predict = np.array(predict)
            predict = predict*float(np_data_max - np_data_min) + np_data_min

            if (dev == torch.device("cuda")):
                torch.cuda.empty_cache()

            st.session_state[module_name]['numpy_result']  = Numpy3D(predict)
            st.session_state[module_name]['is_predicted']  = True
else:
    st.error("Please import 3D seismic first")
    st.stop()
        

if st.session_state[module_name]['is_predicted']:
    step2_viz = VISUALIZATION(st.session_state[module_name]['numpy_data']  , st.session_state.seismic_type)
    step2_viz.viz_sidebyside_3d(st.session_state[module_name]['numpy_data'], st.session_state[module_name]['numpy_result'] , minmax = True, key=30)


with st.expander("🟢 Step 3 - Save the results"):
    if st.session_state[module_name]['is_predicted']:
        numpy_result = st.session_state[module_name]['numpy_result'].get_cube()
        st.session_state[module_name]['step3_status'] = save_data_form(st.session_state[module_name], seismic, numpy_result, st.session_state[module_name]['step3_status'])
        st.info(st.session_state[module_name]['step3_status'])

sidebar()