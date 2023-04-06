import os
import numpy as np

import git
from tqdm import tqdm
import streamlit as st
from custom_blocks import sidebar, crop_and_load_volume, save_data_form
from visualization_helpers import VISUALIZATION
from data_classes import Numpy3D
from utils import find_files_in_directory, predict_with_mask

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

class FaultNet:
    """This is a wrapper class for predict function in FaultNet repo
    """
    def __init__(self, model, device) -> None:
        self.model = model
        self.device = device

    def prediction_pytorch(self, model, data, device):
        """Function that converts input data into correct format

        Args:
            model (_type_): neural network
            data (_type_): dataset for prediction
            device (_type_): GPU or CPU

        Returns:
            _type_: _description_
        """
        model.eval()
        data = data[0, :, :, :, 0]
        result = prediction(model, data, device)
        return result[np.newaxis, :, :, :, np.newaxis]

    def predict(self, data,verbose=1):
        return self.prediction_pytorch(self.model, data, self.device)


st.markdown("### 🛈 Efficient Training of 3D Seismic Image Fault Segmentation Network under Sparse Labels by Weakening Anomaly Annotation by Yimin Dou (Links: [Paper](https://arxiv.org/abs/2110.05319), [GitHub](https://github.com/douyimin/FaultNet))")

with st.expander("Abstract"):
    st.markdown(""" ### Authors: Yimin Dou, Kewen Li, Jianbing Zhu, Timing Li, Shaoquan Tan, Zongchao Huang
Data-driven fault detection has been regarded as a 3D image segmentation task. The models trained from synthetic data are difficult to generalize in some surveys. Recently, training 3D fault segmentation using sparse manual 2D slices is thought to yield promising results, but manual labeling has many false negative labels (abnormal annotations), which is detrimental to training and consequently to detection performance. Motivated to train 3D fault segmentation networks under sparse 2D labels while suppressing false negative labels, we analyze the training process gradient and propose the Mask Dice (MD) loss. Moreover, the fault is an edge feature, and current encoder-decoder architectures widely used for fault detection (e.g., U-shape network) are not conducive to edge representation. Consequently, Fault-Net is proposed, which is designed for the characteristics of faults, employs high-resolution propagation features, and embeds MultiScale Compression Fusion block to fuse multi-scale information, which allows the edge information to be fully preserved during propagation and fusion, thus enabling advanced performance via few computational resources. Experimental demonstrates that MD loss supports the inclusion of human experience in training and suppresses false negative labels therein, enabling baseline models to improve performance and generalize to more surveys. Fault-Net is capable to provide a more stable and reliable interpretation of faults, it uses extremely low computational resources and inference is significantly faster than other models. Our method indicates optimal performance in comparison with several mainstream methods.""")

st.markdown("## My video Unboxing the FaultNet")
st.video('https://youtu.be/PxSNxuqaE14')

st.markdown("## 🔰 Install external dependencies")
with st.expander("GPU version"):
    st.code('''conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
conda install -c anaconda scikit-image''')
with st.expander("CPU version"):
    st.code('''conda install pytorch torchvision torchaudio cpuonly -c pytorch 
conda install -c anaconda scikit-image''')

st.markdown("## 🧬 Clone the FaultNet repository")
st.info("Use the button below to clone the repo, but if it's not working as expected then please download the repo from https://github.com/douyimin/FaultNet and place it in appdata/geophysics/faults/")
is_clone = st.button("Clone")
if is_clone:
    is_path_exist = os.path.isdir("./appdata/geophysics/faults/FaultNet/")
    if is_path_exist == False:
        with st.spinner("Clonning the repo"):
            git.Repo.clone_from("https://github.com/douyimin/FaultNet.git", "./appdata/geophysics/faults/FaultNet", progress=CloneProgress())
    else:
        st.warning("The path ./appdata/geophysics/faults/FaultNet/ already exist.")

st.markdown("## ✨ Here we go with the APP")

import torch

if "seismic" not in st.session_state or st.session_state.seismic_type=="2D":
        st.error("Please import 3D seismic first")
        st.stop()

seismic = st.session_state.seismic
seismic_type = st.session_state.seismic_type

MODULE_NAME = 'FaultNet'

# Initialize state

if MODULE_NAME not in st.session_state:
    st.session_state[MODULE_NAME] = {"numpy_data" : None,
        "numpy_result" : None, "is_predicted" : False, 'cropped_info' : None , \
        "step1_status" : None, "step2_status" : None, "step3_status" : None} 

is_path_exist = os.path.isdir("./appdata/geophysics/faults/FaultNet/")
if is_path_exist is False:
    st.stop()
else:
    from appdata.geophysics.faults.FaultNet.utils import prediction

    with st.expander("🟢 Step 1 - this APP requires to load your seismic into RAM"):
        st.info("Here is your original seismic on your disk.")

        step1_viz = VISUALIZATION(seismic, st.session_state.seismic_type)
        step1_viz.viz_data_3d(seismic, key=10, is_fspect=False)

        st.info("Save your precious RAM by cropping the volume here. Use the sliders to crop the volume in 3 dimensions.")

        st.session_state[MODULE_NAME]['numpy_data'], st.session_state[MODULE_NAME]['cropped_info'] = \
            crop_and_load_volume(seismic, st.session_state[MODULE_NAME]['numpy_data'], \
            st.session_state[MODULE_NAME]['cropped_info'])

        if st.session_state[MODULE_NAME]['numpy_data'] is not None:    
            st.info("Here is the cropped seismic on your RAM.")
            step1_crop_viz = VISUALIZATION(st.session_state[MODULE_NAME]['numpy_data'] , st.session_state.seismic_type)
            step1_crop_viz.viz_data_3d(st.session_state[MODULE_NAME]['numpy_data'] , key = 20, is_fspect=False)

    with st.expander("🟢 Step 2 - predict faults with machine learning"):
        st.info("Here you can select several different weights for computing faults. Try out different ones to find the best for your project.")

        inference_form = st.form("Inference")
        weight_file_list = sorted(find_files_in_directory(r'appdata/geophysics/faults/FaultNet/network/', '.pt'))
        weight_selected = inference_form.selectbox(
            'Available weights',
            (weight_file_list))

        if (len(weight_file_list) == 0):
            st.error(''' Here's the problem, it seems like there is no weights in the folder.
            Please download the repo from https://github.com/douyimin/FaultNet and place it in appdata/geophysics/faults/
            ''')
        inference_submit = inference_form.form_submit_button("Submit")
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if inference_submit:

            #TODO may be two times memory allocation
            numpy_data = st.session_state[MODULE_NAME]['numpy_data'].get_cube()
            model = torch.jit.load('appdata/geophysics/faults/FaultNet/network/'+weight_selected).to(device)
            if device.type != 'cpu': model = model.half()

            print("Loaded model from disk")
            # numpy_data = std_mean_normalization(numpy_data)
            print (numpy_data.shape)
            ptrch = FaultNet(model, device)
            _ , predict = predict_with_mask(ptrch, numpy_data.T)
            #predict = prediction(model, numpy_data.T, device)

            st.session_state[MODULE_NAME]['numpy_result'] = Numpy3D(100*predict.T)
            st.session_state[MODULE_NAME]['is_predicted'] = True
        if st.session_state[MODULE_NAME]['is_predicted']:
            step2_viz = VISUALIZATION(st.session_state[MODULE_NAME]['numpy_data']  , st.session_state.seismic_type)
            step2_viz.viz_sidebyside_3d(st.session_state[MODULE_NAME]['numpy_data'], st.session_state[MODULE_NAME]['numpy_result'] , key=30)


    with st.expander("🟢 Step 3 - save the computed fault probability cube"):
        st.write("")
        if st.session_state[MODULE_NAME]['is_predicted']:
            st.session_state[MODULE_NAME]['step3_status'] = save_data_form(st.session_state[MODULE_NAME], seismic, st.session_state[MODULE_NAME]['numpy_result'].get_cube(), st.session_state[MODULE_NAME]['step3_status'])
            st.info(st.session_state[MODULE_NAME]['step3_status'])

sidebar()
