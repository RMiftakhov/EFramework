# conda install -c anaconda keras-gpu or conda install -c conda-forge keras
# conda install -c anaconda tensorflow-gpu or conda install -c conda-forge tensorflow 
# conda install -c anaconda scikit-image
#skimage'

from ast import Assert
import streamlit as st
from custom_blocks import sidebar, crop_and_load_volume, save_data_form
from visualization_helpers import VISUALIZATION
from data_classes import Numpy3D
from utils import find_files_in_directory, predict_with_mask, std_mean_normalization

from keras.models import load_model
from appdata.geophysics.faults.faultSeg.unet3 import *
from appdata.geophysics.faults.faultSeg.unet3 import cross_entropy_balanced

st.markdown("### 🛈 FaultSeg3D: using synthetic datasets to train an end-to-end convolutional neural network for 3D seismic fault segmentation by Xinming Wu (Links: [Paper](https://library.seg.org/doi/10.1190/geo2018-0646.1), [GitHub](https://github.com/xinwucwp/faultSeg))")

with st.expander("Abstract"):
    st.markdown(""" ### Authors: Xinming Wu, Luming Liang, Yunzhi Shi, and Sergey Fomel

Delineating faults from seismic images is a key step for seismic structural interpretation, reservoir characterization, and well placement. In conventional methods, faults are considered as seismic reflection discontinuities and are detected by calculating attributes that estimate reflection continuities or discontinuities. We consider fault detection as a binary image segmentation problem of labeling a 3D seismic image with ones on faults and zeros elsewhere. We have performed an efficient image-to-image fault segmentation using a supervised fully convolutional neural network. To train the network, we automatically create 200 3D synthetic seismic images and corresponding binary fault labeling images, which are shown to be sufficient to train a good fault segmentation network. Because a binary fault image is highly imbalanced between zeros (nonfault) and ones (fault), we use a class-balanced binary cross-entropy loss function to adjust the imbalance so that the network is not trained or converged to predict only zeros. After training with only the synthetic data sets, the network automatically learns to calculate rich and proper features that are important for fault detection. Multiple field examples indicate that the neural network (trained by only synthetic data sets) can predict faults from 3D seismic images much more accurately and efficiently than conventional methods. With a TITAN Xp GPU, the training processing takes approximately 2 h and predicting faults in a 128×128×128 seismic volume takes only milliseconds.""")

col1, col2 = st.columns(2)
with col1: 
    st.markdown("## My video about the paper")
    st.video('https://youtu.be/OLWemwDcBp0')
with col2:
    st.markdown("## My video Unboxing the GitHub")
    st.video('https://youtu.be/18ovlxGEWBk')

st.markdown("## 🔰 Install external dependencies")
with st.expander("GPU version"):
    st.code('''conda install -c anaconda keras-gpu
conda install -c anaconda tensorflow-gpu 
conda install -c anaconda scikit-image''')
with st.expander("CPU version"):
    st.code('''conda install -c conda-forge keras 
conda install -c conda-forge tensorflow 
conda install -c anaconda scikit-image''')

st.markdown("## ✨ Here we go with the APP")

if "seismic" not in st.session_state or st.session_state.seismic_type=="2D":
        st.error("Please import 3D seismic first")
        st.stop()

seismic = st.session_state.seismic
seismic_type = st.session_state.seismic_type

module_name = 'FaultSeg3D'

# Initialize state

if module_name not in st.session_state:
    st.session_state[module_name] = {"numpy_data" : None,
        "numpy_result" : None, "is_predicted" : False, 'cropped_info' : None , \
        "step1_status" : None, "step2_status" : None, "step3_status" : None} 

with st.expander("🟢 Step 1 - this APP requires to load your seismic into RAM"):
    st.info("Here is your original seismic on your disk.")

    step1_viz = VISUALIZATION(seismic, st.session_state.seismic_type)
    step1_viz.viz_data_3d(seismic, key=10, is_fspect=False)

    st.info("Save your precious RAM by cropping the volume here. Use the sliders to crop the volume in 3 dimensions.")

    st.session_state[module_name]['numpy_data'], st.session_state[module_name]['cropped_info'] = \
        crop_and_load_volume(seismic, st.session_state[module_name]['numpy_data'], \
        st.session_state[module_name]['cropped_info'])

    if st.session_state[module_name]['numpy_data'] is not None:    
        st.info("Here is the cropped seismic on your RAM.")
        step1_crop_viz = VISUALIZATION(st.session_state[module_name]['numpy_data'] , st.session_state.seismic_type)
        step1_crop_viz.viz_data_3d(st.session_state[module_name]['numpy_data'] , key = 20, is_fspect=False)

with st.expander("🟢 Step 2 - predict faults with machine learning"):
    st.info("Here you can select several different weights for computing faults. Try out different ones to find the best for your project.")

    inference_form = st.form("Inference")
    weight_file_list = sorted(find_files_in_directory(r'appdata/geophysics/faults/faultSeg/model/', '.hdf5'))
    weight_selected = inference_form.selectbox(
        'Available weights',
        (weight_file_list))

    if (len(weight_file_list) == 0):
        st.error('''There is no weights in the model folder. 
        Please download the pretrained models from https://drive.google.com/drive/folders/1q8sAoLJgbhYHRubzyqMi9KkTeZWXWtNd
        and place them here EFramework/appdata/geophysics/faults/faultSeg/model.
''')
    inference_submit = inference_form.form_submit_button("Submit")
    if inference_submit:

        #TODO may be two times memory allocation
        numpy_data = st.session_state[module_name]['numpy_data'].get_cube()
        if (weight_selected == "pretrained_model.hdf5"):
            # load json and create model 
            json_file = open('appdata/geophysics/faults/faultSeg/model/model3.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("appdata/geophysics/faults/faultSeg/model/"+weight_selected)
            
            print("Loaded model from disk")

            numpy_data = std_mean_normalization(numpy_data)
            print (numpy_data.shape)
            _ , predict = predict_with_mask(loaded_model, numpy_data, normalize_patch=True)
            st.session_state[module_name]['numpy_result']  = Numpy3D(100*predict)
            st.session_state[module_name]['is_predicted'] = True
        else:
            loaded_model = load_model("appdata/geophysics/faults/faultSeg/model/"+weight_selected, custom_objects={'cross_entropy_balanced': cross_entropy_balanced})
            print("Loaded model from disk")
            numpy_data = std_mean_normalization(numpy_data)
            print (numpy_data.shape)
            _ , predict = predict_with_mask(loaded_model, numpy_data.T)
            st.session_state[module_name]['numpy_result'] = Numpy3D(100*predict.T)
            st.session_state[module_name]['is_predicted'] = True
    if st.session_state[module_name]['is_predicted']:
        step2_viz = VISUALIZATION(st.session_state[module_name]['numpy_data']  , st.session_state.seismic_type)
        step2_viz.viz_sidebyside_3d(st.session_state[module_name]['numpy_data'], st.session_state[module_name]['numpy_result'] , key=30)


with st.expander("🟢 Step 3 - save the computed fault probability cube"):
    if st.session_state[module_name]['is_predicted']:
        st.session_state[module_name]['step3_status'] = save_data_form(st.session_state[module_name], seismic, st.session_state[module_name]['numpy_result'].get_cube(), st.session_state[module_name]['step3_status'])
        st.info(st.session_state[module_name]['step3_status'])


sidebar()