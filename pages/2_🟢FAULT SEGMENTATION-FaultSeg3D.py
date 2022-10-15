# conda install -c anaconda keras-gpu or conda install -c conda-forge keras
# conda install -c anaconda tensorflow-gpu or conda install -c conda-forge tensorflow 
# conda install -c anaconda scikit-image

# @only for sepredict3D files


from ast import Assert
import streamlit as st
import os
from custom_blocks import sidebar, crop_and_load_volume, save_data_form
from visualization_helpers import VISUALIZATION
from data_classes import Numpy3D
from utils import find_files_in_directory, predict_with_mask, std_mean_normalization
import numpy as np

st.markdown("## FaultSeg3D: using synthetic datasets to train an end-to-end convolutional neural network for 3D seismic fault segmentation by Xinming Wu (Links: [Paper](https://library.seg.org/doi/10.1190/geo2018-0646.1), [GitHub](https://github.com/xinwucwp/faultSeg))")

with st.expander("Abstract"):
    st.write("""""")
col1, col2 = st.columns(2)
with col1: 
    st.markdown("## My video about the paper")
    st.video('https://youtu.be/OLWemwDcBp0')
with col2:
    st.markdown("## My video Unboxing GitHub")
    st.video('https://youtu.be/18ovlxGEWBk')

st.markdown("## ✨ Here we go with the APP")

if "seismic" not in st.session_state or st.session_state.seismic_type=="2D":
        st.error("Please import 3D seismic first")
        st.stop()

seismic = st.session_state.seismic
seismic_type = st.session_state.seismic_type


# Initialize state

if 'FaultSeg3D' not in st.session_state:
    st.session_state['FaultSeg3D'] = {"numpy_data" : None,
        "numpy_result" : None, "is_predicted" : False, 'cropped_info' : None , \
        "step1_status" : None, "step2_status" : None, "step3_status" : None} 

with st.expander("🟢 Step 1 - To work with this APP we need to load seismic into RAM"):
    st.subheader("To save your precious ram, you can now crop the volume here")

    Viz = VISUALIZATION(seismic, st.session_state.seismic_type)
    Viz.visualize_seismic_3D(seismic, key=10, is_fspect=False)

    st.subheader("Cropping the volume")

    st.session_state['FaultSeg3D']['numpy_data'], st.session_state['FaultSeg3D']['cropped_info'] = \
        crop_and_load_volume(seismic, st.session_state['FaultSeg3D']['numpy_data'], \
        st.session_state['FaultSeg3D']['cropped_info'])

    if st.session_state['FaultSeg3D']['numpy_data'] is not None:    
        Viza = VISUALIZATION(st.session_state['FaultSeg3D']['numpy_data'] , st.session_state.seismic_type)
        Viza.visualize_seismic_3D(st.session_state['FaultSeg3D']['numpy_data'] , key = 20, is_fspect=False)

    # if range changes what to do?
    #TODO change corner point on plot

with st.expander("🟢 Step 2 - Calculation"):
    st.subheader("Here we select the weights for computation")
    inference_form = st.form("Inference")
    weight_file_list = sorted(find_files_in_directory(r'appdata/faultSeg/model/', '.hdf5'))
    weight_selected = inference_form.selectbox(
        'Available weights',
        (weight_file_list))
    inference_submit = inference_form.form_submit_button("Submit")
    if inference_submit:

        from keras.models import load_model
        from appdata.faultSeg.unet3 import *
        from appdata.faultSeg.unet3 import cross_entropy_balanced
        import numpy as np

        #TODO may be two time memory allocation
        numpy_data = st.session_state['FaultSeg3D']['numpy_data'].get_cube()
        if (weight_selected == "pretrained_model.hdf5"):
            # load json and create model 
            json_file = open('appdata/faultSeg/model/model3.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("appdata/faultSeg/model/"+weight_selected)
            
            print("Loaded model from disk")

            numpy_data = std_mean_normalization(numpy_data)
            print (numpy_data.shape)
            _ , predict = predict_with_mask(loaded_model, numpy_data, normalize_patch=True)
            st.session_state['FaultSeg3D']['numpy_result']  = Numpy3D(predict)
            st.session_state['FaultSeg3D']['is_predicted'] = True
        else:
            loaded_model = load_model("appdata/faultSeg/model/"+weight_selected, custom_objects={'cross_entropy_balanced': cross_entropy_balanced})
            print("Loaded model from disk")
            numpy_data = std_mean_normalization(numpy_data)
            print (numpy_data.shape)
            _ , predict = predict_with_mask(loaded_model, numpy_data.T)
            st.session_state['FaultSeg3D']['numpy_result'] = Numpy3D(predict.T)
            st.session_state['FaultSeg3D']['is_predicted'] = True

if (st.session_state['FaultSeg3D']['is_predicted']):
    Viza = VISUALIZATION(st.session_state['FaultSeg3D']['numpy_result']  , st.session_state.seismic_type)
    Viza.visualize_sidebyside_3D(st.session_state['FaultSeg3D']['numpy_data'], st.session_state['FaultSeg3D']['numpy_result'] , key=30)


with st.expander("🟢 Step 3 - Save the results"):
    st.session_state['FaultSeg3D']['step3_status'] = save_data_form(st.session_state['FaultSeg3D'], seismic, st.session_state['FaultSeg3D']['numpy_result'].get_cube(), st.session_state['FaultSeg3D']['step3_status'])
    st.info(st.session_state['FaultSeg3D']['step3_status'])





st.warning("Add: - fix bug with viz for fseg, - turn off axes in viz - add wait for process, - add terminal output - add export to numpy and segy - do profiling")

sidebar()