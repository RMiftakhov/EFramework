import streamlit as st
import numpy as np
from PIL import Image
from utils import img_to_html_custom, img_to_bytes, save_to_numpy, save_to_segy
from data_classes import Numpy3D
import os


def footer():
    st.markdown('---')
    st.markdown('# 🔥 Geoplat AI - AI based integrated geoscience solutions')
    st.markdown('#### If you are looking for a commercial product that can improve your *GEOSCIENCE* workflow with AI/ML, then try out [Geoplat AI](https://geoplat.ai).')
    st.markdown(""" \n
    🌐 Website: [Geoplat.ai](https://geoplat.ai) \n
    📧 Email: r.miftakhov@geoplat.com""")
    image = Image.open('images/GeoplatMarket.jpg')
    st.image(image)
    st.video("https://youtu.be/6Fh18AoC4qE")


def sidebar():
    with st.sidebar:
        st.markdown("""
        - **Seismic Type:** {}
        - **Seismic Name:** {}
        """.format(st.session_state.seismic_type if 'seismic_type' in st.session_state else "--", \
            os.path.basename(st.session_state.filename) if 'filename' in st.session_state else "--"))
        st.markdown("""
        &nbsp;
        """)
        st.markdown(img_to_html_custom('images/GeoplatLogo.png', 1636/10, 376/10, 'https://www.geoplat.ai'), unsafe_allow_html=True)
        st.markdown(""" ```
        Ruslan Miftakhov 
        r.miftakhov@geoplat.com
        """)

def crop_and_load_volume(data, converted_to_numpy3d, cropped_info):
    with st.form("Cropping"):
        col1, col2, col3 = st.columns(3)
        inlines_indx = col1.slider( 'Select a range for Inlines',
        0, data.get_n_ilines()-1, (0, data.get_n_ilines())) 

        xlines_indx = col2.slider( 'Select a range for Xlines',
        0, data.get_n_xlines()-1, (0, data.get_n_xlines())) 
        
        zslice_indx = col3.slider( 'Select a range for Zslice',
        0, data.get_n_zslices()-1, (0, data.get_n_zslices())) 
        
        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            cropped_info = np.array([[inlines_indx[0], inlines_indx[1]], \
                [xlines_indx[0], xlines_indx[1]], \
                    [zslice_indx[0], zslice_indx[1]]])
            np_data = data.cropped_numpy(inlines_indx[0], inlines_indx[1], \
            xlines_indx[0], xlines_indx[1],\
                zslice_indx[0], zslice_indx[1])
            converted_to_numpy3d = Numpy3D(np_data)
            col1, col2, col3 = st.columns(3)
            col1.info(f"Number of Inlines [{inlines_indx[1]-inlines_indx[0]}]")
            col2.info(f"Number of Xlines [{xlines_indx[1] - xlines_indx[0]}]")
            col3.info(f"Time [{zslice_indx[1]-zslice_indx[0]}]")
            st.success('Volume is loaded')
    return converted_to_numpy3d, cropped_info

def save_data_form(session_state, seismic, numpy_data, status):
    with st.form("save_data"):
        col1, col2, col3 = st.columns(3)
        path = col1.text_input("Path to Folder")
        file_name = col2.text_input("File Name")
        format = col3.radio( "What format? ", ('SEGY', 'NUMPY_SUBSET'))
        submitted = st.form_submit_button("Save")
        if submitted:
            if format == "SEGY":
                save_to_segy(seismic, path+file_name, numpy_data, session_state)
            else:
                save_to_numpy(path+file_name, numpy_data)
                # option = st.radio( "Option", ('Save subset', 'Save in original dimensions - It will create the volume in RAM. Are you sure?'))
                # if st.form_submit_button("Save "):
                #     if option == 'Save subset':
                #         status = None
                #         save_to_numpy(path+file_name, numpy_data)
                #     else:
                #         status = "Save in original dimensions"
    return status