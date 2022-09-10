import streamlit as st
from PIL import Image
import base64
from pathlib import Path
import os

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path, link=''):
    img_html = "<a href='{}'><img src='data:image/png;base64,{}' class='img-fluid' width='64' height='64'>".format(
        link,
        img_to_bytes(img_path)
    )
    return img_html

st.set_page_config(layout="wide")

with st.sidebar:
    st.header('Seismic')
    st.write("- ")

col1, col2 = st.columns(2)
with col1:
    st.markdown("# 🚀 Hey friends — I’m Ruslan.")
      
col1, col2 = st.columns(2)
with col1:
    
    st.markdown("I am **fascinated** by artificial intelligence and its applications in the oil and gas industry. I've spent the **5+** years learning, teaching, building, and deploying AI-based solutions for seismic interpretation, geological modeling, and reservoir simulation.", unsafe_allow_html=True)
    st.markdown("In 2021, I started a YouTube channel where I cover all aspects of our industry (drilling, exploration, production, and reservoir) as well as hands-on AI/ML programming using Python and Tensorflow/PyTorch")
    col1.markdown("The list of [Datasets](https://ruslanmiftakhov.com/blog#!/tfeeds/937619882821/c/Datasets) and [Tools](https://ruslanmiftakhov.com/blog#!/tfeeds/937619882821/c/Tools) I am updading from time to time on my website")
with col2:
    st.video('https://youtu.be/6owwDkEhkgg')



col11, col22, col33, col44 = st.columns(4)
col33.markdown('## YouTube '+img_to_html('images/YouTube.jpg', 'https://www.youtube.com/channel/UC1HyCbG5SO4hC7b_Ddl8cGg?sub_confirmation=1'), unsafe_allow_html=True)
col44.markdown('## LinkedIn '+img_to_html('images/linkedin.png', 'https://www.linkedin.com/in/ruslan-miftakhov/'), unsafe_allow_html=True)

st.markdown('---')
st.markdown('### What this app about❓')


st.markdown('This app includes some of the best **open-source** O&G AI/ML tools so that you may test them out on your own data. \
    In most cases, I will include code that allows you to input the sample data *(or your own)*, perform the computation, and save the results in each application.')

col11, col22, col33 = st.columns(3)
col11.markdown("""**Open AI/ML Algorithms** 
- 2D Self Supervised Denoising by [Claire](https://cebirnie92.github.io/) (Links: [GitHub](https://github.com/swag-kaust/Transform2022_SelfSupervisedDenoising) and [YouTube](https://youtu.be/d9yv90-JCZ0))✔️ 
- 3D Seismic Fault Segmentation 
- 3D Seismic Denoising+SuperResolution 
- 2D/3D Seismic Facies Prediction
- 3D Salt/Karst Delineation
- Reconstruction of Missing Seismic Data
- Neural Network for Acoustic Impedance prediction
- Well-Log Lithology Prediction
- Well-Log Synthesis
- Well-to-well correlation
- Centimeter-Scale Lithology and Facies Prediction using Core Images
- First-Break Picking
- and many more... """)
col22.markdown("""**Import**  
- 2D/3D Post-Stacked Seismic import with SegyIO ✔️
- 2D Post-Stacked Seismic import as Numpy Array ✔️ 
- 3D Post-Stacked Seismic import as Numpy Array
- Well-Log data
- Well Trajectory data
- Surface data (attributes, horizon, ...)
- ...""")
col33.markdown("""**Visualization** 
- 2D Seismic with user-defined colormap ✔️
- 3D Interactive Seismic(section view) with user-defined colormap ✔️
- Spectrum Frequency Plot ✔️
- ...""")
 


image = Image.open('images/Check Out My AIML Solutions.png')
col11.image(image)

with st.sidebar:
    col1, col2 = st.columns(2)
    st.markdown("""
    - **Seismic Type:** {}
    - **Seismic Name:** {}
    """.format(st.session_state.seismic_type if 'seismic_type' in st.session_state else "--", \
        os.path.basename(filename) if 'filename' in st.session_state else "--"))
