import streamlit as st
from PIL import Image
from custom_blocks import footer, sidebar
# from utils import img_to_html

from pathlib import Path
import base64

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

col1, col2 = st.columns(2)
with col1:
    st.markdown("# 🚀 Hey friends — I’m Ruslan.")
      
col1, col2 = st.columns(2)
with col1:
    
    st.markdown("As the Chief Technical Officer of [Geoplat](https://geoplat.ai), my passion for AI and its applications in the oil and gas industry drives my work. With years of experience in learning, teaching, developing, and launching AI-based solutions for geoscience and petroleum engineering, I am dedicated to advancing the use of technology in our industry.", unsafe_allow_html=True)
    st.markdown("In 2021, I launched a YouTube channel to share my knowledge and expertise in all aspects of the oil and gas industry, including drilling, exploration, production, and reservoir management. I also provide hands-on instruction on AI/ML programming using Python and Tensorflow/PyTorch.")
    st.markdown("I am continuously updating my website with the latest resources, including  [Open-Datasets](https://ruslanmiftakhov.com/blog#!/tfeeds/937619882821/c/Datasets) and [Open-Tools](https://ruslanmiftakhov.com/blog#!/tfeeds/937619882821/c/Tools), to help others in the industry stay informed and stay ahead of the curve.")
with col2:
    st.video('https://youtu.be/n5wsGcQ3tAc')


col11, col22, col33, col44 = st.columns(4)
col33.markdown('## YouTube '+img_to_html('images/Square-YouTube-Logo-PNG-1024x1024.png', 'https://www.youtube.com/channel/UC1HyCbG5SO4hC7b_Ddl8cGg?sub_confirmation=1'), unsafe_allow_html=True)
col44.markdown('## LinkedIn '+img_to_html('images/linkedin.png', 'https://www.linkedin.com/in/ruslan-miftakhov/'), unsafe_allow_html=True)


st.markdown('---')
st.markdown('### What is this app about❓')

st.markdown('Introducing EFramework, an open-source development tool designed to democratize the use of AI in the oil and gas industry. Our app serves as a framework to bring together some of the best open-source AI/ML tools in the industry, making them accessible to a wider range of users. With EFramework, there is no need to be a skilled programmer to run open-source ML tools on your own data.\
We provide easy-to-use code that allows you to input example data (or your own) and compute it, saving the results in each application. This means you can easily explore and experiment with different AI tools, without needing extensive programming knowledge. EFramework is a valuable resource for professionals and researchers in the oil and gas industry, as well as anyone interested in learning more about the potential of AI in this field.')

col11, col22, col33 = st.columns(3)
col11.markdown("""**Open AI/ML Algorithms** 
- 2D Self Supervised Denoising by [Claire](https://cebirnie92.github.io/) (Links: [GitHub](https://github.com/swag-kaust/Transform2022_SelfSupervisedDenoising) and [YouTube](https://youtu.be/d9yv90-JCZ0))
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

st.markdown(''' *The whole list of intended AI/ML applications:*
| Geophysics | Petrophysics | Drilling Engineering | Reservoir Engineering | Production Engineering |
| --- | --- | --- | --- | --- | 
| 2D Self Supervised Denoising | Well-Log Lithology/Property Prediction | ROP Prediction/Optimization | Well-to-Well Interference | Predicting Well Rate Potential
| 2D/3D Seismic Fault Segmentation | Well-Log Outlier detection | Drillstring-Vibration Prediction | Proxy Reservoir Simulation | Virtual Rate Metering
| 2D/3D Seismic Denoising+SuperResolution | Well-Log Synthesis | Lost-Circulation Prediction | Reservoir Optimization | Predicting Well Failures 
| 2D/3D Seismic Facies Prediction | Well-to-Well correlation | DPS Incident Detection | PVT | Predicting Critical Oil Rate
| 3D Salt/Karst Delineation | | Real-Time Gas-Influx Detection | Decline Curve Analisys | Recommending Optimum Well Spacing in NFRs
| Reconstruction of Missing Seismic Data | | Drillstring-Washout Detection | | Predicting Threshold Reservoir Radius
| Neural Network for Acoustic Impedance prediction | | Abnormal-Drilling Detection | | Identification of Downhole Conditions in Sucker Rod Pumped Wells
| First-Break Picking | | Drill-Fluid Design | | Prediction of Multilateral Inflow Control Valve Flow Performance
''')


footer()
sidebar()
