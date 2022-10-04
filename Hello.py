import streamlit as st
from PIL import Image
from custom_blocks import footer, sidebar
from utils import img_to_html

st.set_page_config(layout="wide")

col1, col2 = st.columns(2)
with col1:
    st.markdown("# 🚀 Hey friends — I’m Ruslan.")
      
col1, col2 = st.columns(2)
with col1:
    
    st.markdown("I am CTO at [Geoplat](https://geoplat.ai), and I am passionate about AI and its applications in the oil and gas industry. I have spent many years learning, teaching, developing, and launching AI-based solutions for geoscience and petroleum engineering.", unsafe_allow_html=True)
    st.markdown("In 2021, I started a YouTube channel where I cover all aspects of our industry (drilling, exploration, production, and reservoir) as well as hands-on AI/ML programming using Python and Tensorflow/PyTorch.")
    st.markdown("I am updading from time to time on my website: the list of [Open-Datasets](https://ruslanmiftakhov.com/blog#!/tfeeds/937619882821/c/Datasets) and [Open-Tools](https://ruslanmiftakhov.com/blog#!/tfeeds/937619882821/c/Tools).")
with col2:
    st.video('https://youtu.be/6owwDkEhkgg')


col11, col22, col33, col44 = st.columns(4)
col33.markdown('## YouTube '+img_to_html('images/Square-YouTube-Logo-PNG-1024x1024.png', 'https://www.youtube.com/channel/UC1HyCbG5SO4hC7b_Ddl8cGg?sub_confirmation=1'), unsafe_allow_html=True)
col44.markdown('## LinkedIn '+img_to_html('images/linkedin.png', 'https://www.linkedin.com/in/ruslan-miftakhov/'), unsafe_allow_html=True)

st.markdown('---')
st.markdown('### What this app about❓')

st.markdown('This app serves as a framework to include some of the best **open-source** O&G AI/ML tools. It democritises the use of AI tools by lifting a need to be a skilled programmer to run open-source ML tools on your own data. \
In most cases, I will include code that allows you to enter the example data (or your own), compute it, and save the results in each application.')

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