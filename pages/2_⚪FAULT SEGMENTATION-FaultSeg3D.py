import streamlit as st
import os

with st.sidebar:
    col1, col2 = st.columns(2)
    st.markdown("""
    - **Seismic Type:** {} 
    - **Seismic Name:** {}
    """.format(st.session_state.seismic_type if 'seismic_type' in st.session_state else "--", \
        os.path.basename(st.session_state.filename) if 'filename' in st.session_state else "--"))

st.markdown("## FaultSeg3D: using synthetic datasets to train an end-to-end convolutional neural network for 3D seismic fault segmentation by Xinming Wu (Links: [Paper](https://library.seg.org/doi/10.1190/geo2018-0646.1), [GitHub](https://github.com/xinwucwp/faultSeg))")

with st.expander("Abstract - The potential of self-supervised networks for random noise suppression in seismic data"):
    st.write("""Noise suppression is an essential step in any seismic processing workflow. A portion of this noise, particularly in land datasets, presents itself as random noise. In recent years, neural networks have been successfully used to denoise seismic data in a supervised fashion. However, supervised learning always comes with the often unachievable requirement of having noisy-clean data pairs for training. Using blind-spot networks, we redefine the denoising task as a self-supervised procedure where the network uses the surrounding noisy samples to estimate the noise-free value of a central sample. Based on the assumption that noise is statistically independent between samples, the network struggles to predict the noise component of the sample due to its randomnicity, whilst the signal component is accurately predicted due to its spatio-temporal coherency. Illustrated on synthetic examples, the blind-spot network is shown to be an efficient denoiser of seismic data contaminated by random noise with minimal damage to the signal; therefore, providing improvements in both the image domain and down-the-line tasks, such as inversion. To conclude the study, the suggested approach is applied to field data and the results are compared with two commonly used random denoising techniques: FX-deconvolution and Curvelet transform. By demonstrating that blind-spot networks are an efficient suppressor of random noise, we believe this is just the beginning of utilising self-supervised learning in seismic applications.""")
col1, col2 = st.columns(2)
with col1: 
    st.markdown("## My video about the paper")
    st.video('https://youtu.be/OLWemwDcBp0')
with col2:
    st.markdown("## My video Unboxing GitHub")
    st.video('https://youtu.be/18ovlxGEWBk')

st.markdown("## ✨ Here we go with the APP")