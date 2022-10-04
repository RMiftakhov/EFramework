import streamlit as st
import os
from custom_blocks import sidebar

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

sidebar()