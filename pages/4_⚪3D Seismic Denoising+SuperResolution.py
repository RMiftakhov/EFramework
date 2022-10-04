import streamlit as st
import os
from custom_blocks import sidebar

st.markdown("## 3D Seismic Denoising+SuperResolution")

with st.expander("Abstract"):
    st.write("""""")
    
st.markdown("Some of my videos about the subject")
col1, col2 = st.columns(2)
with col1: 
    st.video('https://youtu.be/Pr3wLpKtMxk')
with col2:
    st.video('https://youtu.be/44NRwabN1NY')


st.markdown("## ✨ Here we go with the APP")

sidebar()