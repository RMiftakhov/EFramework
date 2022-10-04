import streamlit as st
import os
from custom_blocks import sidebar

st.markdown("## 2D/3D Seismic Facies Prediction")

with st.expander("Abstract"):
    st.write("""""")
    
st.markdown("Some of my videos about the subject")
col1, col2 = st.columns(2)
with col1: 
    st.video('https://youtu.be/sfNAkJNAiNY')
with col2:
    st.video('https://youtu.be/q6quHKejC0o')


st.markdown("## ✨ Here we go with the APP")

sidebar()