import streamlit as st
import os
from custom_blocks import sidebar

st.markdown("## Reconstruction of Missing Seismic Data")

with st.expander("Abstract"):
    st.write("""""")
    
st.markdown("Some of my videos about the subject")
col1, col2 = st.columns(2)
with col1: 
    st.video('https://youtu.be/QZNgY8rRj40')



st.markdown("## ✨ Here we go with the APP")

sidebar()