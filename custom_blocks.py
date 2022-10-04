import streamlit as st
from PIL import Image
from utils import img_to_html_custom, img_to_bytes
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
        col1, col2 = st.columns(2)
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

