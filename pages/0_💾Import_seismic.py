import streamlit as st
import segyio
import plotly.express as px
from visualization_helpers import VISUALIZATION
import os
from data_classes import SegyIO3D, SegyIO2D, Numpy2D

def get_segy_header(file_name):
    f = segyio.open(file_name, ignore_geometry = True)
    return segyio.tools.wrap(f.text[0])
    
def get_inline_xline_position(file_name):
    f = segyio.open(file_name, ignore_geometry = True)
    _ntraces    = len(f.trace)
    _inlines    = []
    _crosslines = []

    for i in range(_ntraces):
        headeri = f.header[i]
        _inlines.append(headeri[segyio.su.iline])
        _crosslines.append(headeri[segyio.su.xline])

    print(f'{_ntraces} traces')
    print(f'first 10 inlines: {_inlines[:10]}')
    print(f'first 10 crosslines: {_crosslines[:10]}')

    #The loop variable i is only used to look up the the right header, but segyio can manage this for u
    return _inlines, _crosslines

st.markdown("# 💾 Import Seismic from File")
data_option= st.radio(
     "🔘 There are a few options:",
     ('Segy2D', 'Segy3D', 'Numpy2D', 'Numpy3D'), horizontal=True)

if data_option == 'Segy3D' or data_option == 'Segy2D':
    st.title("Import Seismic in SEGY format")
    if 'filename' not in st.session_state:
        st.session_state.filename = ''

    if 'inline_byte' not in st.session_state:
        # These are default setting for most of the SEGY files
        st.session_state.inline_byte = 189
        st.session_state.xline_byte = 193

    if 'failed_seismic' not in st.session_state:
        # This flag is raised if something went wrong
        st.session_state.failed_seismic = False

    filename = st.text_input('Please pass here the whole path like: C:/Ichthys 3D.segy')

    # If the user chooses the other filename, then we have to reset params
    if filename != st.session_state.filename:
        st.session_state.inline_byte = 189
        st.session_state.xline_byte = 193
        st.session_state.failed_seismic = False

    st.session_state.filename = filename
    st.write('The selected file is: ', filename)

    tab1, tab2, tab3 = st.tabs(["Check header", "Import the file", "Troubleshoot"])

    if filename:
        with tab1:
            st.code(get_segy_header(filename))
        with tab3:
            if st.button("Plot inline/crossline as scatter plot"):
                with st.spinner('Wait for it...'):

                    inlines, crosslines = get_inline_xline_position(filename)
                    # Plot the inline and crossline as a scatter plot
                    fig = px.scatter(x=crosslines, y=inlines)
                    fig.update_traces(marker=dict(size=5))     
                    st.write(fig)
        
        with tab2:
            with st.expander("Inline/Xline fields in the trace headers"):
                with st.form("my_form"):
                    st.session_state.inline_byte = st.number_input('Inline', value=int(st.session_state.inline_byte) , format='%i')
                    st.session_state.xline_byte = st.number_input('Xline', value=int(st.session_state.xline_byte), format='%i')
                    submitted = st.form_submit_button("Read File")
            try:
                segyfile = SegyIO2D(filename) if data_option=="Segy2D" else SegyIO3D(filename, st.session_state.inline_byte, st.session_state.xline_byte)
                st.session_state.seismic_type = "2D" if data_option=="Segy2D" else "3D"

                
                Viz = VISUALIZATION(segyfile, st.session_state.seismic_type)
                Viz.visualize_seismic_2D(segyfile.get_iline(), is_fspect=True) if data_option=="Segy2D" else Viz.visualize_seismic_3D(segyfile, is_fspect=True)
                

                st.session_state.seismic = segyfile
                st.success('It appears that the survey is correctly read. AI/ML methods are now available in this app!')
                        
            except RuntimeError as err: 
                st.session_state.failed_seismic = True
                st.write("Oops!  Something went wrong.  Try again...", err)

elif (data_option == 'Numpy2D'):
    st.title("Import Seismic As Numpy Array")

    if 'filename' not in st.session_state:
        st.session_state.filename = ''
    np_text = 'Please pass here the whole path like: C:/Ichthys 3D.npy'
    filename = st.text_input(np_text, st.session_state.filename)

    st.session_state.filename = filename
    st.write('The selected file is: ', filename)
    if filename:
        try:
            seismic = SegyIO2D(filename) if data_option=="Segy2D" else Numpy2D(filename)
            # make input file devisable by 4
            seismic.make_axis_devisable_by(4)
            st.session_state.seismic = seismic
            st.session_state.seismic_type = "2D"


            Viz = VISUALIZATION(seismic, st.session_state.seismic_type)
            Viz.visualize_seismic_2D(seismic.get_iline(), True)

            st.success('It appears that the survey is correctly read. AI/ML methods are now available in this app!')
        except RuntimeError as err: 
            st.write("Oops!  That was no valid number.  Try again...", err)
else:
    st.error("Not implemented.")
   
with st.sidebar:
    col1, col2 = st.columns(2)
    st.markdown(f"""
    - **Seismic Type:** {st.session_state.seismic_type if 'seismic_type' in st.session_state else "--"}
    - **Seismic Name:** {os.path.basename(filename) if 'filename' in st.session_state else "--"}
    """ )
