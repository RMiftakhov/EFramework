from random import randint
import plotly.express as px
import streamlit as st
import numpy as np
from utils import fspectra, signaltonoise_dB, signaltonoise
import plotly.graph_objects as go
import tempfile
from streamlit_image_comparison import image_comparison


class VISUALIZATION:
    def __init__(self, _seismic_data, seismic_type):
        self._seismic_type = seismic_type
        self._vm = _seismic_data.get_vm()
        self._n_samples = _seismic_data.get_n_zslices()
        self._n_il = _seismic_data.get_n_ilines()
        self._n_xl = _seismic_data.get_n_xlines()
        self._sample_rate = _seismic_data.get_sample_rate()
        self._cmap_option = "gray"

    def plot_slice(self,_segyfile, inline_indx, iline_old, xline_indx, xline_old, time_indx, t_old, cmap, vm):
        seis = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                        [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
                    ], dtype=np.uint8)
        seis_plt = px.imshow(seis)

        if np.abs(iline_old - inline_indx) > 0:
            iline_old = inline_indx
            seis = _segyfile.get_iline(inline_indx).T
            seis_plt = plot_inl(seis, cmap, -vm, vm)
        elif np.abs(xline_old - xline_indx) > 0:
            xline_old = xline_indx
            seis = _segyfile.get_xline(xline_indx).T
            seis_plt =  plot_xln(seis, cmap, -vm, vm)
        elif np.abs(t_old - time_indx) > 0:
            t_old = time_indx
            seis = _segyfile.get_zslice(time_indx).T
            seis_plt = plot_top(seis,  cmap, -vm, vm)
        seis_plt.update_layout(
            plot_bgcolor='rgba(0,0,0,0)')
        return seis, seis_plt, iline_old, xline_old, t_old

    def plotly_color_select(self):
        colorscales = ["gray", "Greys", "RdBu", "RdGy"] #px.colors.named_colorscales()
        option = st.selectbox(
            'Color Map',
            (colorscales))
        # st.write('You selected:', option)
        return option

    def visualize_seismic_3D(self, seismic_data, is_fspect, is_show_metrics=True):     
        vm, n_samples, n_il, n_xl = self._vm, self._n_samples, self._n_il, self._n_xl

        if is_show_metrics:
            col1, col2, col3, col4, col5 = st.columns(5)
            col2.metric("Number of Samples", n_samples)
            col3.metric("Number of Inline", n_il)
            col4.metric("Number of Xline", n_xl)
            col5.metric("Sample Rate", self._sample_rate)
            with col1:
                self._cmap_option = self.plotly_color_select()

        if 'keys' not in st.session_state:
            st.session_state.keys = 1
            st.session_state.iline_old = 0
            st.session_state.xline_old = 0
            st.session_state.t_old = 0

        col1, col2, col3 = st.columns(3)
        with col1:
            inline_indx = st.slider('Inline', 0, n_il-1, n_il//2)
        with col2:
            xline_indx = st.slider('Xline', 0, n_xl-1, n_xl//2)
        with col3:
            time_indx = st.slider('Time', 0, n_samples-1, n_samples//2)
        
        seis, seis_3d_plot, st.session_state.iline_old, st.session_state.xline_old, st.session_state.t_old = self.plot_slice(seismic_data, inline_indx, st.session_state.iline_old, xline_indx, st.session_state.xline_old, time_indx, st.session_state.t_old, cmap=self._cmap_option, vm=vm)
        st.write(seis_3d_plot)
        if is_fspect:
            fspect_plt = self.plot_fspectra(seis, 'Original')
            st.write(fspect_plt)

    def visualize_seismic_2D(self, seismic_data, is_fspect, is_show_metrics=True):
        vm, n_samples, n_il, n_xl = self._vm, self._n_samples, self._n_il, self._n_xl

        if is_show_metrics:
            col1, col2, col3, col4 = st.columns(4)
            col2.metric("Number of Samples", n_samples)
            col3.metric("Number of Inline", n_il)
            col4.metric("Number of Xline", n_xl)
            with col1:
                self._cmap_option = self.plotly_color_select()

        seis_plt = plot_seis(seismic_data,cmap=self._cmap_option, vmin=-vm, vmax=vm)
        st.write(seis_plt)
        if is_fspect:
            with st.expander("Amplitude spectra"):
                fspect_plt = self.plot_fspectra(seismic_data, 'Original')
                st.write(fspect_plt)

    def compare_two_fig_2D(self, data1, data1_name, data2, data2_name, is_fspect, sample_rate):
        vm, cmap_option = self._vm, self._cmap_option
        compare_two_fig_2D_helper(data1, data2, cmap_option, vm)
        if is_fspect:
            with st.expander("Amplitude spectra"):
                fspect_plt = plot_fspectra(data1, data1_name, sample_rate, data2=data2, data2_name=data2_name)
                st.write(fspect_plt)

    def plot_fspectra(self, data1, data1_name):
        col1, col2, col3, col4, col5 = st.columns(5)
        smooth = col2.slider('Smoothing', 0.0, 4.0, 2.0)
        sample_rate = col1.number_input("Sample Rate", min_value=1, value=int(self._sample_rate/1000), format='%i')
        col3.text((signaltonoise(data1)))

        fspect_plt = plot_fspectra(data1, data1_name, sample_rate, smooth)
        st.write(fspect_plt)
    def plot_fspectra_2(self, data1, data1_name, data2, data2_name):
        fspect_plt = plot_fspectra(data1, data1_name, int(self._sample_rate/1000), data2=data2, data2_name=data2_name)
        st.write(fspect_plt)

    def plot_fspectra_3(self, data1, data1_name, data2, data2_name, data3, data3_name):
        fspect_plt = plot_fspectra(data1, data1_name, int(self._sample_rate/1000), data2=data2, data2_name=data2_name, data3=data3, data3_name=data3_name)
        st.write(fspect_plt)

def plot_seis(seis, cmap, vmin, vmax): 
    seis_plot = px.imshow(seis, zmin=vmin, zmax=vmax, aspect='auto', labels=dict(x="Xline_idx", y="Time_idx", color="Amplitude"), color_continuous_scale=cmap)
    return seis_plot

def plot_inl(seis, cmap, vmin, vmax): 
    seis_plot = px.imshow(seis, zmin=vmin, zmax=vmax, aspect='auto', labels=dict(x="Xline_idx", y="Time_idx", color="Amplitude"), color_continuous_scale=cmap)
    return seis_plot

def plot_xln(seis, cmap, vmin, vmax):
    seis_plot = px.imshow(seis, zmin=vmin, zmax=vmax, aspect='auto', labels=dict(x="Iline_idx", y="Time_idx", color="Amplitude"), color_continuous_scale=cmap)
    return seis_plot

def plot_top(seis, cmap, vmin, vmax):
    seis_plot = px.imshow(seis, zmin=vmin, zmax=vmax, aspect='auto', labels=dict(x="Xline_idx", y="Iline_idx", color="Amplitude"), color_continuous_scale=cmap)
    return seis_plot

st.experimental_memo
def save_figure_in_temp(fig1):
    fig1_path = tempfile.NamedTemporaryFile()
    fig1.write_image(fig1_path.name+".jpg")
    return fig1_path.name+".jpg"

# @st.experimental_memo(suppress_st_warning=True)
def compare_two_fig_2D_helper(data1, data2, cmap_option, vm):         
    data1_plt = plot_seis(data1,cmap=cmap_option, vmin=-vm, vmax=vm)
    data2_plt = plot_seis(data2,cmap=cmap_option, vmin=-vm, vmax=vm)
    fig1_path = save_figure_in_temp(data1_plt)
    fig2_path = save_figure_in_temp(data2_plt)
    image_comparison(
            img1=fig1_path,
            img2=fig2_path,
        )

def add_trace_to_fspectra_fig(fig, freq, data, data_name, sample_rate, smooth):
    _, ampb = fspectra(data, dt=sample_rate, sigma=smooth)
    fig.add_trace(go.Scatter(x=freq, y=ampb,
                    mode='lines',
                    name=data_name))
    return fig
    
def plot_fspectra(data1, data1_name, sample_rate, smooth, *args, **kwargs):
    data2, data3 = kwargs.get('data2', None), kwargs.get('data3', None)
    data2_name, data3_name = kwargs.get('data2_name', None), kwargs.get('data3_name', None)

    freq, ampa = fspectra(data1, dt=sample_rate, sigma=smooth)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq, y=ampa,
                        mode='lines',
                        name=data1_name))
    if data2 is not None:
        fig = add_trace_to_fspectra_fig(fig, freq, data2, data2_name, sample_rate, smooth)
    if data3 is not None:
        fig = add_trace_to_fspectra_fig(fig, freq, data3, data3_name, sample_rate, smooth)
    fig.update_layout(title='Amplitude spectra',
                        xaxis_title='Frequency (Hz)',
                        yaxis_title='Amplitude',
                        xaxis_range=[0,110],
                        )
    print("sample_ratesample_ratesample_ratesample_rate, ", sample_rate)
    return fig