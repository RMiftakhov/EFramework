import plotly.express as px
import streamlit as st
import numpy as np
from scipy.ndimage import gaussian_filter


def fspectra(data, dt = 1, sigma=2):
    """
    Calculate the frequency spectra
    """
    # Amplitude values
    
    # Get the absolute value of the Fourier coefficients
    fc = np.abs(np.fft.rfft(data, axis = 0))
    # Take the mean to get the amplitude values of the spectra
    a = np.mean(fc, axis = 1)
    # Get the frequency values corresponding to the coefficients
    # We need the length of the window and the sample interval in seconds   
    dts = dt / 1000
    length = data.shape[-1]
    f = np.fft.rfftfreq(length, d = dts)
    return f, gaussian_filter(a, sigma=sigma)

def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)