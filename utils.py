import plotly.express as px
import streamlit as st
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path
import base64


def fspectra(_data, dt = 1, sigma=2):
    """
    Calculate the frequency spectra
    """
    # Amplitude values

    data = _data[:, _data.any(0)]
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


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path, link=''):
    img_html = "<a href='{}'><img src='data:image/png;base64,{}' class='img-fluid' width='64' height='64'>".format(
        link,
        img_to_bytes(img_path)
    )
    return img_html

def img_to_html_custom(img_path, width, height, link=''):
    img_html = f"<a href='{link}'><img src='data:image/png;base64,{img_to_bytes(img_path)}' class='img-fluid' width='{width}' height='{height}'>"
    return img_html