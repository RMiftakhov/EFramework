import plotly.express as px
import streamlit as st
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path
import base64
import os


def fspectra(_input_data, dt = 1, sigma=2):
    """
    Calculate the frequency spectra
    """
    # Amplitude values

    input_data = _input_data[:, _input_data.any(0)]
    # Get the absolute value of the Fourier coefficients
    fc = np.abs(np.fft.rfft(input_data, axis = 0))
    # Take the mean to get the amplitude values of the spectra
    a = np.mean(fc, axis = 1)
    # Get the frequency values corresponding to the coefficients
    # We need the length of the window and the sample interval in seconds   
    dts = dt / 1000
    length = input_data.shape[-1]
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
    img_html = "<a href='{}'><img src='input_data:image/png;base64,{}' class='img-fluid' width='64' height='64'>".format(
        link,
        img_to_bytes(img_path)
    )
    return img_html

def img_to_html_custom(img_path, width, height, link=''):
    img_html = f"<a href='{link}'><img src='input_data:image/png;base64,{img_to_bytes(img_path)}' class='img-fluid' width='{width}' height='{height}'>"
    return img_html

def find_files_in_directory(dir_path, ext):
    # list to store files
    res = []
    # Iterate directory
    for file in os.listdir(dir_path):
        # check only text files
        if file.endswith(ext):
            res.append(file)
    return res

def std_mean_normalization(input_data):
    """_summary_

    Args:
        input_data (array): _description_

    Returns:
        array: _description_
    """
    return (input_data - np.mean(input_data)) / np.std(input_data)

def get_mask(os, t_dim=[128,128,128]):
    """set gaussian weights in the overlap bounaries

    Args:
        os (_type_): overlap width
        t_dim (list, optional): training image dimensions. Defaults to [128,128,128].

    Returns:
        _type_: _description_
    """
    # training image dimensions
    n1, n2, n3 = t_dim[0], t_dim[1], t_dim[2]

    sc = np.zeros((n1,n2,n3),dtype=np.single)
    sc = sc+1
    sp = np.zeros((os),dtype=np.single)
    sig = os/4
    sig = 0.5/(sig*sig)
    for ks in range(os):
        ds = ks-os+1
        sp[ks] = np.exp(-ds*ds*sig)
        for k1 in range(os):
            for k2 in range(n2):
                for k3 in range(n3):
                    sc[k1][k2][k3]=sp[k1]
                    sc[n1-k1-1][k2][k3]=sp[k1]
        for k1 in range(n1):
            for k2 in range(os):
                for k3 in range(n3):
                    sc[k1][k2][k3]=sp[k2]
                    sc[k1][n3-k2-1][k3]=sp[k2]
        for k1 in range(n1):
            for k2 in range(n2):
                for k3 in range(os):
                    sc[k1][k2][k3]=sp[k3]
                    sc[k1][k2][n3-k3-1]=sp[k3]
    return sc

def predict_with_mask(loaded_model, input_data, os=12, normalize_patch=False, t_dim=[128,128,128]):
    """ function that predicts the whole slice with a window-based approach 

    Args:
        loaded_model (tensorflow): neural network
        input_data (array): the whole slice
        os (int, optional): overlap width. Defaults to 12.
        normalize_patch (bool, optional): if to do MinMax normalization on each patch. Defaults to False.
        t_dim (list, optional): training image dimensions. Defaults to [128,128,128].

    Returns:
        _type_: _description_
    """
    # training image dimensions
    n1, n2, n3 = t_dim[0], t_dim[1], t_dim[2]

    # input_data dimensions
    m1,m2, m3 = input_data.shape[0], input_data.shape[1], input_data.shape[2]
    
    c1 = int(np.round((m1+os)/(n1-os)+0.5))
    c2 = int(np.round((m2+os)/(n2-os)+0.5))
    c3 = int(np.round((m3+os)/(n3-os)+0.5))

    p1 = (n1-os)*c1+os
    p2 = (n2-os)*c2+os
    p3 = (n3-os)*c3+os

    input_data = np.reshape(input_data,(m1,m2,m3))
    gp = np.zeros((p1,p2,p3),dtype=np.single)
    predict = np.zeros((p1,p2,p3),dtype=np.single)
    mk = np.zeros((p1,p2,p3),dtype=np.single)
    gs = np.zeros((1,n1,n2,n3,1),dtype=np.single)
    gp[0:m1,0:m2,0:m3] = input_data

    sc = get_mask(os, t_dim)

    for k1 in range(c1):
        for k2 in range(c2):
            for k3 in range(c3):
                b1 = k1*n1-k1*os
                e1 = b1+n1
                b2 = k2*n2-k2*os
                e2 = b2+n2
                b3 = k3*n3-k3*os
                e3 = b3+n3
                gs[0,:,:,:,0]=gp[b1:e1,b2:e2,b3:e3]
                if normalize_patch:
                    gs = gs-np.min(gs)
                    gs = gs/np.max(gs)
                    gs = gs*255 
                Y = loaded_model.predict(gs,verbose=1)
                Y = np.array(Y)
                predict[b1:e1,b2:e2,b3:e3]= predict[b1:e1,b2:e2,b3:e3]+Y[0,:,:,:,0]*sc
                mk[b1:e1,b2:e2,b3:e3]= mk[b1:e1,b2:e2,b3:e3]+sc
    predict = predict/mk
    predict = predict[0:m1,0:m2,0:m3]
    return input_data, predict

def save_to_numpy(file_path, numpy_data):
    np.save(file_path, numpy_data)

import segyio
def save_to_segy(original_segy, file_path, numpy_data, session_state):
    #TODO make sure that input seismic is segy
    input_sepredict = original_segy.get_file_name()
    output_sepredict = file_path+".sgy"

    with segyio.open(input_sepredict, \
        iline=original_segy.get_iline_byte(), xline=original_segy.get_xline_byte()) as src:
        spec = segyio.spec()

        spec.sorting = int(src.sorting)

        spec.format = int(src.format)
        spec.samples =  src.samples[:] # t[itmin:itmax]
        spec.tracecount = src.tracecount

        spec.ilines = src.ilines
        spec.xlines = src.xlines

        cropped_info = session_state['cropped_info']
        nx, ny, nz = original_segy.get_n_ilines(), original_segy.get_n_xlines(), original_segy.get_n_zslices()

        # numpy_results = session_state['numpy_result'].get_cube()
        with segyio.create(output_sepredict, spec) as dst:
            # Copy all textual headers, including possible extended
            for i in range(src.ext_headers):
                dst.text[i] = src.text[i]
            # Copy the binary header, then insert the modifications needed for the new time axis
            dst.bin = src.bin
            # Copy all trace headers to destination file
            dst.header = src.header 
            # Copy all trace headers to destination file
            dst.header.iline = src.header.iline

            if cropped_info is None:
                for itrace in range(dst.tracecount):
                    dst.header[itrace] =  src.header[itrace]
                    dst.trace[itrace] = np.zeros(len(src.samples)).astype('float32') 
            else:
                for itrace in range(dst.tracecount):
                    dst.header[itrace] =  src.header[itrace]
                    dst.trace[itrace] = np.zeros(len(src.samples)).astype('float32')
                # iter = 0
                # for i in range(cropped_info[0,0], cropped_info[0,1]-1):
                #     for j in range(cropped_info[1,0], cropped_info[1,1]-1):
                #         indx = j + i * ny
                #         # st.write(f"inter {iter}, i {i}, j {j}")
                #         dst.trace[indx] = numpy_results[iter, :].astype('float32') #np.ones(len(src.samples)).astype('float32')#
                #         iter = iter + 1
                iter = 0
                for i in range(cropped_info[0,0]+1, cropped_info[0,1]+1):
                    data = np.zeros([ny, nz]) 
                    data[cropped_info[1,0]:cropped_info[1,1], cropped_info[2,0]:cropped_info[2,1]] = numpy_data[iter]
                    dst.iline[i] = data.astype('float32')
                    iter = iter + 1