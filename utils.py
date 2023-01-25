import plotly.express as px
import streamlit as st
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path
import base64
import os
import segyio
import time
import itertools


def fspectra(_input_data, dt = 1, sigma=2):
    """
    Calculate the frequency spectra
    
    @param _input_data: input data for which the frequency spectra is to be calculated
    @param dt: sample interval in milliseconds (default: 1)
    @param sigma: sigma value for gaussian filter (default: 2)
    @return: tuple of frequency values and amplitude values of the spectra
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
    """
    Calculate the signal-to-noise ratio in decibels
    
    @param a: input array
    @param axis: axis along which the mean and standard deviation is calculated (default: 0)
    @param ddof: degrees of freedom for the standard deviation calculation (default: 0)
    @return: signal-to-noise ratio in decibels
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))

def signaltonoise(a, axis=0, ddof=0):
    """
    Calculate the signal-to-noise ratio
    
    @param a: input array
    @param axis: axis along which the mean and standard deviation is calculated (default: 0)
    @param ddof: degrees of freedom for the standard deviation calculation (default: 0)
    @return: signal-to-noise ratio
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def img_to_bytes(img_path):
    """
    Convert an image to bytes and then base64 encode the bytes
    
    @param img_path: path to the image file
    @return: base64 encoded image as a string
    """
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path, link=''):
    """
    Convert an image to HTML code with an optional link.

    @param img_path: path to the image file
    @param link: optional link to add to the image
    @return: HTML code for the image
    """
    img_html = "<a href='{}'><img src='input_data:image/png;base64,{}' class='img-fluid' width='64' height='64'>".format(
    link,
    img_to_bytes(img_path)
    )
    return img_html

def img_to_html_custom(img_path, width, height, link=''):
    """
    Convert an image to HTML code with custom width and height and an optional link.

    @param img_path: path to the image file
    @param width: width of the image in HTML
    @param height: height of the image in HTML
    @param link: optional link to add to the image
    @return: HTML code for the image
    """
    img_html = f"<a href='{link}'><img src='input_data:image/png;base64,{img_to_bytes(img_path)}' class='img-fluid' width='{width}' height='{height}'>"
    img_html = "<a href='{}'><img src='data:image/png;base64,{}' class='img-fluid' width='{}' height='{}'>".format(
        link,
        img_to_bytes(img_path),
        width,
        height
    )
    return img_html

def find_files_in_directory(dir_path, ext):
    """
    Find all files in a directory with a specific file extension.

    @param dir_path: path to the directory
    @param ext: file extension to search for
    @return: list of files with the specified extension in the directory
    """
    # list to store files
    res = []
    # Iterate directory
    for file in os.listdir(dir_path):
        # check only text files
        if file.endswith(ext):
            res.append(file)
    return res


def min_max_normalization(data):
    """
    Normalize data using min-max normalization.

    @param data: data to normalize
    @return: normalized data
    """
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def std_mean_normalization(input_data):
    """_summary_

    Args:
        input_data (array): _description_

    Returns:
        array: _description_
    """
    return (input_data - np.mean(input_data)) / np.std(input_data)


def get_mask(os, t_dim=[128,128,128]):
    """
    Create a mask of specified dimensions with values decaying from 1 to 0 at the edges.

    @param os: length of the edge transition
    @param t_dim: dimensions of the mask (default: [128,128,128])
    @return: 3D mask of specified dimensions
    """
    # training image dimensions
    n1, n2, n3 = t_dim[0], t_dim[1], t_dim[2]
    # initialize mask with all 1's
    sc = np.zeros((n1,n2,n3),dtype=np.single)
    sc = sc+1

    # create decay values for edge transition
    sp = np.zeros((os),dtype=np.single)
    sig = os/4
    sig = 0.5/(sig*sig)
    for ks in range(os):
        ds = ks-os+1
        sp[ks] = np.exp(-ds*ds*sig)

    # apply decay values to edges of mask
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
        code adapted from https://github.com/xinwucwp/faultSeg

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

    my_bar = st.progress(0)
    metric = st.empty()
    counter = 0
    t_start = 0
    for k1 in range(c1):
        for k2 in range(c2):
            for k3 in range(c3):
                counter += 1
                with metric.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric(label="Pathes out of {}".format(c1*c2*c3), value=counter)
                    col2.metric(label="Estimate wait time", value= "--" if t_start == 0 else time.strftime('%H:%M:%S',time.gmtime(((t_end - t_start)*(c1*c2*c3-counter)))))
                    col3.metric(label="Seconds to predict 1 patch", value= "--" if t_start == 0 else str(round((t_end - t_start), 2)))
                my_bar.progress(counter/(c1*c2*c3))
                t_start = time.time()

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

                t_end = time.time()
    predict = predict/mk
    predict = predict[0:m1,0:m2,0:m3]
    return input_data, predict

def regular_patching_2D(data, 
                        patchsize=[64, 64], 
                        step=[16, 16], 
                        verbose=True):
    """ Regular sample and extract patches from a 2D array
    code adapted from: https://github.com/swag-kaust/Transform2022_SelfSupervisedDenoising  
    :param data: np.array [y,x]
    :param patchsize: tuple [y,x]
    :param step: tuple [y,x]
    :param verbose: boolean
    :return: np.array [patch#, y, x]
    """
    assert patchsize[0]<data.shape[0], f"Number of pixels in the patch has to be smaller than the dimention of the seismic {data.shape}."
    assert patchsize[1]<data.shape[1], f"Number of pixels in the patch has to be smaller than the dimention of the seismic {data.shape}."

    # find starting indices
    x_start_indices = np.arange(0, data.shape[0] - patchsize[0], step=step[0])
    y_start_indices = np.arange(0, data.shape[1] - patchsize[1], step=step[1])
    # add the last patch in XY that might be missed
    if (data.shape[0]>x_start_indices[-1]):
        x_start_indices = np.append(x_start_indices, [data.shape[0]-patchsize[0]])  
    if (data.shape[0]>x_start_indices[-1]):
        y_start_indices = np.append(y_start_indices, [data.shape[1]-patchsize[1]])      
    starting_indices = list(itertools.product(x_start_indices, y_start_indices))

    if verbose:
        print('Extracting %i patches' % len(starting_indices))

    patches = np.zeros([len(starting_indices), patchsize[0], patchsize[1]])

    for i, pi in enumerate(starting_indices):
        patches[i] = data[pi[0]:pi[0]+patchsize[0], pi[1]:pi[1]+patchsize[1]]

    return patches, starting_indices

def save_to_numpy(file_path, numpy_data):
    np.save(file_path, numpy_data)

def save_to_segy_3d(original_segy, file_path, numpy_data, session_state):
    #TODO make sure that input seismic is segy
    input_sepredict = original_segy.get_file_name()
    output_sepredict = file_path+".sgy"

    with segyio.open(input_sepredict, \
        iline=original_segy.get_iline_byte(), xline=original_segy.get_xline_byte()) as src:
        spec = segyio.spec()

        spec.sorting = int(src.sorting)

        spec.format = int(src.format)
        spec.samples =  src.samples[:]
        spec.tracecount = src.tracecount

        spec.ilines = src.ilines
        spec.xlines = src.xlines

        cropped_info = session_state['cropped_info']
        ny, nz = original_segy.get_n_xlines(), original_segy.get_n_zslices()

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
                iter = 0
                for i in range(src.ilines.min(), src.ilines.max()):
                    data = numpy_data[iter]
                    dst.iline[i] = data.astype('float32')
                    iter = iter + 1
            else:
                for itrace in range(dst.tracecount):
                    dst.header[itrace] =  src.header[itrace]
                    dst.trace[itrace] = np.zeros(len(src.samples)).astype('float32')
                iter = 0
                for i in range(src.ilines.min() + cropped_info[0,0], src.ilines.min() + cropped_info[0,1] ): # +1
                    data = np.zeros([ny, nz]) 
                    data[cropped_info[1,0]:cropped_info[1,1], cropped_info[2,0]:cropped_info[2,1]] = numpy_data[iter]
                    dst.iline[i] = data.astype('float32')
                    iter = iter + 1

def save_to_segy_2d(original_segy, file_path, numpy_data, session_state):
    #TODO make sure that input seismic is segy
    input_sepredict = original_segy.get_file_name()
    output_sepredict = file_path+".sgy"

    with segyio.open(input_sepredict, strict=False) as src:
        spec = segyio.spec()

        spec.format = int(src.format)
        spec.samples =  src.samples[:] 
        spec.tracecount = src.tracecount

        # st.write(f"Saving 2D segy. Numpy SHAPE {numpy_data.shape} sample {len(src.samples)}")

        spec.ilines = src.ilines

        nz = original_segy.get_n_zslices()

        # numpy_results = session_state['numpy_result'].get_cube()
        with segyio.create(output_sepredict, spec) as dst:
            # Copy all textual headers, including possible extended
            for i in range(src.ext_headers):
                dst.text[i] = src.text[i]
            # Copy the binary header, then insert the modifications needed for the new time axis
            dst.bin = src.bin
            # Copy all trace headers to destination file
            dst.header = src.header 
            iter = 0
            for itrace in range(dst.tracecount):
                dst.header[itrace] =  src.header[itrace]
                data = np.zeros(len(src.samples))
                data[:nz] = numpy_data[:, iter]
                dst.trace[itrace] = data.astype('float32') 
                iter += 1