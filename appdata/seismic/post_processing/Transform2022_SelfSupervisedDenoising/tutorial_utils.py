import numpy as np
import random
import itertools
from scipy.signal import filtfilt, butter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader 
from tqdm import tqdm



def regular_patching_2D(data, 
                        patchsize=[64, 64], 
                        step=[16, 16], 
                        verbose=True):
    """ Regular sample and extract patches from a 2D array
    :param data: np.array [y,x]
    :param patchsize: tuple [y,x]
    :param step: tuple [y,x]
    :param verbose: boolean
    :return: np.array [patch#, y, x]
    """

    # find starting indices
    x_start_indices = np.arange(0, data.shape[0] - patchsize[0], step=step[0])
    y_start_indices = np.arange(0, data.shape[1] - patchsize[1], step=step[1])
    starting_indices = list(itertools.product(x_start_indices, y_start_indices))

    if verbose:
        print('Extracting %i patches' % len(starting_indices))

    patches = np.zeros([len(starting_indices), patchsize[0], patchsize[1]])

    for i, pi in enumerate(starting_indices):
        patches[i] = data[pi[0]:pi[0]+patchsize[0], pi[1]:pi[1]+patchsize[1]]

    return patches


def add_whitegaussian_noise(d, sc=0.5):
    """ Add white gaussian noise to data patch
    
    Parameters
    ----------
    d: np.array [y,x]
        Data to add noise to
    sc: float 
        noise scaling value
        
    Returns
    -------
        d+n: np.array 
            Created noisy data
        n: np.array 
            Additive noise        
    """
    
    n = np.random.normal(size=d.shape)

    return d + (n * sc), n


def add_bandlimited_noise(d, lc=2, hc=80, sc=0.5):
    """ Add bandlimited noise to data patch
    
    Parameters
    ----------
    d: np.array [y,x]
        Data to add noise to
    lc: float 
        Low cut for bandpass
    hc: float 
        High cut for bandpass
    sc: float 
        Noise scaling value
        
    Returns
    -------
        d+n: np.array 
            Created noisy data
        n: np.array 
            Additive noise        
    """
    n = band_limited_noise(size=d.shape, lowcut=lc, highcut=hc)

    return d + (n * sc), n


def add_trace_wise_noise(d,
                         num_noisy_traces,
                         noisy_trace_value,
                         num_realisations,
                        ):  
    """ Add trace-wise noise to data patch
    
    Parameters
    ----------
    d: np.array [shot,y,x]
        Data to add noise to
    num_noisy_traces: int 
        Number of noisy traces to add to shots
    noisy_trace_value: int 
        Value of noisy traces
    num_realisations: int 
        Number of repeated applications per shot
        
    Returns
    -------
        alldata: np.array 
            Created noisy data
    """
    
    alldata=[]
    for k in range(len(d)):        
        clean=d[k]    
        data=np.ones([num_realisations,d.shape[1],d.shape[2]])
        for i in range(len(data)):    
            corr = np.random.randint(0,d.shape[2], num_noisy_traces) 
            data[i] = clean.copy()
            data[i,:,corr] = np.ones([1,d.shape[1]])*noisy_trace_value
        alldata.append(data)
        
    alldata=np.array(alldata) 
    alldata=alldata.reshape(num_realisations*d.shape[0],d.shape[1],d.shape[2])
    print(alldata.shape)

    return alldata


def butter_bandpass(lowcut, highcut, fs, order=5):
    """ Bandpass filter
    
    Parameters
    ----------
    lowcut: int
        Low cut for bandpass
    highcut: int 
        High cut for bandpass
    fs: int 
        Sampling frequency
    order: int 
        Filter order
        
    Returns
    -------
        b : np.array 
            The numerator coefficient vector of the filter
        a : np.array 
            The denominator coefficient vector of the filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """ Apply bandpass filter to trace
    
    Parameters
    ----------
    data: np.array [1D]
        Data onto which to apply bp filter
    lowcut: int
        Low cut for bandpass
    highcut: int 
        High cut for bandpass
    fs: int 
        Sampling frequency
    order: int 
        Filter order
        
    Returns
    -------
        y : np.array 
            Bandpassed data
    """
    
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def array_bp(data, lowcut, highcut, fs, order=5):
    """ Apply bandpass filter to array of traces
    
    Parameters
    ----------
    data: np.array [2D]
        Data onto which to apply bp filter
    lowcut: int
        Low cut for bandpass
    highcut: int 
        High cut for bandpass
    fs: int 
        Sampling frequency
    order: int 
        Filter order
        
    Returns
    -------
        bp : np.array [2D]
            Bandpassed data
    """
    bp = np.vstack([butter_bandpass_filter(data[:, ix], lowcut, highcut, fs, order)
                    for ix in range(data.shape[1])])

    return bp


def band_limited_noise(size, lowcut, highcut, fs=250):
    """ Generate bandlimited noise
    
    Parameters
    ----------
    size: tuple 
        Size of array on which to create the noise
    lowcut: int
        Low cut for bandpass
    highcut: int 
        High cut for bandpass
    fs: int 
        Sampling frequency
        
    Returns
    -------
        bpnoise : np.array 
            Bandpassed noise
    """

    basenoise = np.random.normal(size=size)
    # Pad top and bottom due to filter effects
    basenoise_pad = np.vstack([np.zeros([50, size[1]]), basenoise, np.zeros([50, size[1]])])
    # Bandpass base noise
    bpnoise =  array_bp(basenoise_pad, lowcut, highcut, fs, order=5)[:,50:-50]

    return bpnoise.T

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    
    Parameters
    ----------
    seed: int 
        Integer to be used for the seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


def weights_init(m):
    """Initialise weights of NN
    
    Parameters
    ----------
    m: torch.model 
        NN
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight)
        nn.init.constant(m.bias, 0)
        

def make_data_loader(noisy_patches, 
                     corrupted_patches, 
                     masks, 
                     n_training,
                     n_test,
                     batch_size,
                     torch_generator
                    ):
    """Make data loader to be used for the training and validation of a blind-spot NN
    
    Parameters
    ----------
    noisy_patches: np.array
        Patches of noisy data to be network target
    corrupted_patches: np.array
        Patches of processed noisy data to be network input
    masks: np.array
        Masks corresponding to corrupted_patches, indicating location of active pixels
    n_training: int
        Number of samples to be used for training
    n_test: int
        Number of samples to be used for validation
    batch_size: int
        Size of data batches to be used during training
    torch_generator: torch.generator
        For reproducibility of data loader
        
    Returns
    -------
        train_loader : torch.DataLoader
            Training data separated by batch
        test_loader : torch.DataLoader
            Validation data separated by batch
    """
    
    # Define Train Set
    # Remember to add 1 to 2nd dim - Pytorch is [#data, #channels, height, width]
    train_X = np.expand_dims(corrupted_patches[:n_training],axis=1)
    train_y = np.expand_dims(noisy_patches[:n_training],axis=1)    
    msk = np.expand_dims(masks[:n_training],axis=1)   
    # Convert to torch tensors and make TensorDataset
    train_dataset = TensorDataset(torch.from_numpy(train_X).float(), 
                                  torch.from_numpy(train_y).float(), 
                                  torch.from_numpy(msk).float(),)

    # Define Test Set
    test_X = np.expand_dims(corrupted_patches[n_training:n_training+n_test],axis=1)
    test_y = np.expand_dims(noisy_patches[n_training:n_training+n_test],axis=1)
    msk = np.expand_dims(masks[n_training:n_training+n_test],axis=1) 
    test_dataset = TensorDataset(torch.from_numpy(test_X).float(), 
                                 torch.from_numpy(test_y).float(), 
                                 torch.from_numpy(msk).float(),)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch_generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def plot_corruption(noisy, 
                    crpt, 
                    mask, 
                    seismic_cmap='RdBu', 
                    vmin=-0.25, 
                    vmax=0.25):
    """Plotting function of N2V pre-processing step
    
    Parameters
    ----------
    noisy: np.array
        Noisy data patch
    crpt: np.array
        Pre-processed data patch
    mask: np.array
        Mask corresponding to pre-processed data patch
    seismic_cmap: str
        Colormap for seismic plots
    vmin: float
        Minimum value on colour scale
    vmax: float
        Maximum value on colour scale
        
    Returns
    -------
        fig : pyplot.figure
            Figure object
        axs : pyplot.axs
            Axes of figure
    """
    
    fig,axs = plt.subplots(1,3,figsize=[15,5])
    axs[0].imshow(noisy, cmap=seismic_cmap, vmin=vmin, vmax=vmax)
    axs[1].imshow(crpt, cmap=seismic_cmap, vmin=vmin, vmax=vmax)
    axs[2].imshow(mask, cmap='binary_r')

    axs[0].set_title('Original')
    axs[1].set_title('Corrupted')
    axs[2].set_title('Corruption Mask')
    
    fig.tight_layout()
    return fig,axs

def plot_training_metrics(train_accuracy_history,
                         test_accuracy_history,
                          train_loss_history,
                          test_loss_history
                         ):
    """Plotting function of N2V training metrics
    
    Parameters
    ----------
    train_accuracy_history: np.array
        Accuracy per epoch throughout training
    test_accuracy_history: np.array
        Accuracy per epoch throughout validation
    train_loss_history: np.array
        Loss per epoch throughout training
    test_accuracy_history: np.array
        Loss per epoch throughout validation
        
    Returns
    -------
        fig : pyplot.figure
            Figure object
        axs : pyplot.axs
            Axes of figure
    """
    fig,axs = plt.subplots(1,2,figsize=(15,4))
    
    axs[0].plot(train_accuracy_history, 'r', lw=2, label='train')
    axs[0].plot(test_accuracy_history, 'k', lw=2, label='validation')
    axs[0].set_title('RMSE', size=16)
    axs[0].set_ylabel('RMSE', size=12)

    axs[1].plot(train_loss_history, 'r', lw=2, label='train')
    axs[1].plot(test_loss_history, 'k', lw=2, label='validation')
    axs[1].set_title('Loss', size=16)
    axs[1].set_ylabel('Loss', size=12)
    
    for ax in axs:
        ax.legend()
        ax.set_xlabel('# Epochs', size=12)
    fig.tight_layout()
    return fig,axs


def plot_synth_results(clean, 
                       noisy, 
                       denoised,
                       cmap='RdBu', 
                       vmin=-0.25, 
                       vmax=0.25):
    """Plotting function of synthetic results from denoising
    
    Parameters
    ----------
    clean: np.array
        Clean data patch
    noisy: np.array
        Noisy data patch
    denoised: np.array
        Denoised data patch
    cmap: str
        Colormap for plots
    vmin: float
        Minimum value on colour scale
    vmax: float
        Maximum value on colour scale
        
    Returns
    -------
        fig : pyplot.figure
            Figure object
        axs : pyplot.axs
            Axes of figure
    """
    
    fig,axs = plt.subplots(1,4,figsize=[15,4])
    axs[0].imshow(clean, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].imshow(noisy, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[2].imshow(denoised, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[3].imshow(noisy-denoised, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    axs[0].set_title('Clean')
    axs[1].set_title('Noisy')
    axs[2].set_title('Denoised')
    axs[3].set_title('Noise Removed')

    fig.tight_layout()
    return fig,axs

def plot_field_results(noisy, 
                       denoised,
                       cmap='RdBu', 
                       vmin=-0.25, 
                       vmax=0.25):
    """Plotting function of field results from denoising, i.e., where no clean is available
    
    Parameters
    ----------
    noisy: np.array
        Noisy data patch
    denoised: np.array
        Denoised data patch
    cmap: str
        Colormap for plots
    vmin: float
        Minimum value on colour scale
    vmax: float
        Maximum value on colour scale
        
    Returns
    -------
        fig : pyplot.figure
            Figure object
        axs : pyplot.axs
            Axes of figure
    """
    
    fig,axs = plt.subplots(1,2,figsize=[15,8])
    axs[0].imshow(noisy, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].imshow(denoised, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    axs[0].set_title('Noisy')
    axs[1].set_title('Denoised')

    fig.tight_layout()
    return fig,axs

def multi_active_pixels(patch, 
                        num_activepixels, 
                        neighbourhood_radius=5,
                       ):
    """ Function to identify multiple active pixels and replace with values from neighbouring pixels
    
    Parameters
    ----------
    patch : numpy 2D array
        Noisy patch of data to be processed
    num_activepixels : int
        Number of active pixels to be selected within the patch
    neighbourhood_radius : int
        Radius over which to select neighbouring pixels for active pixel value replacement
    Returns
    -------
        cp_ptch : numpy 2D array
            Processed patch 
        mask : numpy 2D array
            Mask showing location of active pixels within the patch 
    """

    n_rad = neighbourhood_radius  # descriptive variable name was a little long

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    # STEP ONE: SELECT ACTIVE PIXEL LOCATIONS
    idx_aps = np.random.randint(0, patch.shape[0], num_activepixels)
    idy_aps = np.random.randint(0, patch.shape[1], num_activepixels)
    id_aps = (idx_aps, idy_aps)
    
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    # STEP TWO: SELECT NEIGHBOURING PIXEL LOCATIONS
    
    # PART 1: Compute Shift
    # For each active pixel compute shift for finding neighbouring pixel and find pixel
    x_neigh_shft = np.random.randint(-n_rad // 2 + n_rad % 2, n_rad // 2 + n_rad % 2, num_activepixels)
    y_neigh_shft = np.random.randint(-n_rad // 2 + n_rad % 2, n_rad // 2 + n_rad % 2, num_activepixels)
    
    # OPTIONAL: don't allow replacement with itself
    for i in range(len(x_neigh_shft)):
        if x_neigh_shft[i] == 0 and y_neigh_shft[i] == 0:
            # This means its replacing itself with itself...
            shft_options = np.trim_zeros(np.arange(-n_rad // 2 + 1, n_rad // 2 + 1))
            x_neigh_shft[i] = np.random.choice(shft_options[shft_options != 0], 1)

    # PART 2: Find x and y locations of neighbours for the replacement
    idx_neigh = idx_aps + x_neigh_shft
    idy_neigh = idy_aps + y_neigh_shft    
    # Ensure neighbouring pixels within patch window
    idx_neigh = idx_neigh + (idx_neigh < 0) * patch.shape[0] - (idx_neigh >= patch.shape[0]) * patch.shape[0]
    idy_neigh = idy_neigh + (idy_neigh < 0) * patch.shape[1] - (idy_neigh >= patch.shape[1]) * patch.shape[1]
    # Get x,y of neighbouring pixels
    id_neigh = (idx_neigh, idy_neigh)
    
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    # STEP THREE: REPLACE ACTIVE PIXEL VALUES BY NEIGHBOURS
    cp_ptch = patch.copy()
    cp_ptch[id_aps] = patch[id_neigh]
    
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    # STEP FOUR: MAKE ACTIVE PIXEL MASK
    # Make mask and corrupted patch
    mask = np.ones_like(patch)
    mask[id_aps] = 0.

    return cp_ptch, mask

def n2v_train(model, 
              criterion, 
              optimizer, 
              data_loader, 
              device):
    """ Blind-spot network training function
    
    Parameters
    ----------
    model : torch model
        Neural network
    criterion : torch criterion
        Loss function 
    optimizer : torch optimizer
        Network optimiser
    data_loader : torch dataloader
        Premade data loader with training data batches
    device : torch device
        Device where training will occur (e.g., CPU or GPU)
    
    Returns
    -------
        loss : float
            Training loss across full dataset (i.e., all batches)
        accuracy : float
            Training RMSE accuracy across full dataset (i.e., all batches) 
    """
    
    model.train()
    accuracy = 0  # initialise accuracy at zero for start of epoch
    loss = 0  # initialise loss at zero for start of epoch

    for dl in tqdm(data_loader):
        # Load batch of data from data loader 
        X, y, mask = dl[0].to(device), dl[1].to(device), dl[2].to(device)
        
        optimizer.zero_grad()
        
        # Predict the denoised image based on current network weights
        yprob = model(X)

        #  Compute loss function only at masked locations and backpropogate it
        ls = criterion(yprob * (1 - mask), y * (1 - mask))
        ls.backward()        
        
        optimizer.step()
        with torch.no_grad():
            yprob = yprob
            ypred = (yprob.detach().cpu().numpy()).astype(float)
            
        # Retain training metrics
        loss += ls.item()  
        accuracy += np.sqrt(np.mean((y.cpu().numpy().ravel( ) - ypred.ravel() )**2))  
        
    # Divide cumulative training metrics by number of batches for training
    loss /= len(data_loader)  
    accuracy /= len(data_loader)  

    return loss, accuracy

def n2v_evaluate(model,
                 criterion, 
                 optimizer, 
                 data_loader, 
                 device):
    """ Blind-spot network evaluation function
    
    Parameters
    ----------
    model : torch model
        Neural network
    criterion : torch criterion
        Loss function 
    optimizer : torch optimizer
        Network optimiser
    data_loader : torch dataloader
        Premade data loader with training data batches
    device : torch device
        Device where network computation will occur (e.g., CPU or GPU)
    
    Returns
    -------
        loss : float
            Validation loss across full dataset (i.e., all batches)
        accuracy : float
            Validation RMSE accuracy across full dataset (i.e., all batches) 
    """
    
    model.train()
    accuracy = 0  # initialise accuracy at zero for start of epoch
    loss = 0  # initialise loss at zero for start of epoch

    for dl in tqdm(data_loader):
        
        # Load batch of data from data loader 
        X, y, mask = dl[0].to(device), dl[1].to(device), dl[2].to(device)
        optimizer.zero_grad()
        
        yprob = model(X)

        with torch.no_grad():            
            # Compute loss function only at masked locations 
            ls = criterion(yprob * (1 - mask), y * (1 - mask))
        
            ypred = (yprob.detach().cpu().numpy()).astype(float)
        
        # Retain training metrics
        loss += ls.item()  
        accuracy += np.sqrt(np.mean((y.cpu().numpy().ravel( ) - ypred.ravel() )**2))  
        
    # Divide cumulative training metrics by number of batches for training
    loss /= len(data_loader)  
    accuracy /= len(data_loader)  

    return loss, accuracy

# Define a basic function to calculate frequency spectra
# As we are using an fft we should really taper or window the data to prevent edge effects, 
# but for this demonstration we are going to conveniently ignore this

# An FFT returns a vector of complex values, in this case we are using the RFFT option to just return the real values 

def fspectra(data, dt = 1):
    """
    Calculate the frequency spectra
    """
    # Amplitude values
    
    # Get the absolute value of the Fourier coefficients
    fc = np.abs(np.fft.rfft(data, axis = -1))
    # Take the mean to get the amplitude values of the spectra
    a = np.mean(fc, axis = 0)
    # Get the frequency values corresponding to the coefficients
    # We need the length of the window and the sample interval in seconds 
                
    dts = dt / 1000
    length = data.shape[-1]
                
    f = np.fft.rfftfreq(length, d = dts)
               
    return f, a