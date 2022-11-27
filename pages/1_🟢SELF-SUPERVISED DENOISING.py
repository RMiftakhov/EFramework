import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from custom_blocks import sidebar, save_data_form

import torch
import torch.nn as nn
from visualization_helpers import VISUALIZATION
from appdata.seismic.post_processing.Transform2022_SelfSupervisedDenoising.unet import UNet
from appdata.seismic.post_processing.Transform2022_SelfSupervisedDenoising.tutorial_utils import regular_patching_2D, add_whitegaussian_noise, add_bandlimited_noise, weights_init, set_seed, make_data_loader,plot_corruption, plot_training_metrics, plot_synth_results, plot_field_results, multi_active_pixels, n2v_train, n2v_evaluate
import time


@st.experimental_memo
def apply_noise(data, option, noises_seq):
    noisydata = np.empty(0)
    if (option == noises_seq[0]):
        noisydata = data
    elif (option == noises_seq[1]):
        noisydata, _ = add_whitegaussian_noise(data, sc=0.1)
    elif (option == noises_seq[2]):
        noisydata, _ = add_bandlimited_noise(data, sc=0.1)
    else: 
        st.error("SOMETHING WRONG!")
    return noisydata

@st.experimental_memo
def prepare_pathes(data, patch_x, patch_y):
    noisy_patches = regular_patching_2D(data, 
                                        patchsize=[patch_x, patch_y], # dimensions of extracted patch
                                        step=[4,6], # step to be taken in y,x for the extraction procedure
                                    )

    # Randomise patch order
    shuffler = np.random.permutation(len(noisy_patches))
    noisy_patches = noisy_patches[shuffler] 
    return noisy_patches

@st.experimental_memo
def construct_blindspot(data, perc_active, neighbourhood_radius):
    # Compute the total number of pixels within a patch
    total_num_pixels = data[0].shape[0]*data[0].shape[1]
    # Compute the number that should be active pixels based on the choosen percentage
    num_activepixels = int(np.floor((total_num_pixels/100) * perc_active))
    print("Number of active pixels selected: \n %.2f percent equals %i pixels"%(perc_active,num_activepixels))

    # Input the values of your choice into your pre-processing function
    crpt_patch, mask = multi_active_pixels(data[5], 
                                        num_activepixels=num_activepixels, 
                                        neighbourhood_radius=neighbourhood_radius, 
                                        )
    return crpt_patch, mask, num_activepixels

@st.experimental_memo(suppress_st_warning=True)
def apply_model(data, _network): 
    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = torch.device(torch.cuda.current_device())
        print(f'Device: {device} {torch.cuda.get_device_name(device)}')
    else:
        st.warning("No GPU available!")
    # Make a new noisy realisation so it's different from the training set but with roughly same level of noise
    # testdata, _ = add_bandlimited_noise(data, sc=0.1)
    testdata = data
    # Convert dataset in tensor for prediction purposes
    torch_testdata = torch.from_numpy(np.expand_dims(np.expand_dims(data,axis=0),axis=0)).float()
    print(torch_testdata.shape)
    # Run test dataset through network
    with torch.no_grad():
        _network.eval()
        test_prediction = _network(torch_testdata.to(device))

    # Return to numpy for plotting purposes
    test_pred = test_prediction.detach().cpu().numpy().squeeze()
    return testdata, test_pred
    
def train_model(noisy_patches, lr, n_epochs, n_training, n_test, batch_size, num_activepixels, neighbourhood_radius, weights_path, weights_name):
    # Select device for training
    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = torch.device(torch.cuda.current_device())
        print(f'Device: {device} {torch.cuda.get_device_name(device)}')
    else:
        st.warning("No GPU available!")
    network = UNet(input_channels=1, 
                output_channels=1, 
                hidden_channels=32,
                levels=2).to(device)
    # Initialise UNet's weights from pre-made function
    network = network.apply(weights_init) 
    lr = lr 
    criterion = nn.L1Loss()  # Loss function
    optim = torch.optim.Adam(network.parameters(), lr=lr)  # Optimiser

    # Initialise arrays to keep track of train and validation metrics
    train_loss_history = np.zeros(n_epochs)
    train_accuracy_history = np.zeros(n_epochs)
    test_loss_history = np.zeros(n_epochs)
    test_accuracy_history = np.zeros(n_epochs)

    # Create torch generator with fixed seed for reproducibility, to be used with the data loaders
    g = torch.Generator()
    g.manual_seed(0)

    my_bar = st.progress(0)
    placeholder = st.empty()
    t_start = 0
    # TRAINING
    for ep in range(n_epochs): 
        with placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Epoch out of {}".format(n_epochs), value=ep)
            col2.metric(label="Estimate wait time", value= "--" if t_start == 0 else time.strftime('%H:%M:%S',time.gmtime(((t_end - t_start)*(n_epochs-ep)))))
            col3.metric(label="Seconds to predict 1 epoch", value= "--" if t_start == 0 else str(round((t_end - t_start), 2)))            
            fig,axs = plot_training_metrics(train_accuracy_history,
                        test_accuracy_history,
                        train_loss_history,
                        test_loss_history
                    )
            st.write(fig)
        my_bar.progress(ep/n_epochs)
        t_start = time.time()
        # RANDOMLY CORRUPT THE NOISY PATCHES
        corrupted_patches = np.zeros_like(noisy_patches)
        masks = np.zeros_like(corrupted_patches)
        for pi in range(len(noisy_patches)):
            
            # USE ACTIVE PIXEL FUNCTION TO COMPUTE INPUT DATA AND MASKS
            corrupted_patches[pi], masks[pi] = multi_active_pixels(noisy_patches[pi], 
                                                                num_activepixels=int(num_activepixels), 
                                                                neighbourhood_radius=neighbourhood_radius,)
        # MAKE DATA LOADERS - using pre-made function 
        train_loader, test_loader = make_data_loader(noisy_patches, corrupted_patches, masks, n_training,
                                                    n_test, batch_size = batch_size,torch_generator=g)
        # TRAIN
        train_loss, train_accuracy = n2v_train(network, criterion, optim, train_loader, device,)
        # Keeping track of training metrics
        train_loss_history[ep], train_accuracy_history[ep] = train_loss, train_accuracy
        # EVALUATE (AKA VALIDATION)
        test_loss, test_accuracy = n2v_evaluate(network, criterion, optim, test_loader, device,)
        # Keeping track of validation metrics
        test_loss_history[ep], test_accuracy_history[ep] = test_loss, test_accuracy
        # PRINTING TRAINING PROGRESS
        print(f'''Epoch {ep}, 
        Training Loss {train_loss:.4f},     Training Accuracy {train_accuracy:.4f}, 
        Test Loss {test_loss:.4f},     Test Accuracy {test_accuracy:.4f} ''')
        placeholder.empty()
        t_end = time.time()
        
    torch.save(network, weights_path+weights_name)
    return train_accuracy_history, test_accuracy_history, train_loss_history, test_loss_history 

st.markdown("# 🛈 2D Self Supervised Denoising by [Claire](https://cebirnie92.github.io/) (Links: [Paper](https://arxiv.org/abs/2109.07344), [GitHub](https://github.com/swag-kaust/Transform2022_SelfSupervisedDenoising))")

with st.expander("Abstract"):
    st.subheader("The potential of self-supervised networks for random noise suppression in seismic data")
    st.subheader("Authors: Claire Birnie, Matteo Ravasi, Tariq Alkhalifah, Sixiu Liu")
    st.write("""Noise suppression is an essential step in any seismic processing workflow. A portion of this noise, particularly in land datasets, presents itself as random noise. In recent years, neural networks have been successfully used to denoise seismic data in a supervised fashion. However, supervised learning always comes with the often unachievable requirement of having noisy-clean data pairs for training. Using blind-spot networks, we redefine the denoising task as a self-supervised procedure where the network uses the surrounding noisy samples to estimate the noise-free value of a central sample. Based on the assumption that noise is statistically independent between samples, the network struggles to predict the noise component of the sample due to its randomnicity, whilst the signal component is accurately predicted due to its spatio-temporal coherency. Illustrated on synthetic examples, the blind-spot network is shown to be an efficient denoiser of seismic data contaminated by random noise with minimal damage to the signal; therefore, providing improvements in both the image domain and down-the-line tasks, such as inversion. To conclude the study, the suggested approach is applied to field data and the results are compared with two commonly used random denoising techniques: FX-deconvolution and Curvelet transform. By demonstrating that blind-spot networks are an efficient suppressor of random noise, we believe this is just the beginning of utilising self-supervised learning in seismic applications.""")
    
col1, col2 = st.columns(2)
with col1: 
    st.markdown("## My video about the paper")
    st.video('https://youtu.be/44NRwabN1NY')
with col2:
    st.markdown("## Claire's video on Transform 2022")
    st.video('https://youtu.be/d9yv90-JCZ0')

st.markdown("## 🔰 Install external dependencies")
with st.expander("GPU version"):
    st.code('''conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch''')
with st.expander("CPU version"):
    st.code('''conda install pytorch torchvision torchaudio cpuonly -c pytorch''')


st.markdown("## ✨ Here we go with the APP")

if "seismic" not in st.session_state or st.session_state.seismic_type=="3D":
    st.error("Please import 2D seismic first")
    st.stop()

module_name = "SelfDenoise2D"

st.write(st.session_state.filename)
seismic = st.session_state.seismic
seismic_type = st.session_state.seismic_type

# the network requites to have as input the size that is multiple of 4
seismic.make_axis_devisable_by(4)

# bunch of states that we need to keep track
if module_name not in st.session_state:
    st.session_state[module_name] = {"numpy_data" : None, "noise_applied": None, \
        "weights_path" : None, "network" : None, \
        "numpy_result" : None, "is_predicted" : False, 'cropped_info' : None, \
        "step1_status" : None, "step2_status" : None, "step3_status" : None, \
        "step4_status" : None, "step5_status" : None, "step6_status" : None} 

viz = VISUALIZATION(seismic, seismic_type)
viz.viz_data_2d(seismic.get_iline(), is_fspect=True)

noise_exp = st.expander("🟢 Step 1 - Here you add noise to noise-free data [⚠️ Feel free to skip it unless you are working with clean seismic]")
noises_seq = ['None', 'White Gaussian Noise', 'Bandlimited Noise']
noise_option = noise_exp.selectbox(
     'Select which noise want to add',
     (noises_seq))
noise_exp.write('You selected: {}'.format(noise_option))

noisydata = apply_noise(seismic.get_iline(), noise_option, noises_seq)

if noise_option != "None":
    with st.spinner('Wait for it...'):
        st.session_state[module_name]['noise_applied'] = True
        viz.compare_two_fig_2D(seismic.get_iline(), "Original", noisydata, "Noisy", True, int(seismic.get_sample_rate()/1000))



patch_exp = st.expander("🔴 Step 2 - Here you slice the seismic data into patches")
patch_exp.info(""" We just have one image to denoise (2d seismic slice), but in order to train the network, we'll need to provide it with more. We will follow standard computer vision practice and crop random patches from the seismic to train the networks.

We first extract them from the seismic at regular intervals and then rearrange them so that they appear in a random sequence. These patches will be divided into train and test datasets throughout the training phase.

Patch size in pixels along X and Y axis. Default is 32pix by 32pix. """)
patch_form = patch_exp.form("Extract Patches")
patch_form.write("The Size of Each Patch")
col1, col2 = patch_form.columns(2)
patch_x = col1.number_input('Along X axis (pix)', value=32, format='%i')
patch_y = col2.number_input('Along Y axis (pix)', value=32, format='%i')
patching_submit  = patch_form.form_submit_button("Submit")

###
if 'patching_submit' not in st.session_state:
    st.session_state.patching_submit = False
    st.session_state.noisy_patches = np.empty(0)

if patching_submit:
    if (patch_x < 16 or patch_y<16):
        st.session_state.patching_submit = False
        st.error("Patch size has to be more than 16pix")
    else:
        st.session_state.patching_submit = True
if st.session_state.patching_submit:
    with st.spinner('Wait for it...'):
        st.session_state.patching_submit = patching_submit
        # Regularly extract patches from the noisy data
        st.session_state.noisy_patches = prepare_pathes(noisydata, patch_x, patch_y)

        fig, axs = plt.subplots(3,6,figsize=[15,7])
        for i in range(6*3):
            vm = np.percentile(st.session_state.noisy_patches[i], 95)
            axs.ravel()[i].imshow(st.session_state.noisy_patches[i], cmap='RdBu', vmin=-vm, vmax=vm)
        fig.tight_layout()
        st.write(f"Extracting {len(st.session_state.noisy_patches)} patches")
        st.write(fig)



print('2_end')


blindspot_exp = st.expander("🔴 Step 3 - Here you further corrupt the training data with Blindspot tech.")
blindspot_exp.info("""
Blindspot corruption operates on patches, swapping out some of the pixels in each patch with pixels from the surrounding areas [Neighbourhood Radius]. 

This could be 1 just one pixel corruption,or a percentage of the total number of pixels in the image [Percent of Pixels to Corrupt]. 

""")
blindspot_form = blindspot_exp.form("Blindspot")
col1, col2 = blindspot_form.columns(2)
perc_active = col1.number_input('Percent of Pixels to Corrupt', value=33, format='%i')
neighbourhood_radius = col2.number_input('Neighbourhood Radius', value=15, format='%i')
blindspot_submit = blindspot_form.form_submit_button("Submit")

###
if 'blindspot_submit' not in st.session_state:
    st.session_state.blindspot_submit = False

if blindspot_submit:
    if perc_active>100 or perc_active<0.0001:
        st.session_state.blindspot_submit = False
        st.error("Percent of Pixels to Corrupt should be within 0 and 100%")
    else:
        st.session_state.blindspot_submit = True

num_activepixels = 0
if st.session_state.blindspot_submit:
    with st.spinner('Wait for it...'):
        crpt_patch, mask, num_activepixels = construct_blindspot(st.session_state.noisy_patches, perc_active, neighbourhood_radius)
        # Visulise the coverage of active pixels within a patch
        noisy_patch = st.session_state.noisy_patches[5]
        fig,axs = plot_corruption(noisy_patch, crpt_patch, mask, vmin=-np.percentile(noisy_patch, 95), vmax=np.percentile(noisy_patch, 95))
        st.write(fig)

print('3_end')


network_exp = st.expander("🔴 Step 4 - Here you set up and train the network OR load a saved model")
step4_option = network_exp.radio(
    "Do you want to train the network from scratch or use an already trained model?",
    ('Go Training', 'Use Trained Model'))
if step4_option == 'Go Training':
    network_form = network_exp.form("network")
    network_form.info("""
    The corrupted patches ☝️ are then fed into a neural network (UNet architecture), while the original patches are used for testing. The network  attempts to recover the value of a corrupted pixel by analyzing the whole image.

    Feel free to experiment with the parameters to attain the best performance on your dataset. The parameters are left as they were in Claire's solution.

    ❗Do not forget to specify the "Path to save the model" and "Name of the model".
    """)
    col1, col2, col3 = network_form.columns(3)
    lr = col1.number_input('Learning Rate', value=0.0001, format='%f') # Learning rate
    criterion = col2.selectbox(
        'Loss Function',
        ('L1_Loss', ""))
    optimizer = col3.selectbox(
        'Optimizer',
        ('Adam', ""))
    col11, col22, col33, col44 = network_form.columns(4)
    # Choose the number of epochs
    n_epochs = col11.number_input('Number of Epochs', value=25, format='%i')
    # Choose number of training and validation samples
    n_training = col22.number_input('Training Samples', value=2048, format='%i')
    n_test = col33.number_input('Testing Samples', value=512, format='%i')
    # Choose the batch size for the networks training
    batch_size = col44.number_input('Batch Size', value=128, format='%i')
    col111, col222 = network_form.columns(2)
    st.session_state.weights_path = col111.text_input('Path to save the model with slash at end, for example /Users/ruslan/Documents/GitHub/')
    st.session_state.weights_name = col222.text_input('Name of the model, for example SyntheticModel.pt')
    st.session_state[module_name]['weights_path'] = st.session_state.weights_path+st.session_state.weights_name
    network_submit = col111.form_submit_button("Submit")

    if network_submit:
        if (num_activepixels == 0):
            st.error("Please setup number of active pixels at step #3")
            network_submit = False
        if len(st.session_state.noisy_patches) == 0:
            st.error("Please setup patched data at step #2")
            network_submit = False
        print(len(st.session_state.weights_path))
        if len(st.session_state.weights_path) < 5 or len(st.session_state.weights_name) < 2:
            st.error("Please input the path and the name of the model")
            network_submit = False
        st.warning("Currently no way to stop the training process from streamlit. You Should either wait all the epocs to finis or CTRL+C will help you")

    if network_submit:
        train_accuracy_history, test_accuracy_history, train_loss_history, test_loss_history = train_model(st.session_state.noisy_patches, lr, n_epochs, n_training, n_test, batch_size, num_activepixels, neighbourhood_radius, st.session_state.weights_path, st.session_state.weights_name)
        # Plotting trainnig metrics using pre-made function
        fig,axs = plot_training_metrics(train_accuracy_history,
                                        test_accuracy_history,
                                        train_loss_history,
                                        test_loss_history
                                    )
        st.write(fig)
        st.session_state[module_name]['step4_status'] = True
else: 
    network_form = network_exp.form("network_load")
    st.session_state[module_name]['weights_path'] = network_form.text_input('Please pass here the whole path to the file', st.session_state.weights_path+st.session_state.weights_name)
    network_form.write('The selected file is: {}'.format(st.session_state[module_name]['weights_path'] ))
    network_submit = network_form.form_submit_button("Submit")
    if network_submit:
        st.session_state[module_name]['step4_status'] = True

if st.session_state[module_name]['step4_status']:
    st.session_state[module_name]['network'] = torch.load(st.session_state[module_name]['weights_path'])


apply_exp = st.expander("🔴 Step 5 - Now is the time to APPLY the trained denoising to the original seismic data")
apply_exp.info("The noisy image does not need any data patching or active pixel pre-processing. That is to say, the network may be given the noisy image for denoising without any intermediate steps.")
apply_form = apply_exp.form("apply")
apply_submit = apply_form.form_submit_button("Submit")

if apply_submit:
    with st.spinner('Wait for it...'):
        _, st.session_state[module_name]['numpy_result'] = apply_model(noisydata, st.session_state[module_name]['network'])
        st.session_state[module_name]['step5_status'] = True
        
if st.session_state[module_name]['step5_status']:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Noisy Data")
        #viz.viz_data_2d(noisydata, is_fspect=False, is_show_metrics=False)
    with col2:
        st.markdown("### Data after Cleaning")
        #viz.viz_data_2d(st.session_state[module_name]['numpy_result'], is_fspect=False, is_show_metrics=False)
    viz.viz_sidebyside_2d(noisydata, st.session_state[module_name]['numpy_result'],key=150, is_fspect=False, is_show_metrics=True)   
    if st.session_state[module_name]['noise_applied'] not in st.session_state:
        viz.plot_fspectra_2(seismic.get_iline(), "Original", data2=st.session_state[module_name]['numpy_result'], data2_name="Clean")
    else:
        viz.plot_fspectra_3(seismic.get_iline(), "Original", data2=noisydata, data2_name="Noisy",  data3=st.session_state[module_name]['numpy_result'], data3_name="Clean")



with st.expander("🟢 Final Step 6 - Save the results"):
    st.write("Here you can save the result")
    if st.session_state[module_name]['step5_status']:
        st.session_state[module_name]['step6_status'] = \
        save_data_form(st.session_state[module_name], seismic, st.session_state[module_name]['numpy_result'], st.session_state[module_name]['step3_status'])
    else: 
        st.session_state[module_name]['step6_status'] = "At step 5, apply the trained model"
    st.info(st.session_state[module_name]['step6_status'])


sidebar()