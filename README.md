# EFramework-Apply AI/ML tools to your DATA
## 🔥 Democratize the use of AI/ML tools for Energy.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/n5wsGcQ3tAc/0.jpg)](https://www.youtube.com/watch?v=n5wsGcQ3tAc)

## What is it? 
Introducing EFramework, an open-source development tool designed to democratize the use of AI in the oil and gas industry. Our app serves as a framework to bring together some of the best open-source AI/ML tools in the industry, making them accessible to a wider range of users. With EFramework, there is no need to be a skilled programmer to run open-source ML tools on your own data.We provide easy-to-use code that allows you to input example data (or your own) and compute it, saving the results in each application. This means you can easily explore and experiment with different AI tools, without needing extensive programming knowledge. EFramework is a valuable resource for professionals and researchers in the oil and gas industry, as well as anyone interested in learning more about the potential of AI in this field.

*The list of intended AI/ML applications:*
| Geophysics | Petrophysics | Drilling Engineering | Reservoir Engineering | Production Engineering |
| --- | --- | --- | --- | --- | 
| ✅ 2D Self Supervised Denoising | Well-Log Lithology/Property Prediction | ROP Prediction/Optimization | Well-to-Well Interference | Predicting Well Rate Potential 
| ✅ 3D Seismic Fault Segmentation | Well-Log Outlier detection | Drillstring-Vibration Prediction | Proxy Reservoir Simulation | Virtual Rate Metering
| 2D/3D Seismic Denoising+SuperResolution | Well-Log Synthesis | Lost-Circulation Prediction | Reservoir Optimization | Predicting Well Failures 
| 2D/3D Seismic Facies Prediction | Well-to-Well correlation | DPS Incident Detection | PVT | Predicting Critical Oil Rate
| 3D Salt/Karst Delineation | | Real-Time Gas-Influx Detection | Decline Curve Analisys | Recommending Optimum Well Spacing in NFRs
| Reconstruction of Missing Seismic Data | | Drillstring-Washout Detection | | Predicting Threshold Reservoir Radius
| Acoustic Impedance prediction | | Abnormal-Drilling Detection | | Identification of Downhole Conditions in Sucker Rod Pumped Wells
| First-Break Picking | | Drill-Fluid Design | | Prediction of Multilateral Inflow Control Valve Flow Performance


## Installation 
1. Clone the repo
2. Lauch a terminal in the EFramework folder
3. Create a new conda enviroment `conda env create -f environment.yml`
4. Activate the enviroment `conda activate EFramework`
5. Run the app `streamlit run Hello.py`

## Integrated Modules
- [SeismicSelfDenoise2D](https://github.com/swag-kaust/Transform2022_SelfSupervisedDenoising) - 2D Self Supervised Seismic Denoising. Self-supervised learning offers a solution to the common limitation of the lack of noisy-clean pairs of data for training deep learning seismic denoising procedures/
- [FaultSeg3D](https://github.com/xinwucwp/faultSeg) - End-to-end convolutional neural network for 3D seismic fault segmentation that trained using synthetic datasets.
- [FaultNet](https://github.com/douyimin/FaultNet) - Efficient Training of 3D Seismic Fault Segmentation Network under Sparse Labels by Weakening Anomaly Annotation

All code comes as is, with no guarantees. All credit for original work to the authors of respective publications and implementations.
