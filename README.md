# EFramework-Apply AI/ML tools to your DATA
## 🔥 Democratize the use of AI/ML tools for Energy.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/n5wsGcQ3tAc/0.jpg)](https://www.youtube.com/watch?v=n5wsGcQ3tAc)

## What is it? 
This app serves as a framework to include some of the best **open-source** Energy Machine Learning tools. It aims at democritising the use of AI tools by lifting a need to be a skilled programmer to run open-source ML tools on your data. 

In most cases, I'll include implementation that allows you to enter the example data *(or your own)*, compute it, and save the results in each application.

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

All code comes as is, with no guarantees. All credit for original work to the authors of respective publications and implementations.
