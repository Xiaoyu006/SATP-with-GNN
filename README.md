# GNN-RNN-Based-Trajectory-Prediction-ITSC2021
This repo contains the code for the paper 'Graph and Recurrent Neural Network-based Vehicle Trajectory Prediction For Highway Driving' at ITSC-2021.
## Data Pre-processing
Select the vehicles which have change their lanes only once through out the study area.

`python once_lc_veh_selector.py`

Preprocess data for the single trajectory prediction task.

`python stp_data_pre.py`

The dataset is managed by 

`stp_gr_dataset.py`

### Models
The proposed two channel model with GNN & RNN is implemented in `stp_gr_model.py`

The proposed interaction-only one channel model is implemented in `stp_g_model.py`

The proposed dynamics-only one channel model is implemented in `stp_r_model.py`

These models are based on a base model implemented in `stp_base_model.py`

## Training
Train the two channel model:

`python train.py`

## Citation
If you find this repo helpful in your work, please cite our paper as:

@inproceedings{mo2021graph, 
  title={Graph and Recurrent Neural Network-based Vehicle Trajectory Prediction For Highway Driving}, 
  author={Mo, Xiaoyu and Xing, Yang and Lv, Chen}, 
  booktitle={2021 IEEE International Intelligent Transportation Systems Conference (ITSC)}, 
  pages={1934--1939}, 
  year={2021}, 
  organization={IEEE} 
}

Thank you!
