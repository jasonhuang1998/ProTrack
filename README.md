# ProTrack
## Table of Content
- [Introduction](#introduction)
- [Input Files](#input-files)
- [Configs](#configs)
- [Output](#output)
- [Usage](#usage)
- [Requirements](#requirements)
- [Reference](#reference)
## Introduction
- Implement the algorithm in "ProTrack: Detecting Proximity and Trajectory from Passive Wireless Traces of Mobile Devices".
- The paper used Bluetooth and WiFi as inputs, but we used Bluetooth only.
## Input Files
1. Create a folder in ./rawdata named with the date of the experiment.
2. Put wireless files in it.
## Configs
Create a yaml config file in ./config and put all the parameters in the config file in the format shown in the example already in ./config.
- date : experiment date
- module : run selected modules
- model/input file location : location of input file
- date preprocess/time window : the size of time window when preprocessing data
- sniffer list : the list of sniffers that needs to be calculated
- mobile list : the mobile list and their uuid
- relation list : the relation between each pair of cellphones
    * 0 : companion
    * 1 : leader-follower
    * 2 : independent
- start/end time : the start/end time of the experiment
## Output
The classification report
## Usage
```
pip install pipenv
pipenv install
pipenv run python main.py
```
## Requirements
* pandas
* pyyaml
* scikit-learn
* xgboost
## Reference
S. -I. Sou, F. -J. Wu and J. -Y. Tsai, "ProTrack: Detecting Proximity and Trajectory from Passive Wireless Traces of Mobile Devices," ICC 2022 - IEEE International Conference on Communications, Seoul, Korea, Republic of, 2022, pp. 4098-4103, doi: 10.1109/ICC45855.2022.9838892.