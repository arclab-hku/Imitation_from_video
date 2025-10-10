# Learn from videos #

Codebase for the "Learning Aggressive Animal Locomotion Skills for Quadrupedal Robots Solely
from Monocular Videos" project. This repository contains the code necessary to Aggressive skills using small amounts of video extract reference data .

Experiments are performed using the Aliengo robot from Unitree. This repository is based off of Nikita Rudin's [legged_gym](https://github.com/leggedrobotics/legged_gym) repo and Alejandro Escontrela's [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware.git), and enables us to train policies using [Isaac Gym](https://developer.nvidia.com/isaac-gym).

**Maintainer**: Liu ZHAO(zhaol@connect.hku.hk), Zeren LUO(zerluo@connect.hku.hk)

**Affiliation**: University of Hong Kong
<p align="center">
  <img src="img/output.gif" alt="Output GIF" width="800">
</p>

### Fast start 

1. video motion data is in the folder "dataset/video_motion_***"
2. The trained model can be found here:[model_25000.pt](logs/aliengo_amp/video_limp/model_25000.pt)
   
   put the logs in folder Imitation_from_video  like: Imitation_from_video/logs/aliengo_amp/video_XX
3. Before training, change the motion data path as which motion to learn in /legged_gym/envs/aliengo/aliengo_amp_config.py
   - `MOTION_FILES = glob.glob('datasets/video_motion_limp_aliengo/*')`
4. Test by vis.sh
   - `./legged_gym/scripts/aliengo_sh/proprio_base/vis.sh video_limp`
5. Train by train.sh
   - `./legged_gym/scripts/aliengo_sh/proprio_base/train.sh 0 0 video_test`
     
### Demo

Backflip result:

<p align="center">
  <img src="img/bk_realwithgazabo.png" alt="Backflip" width="800">
</p>

### Installation ###

Refer the AMP_for_hardware https://github.com/Alescontrela/AMP_for_hardware.git

### License

The source code of this package is released under [GPLv2](https://www.gnu.org/licenses/) license. We only allow it free for academic usage with several patents. 
For commercial use or cooperation, please contact Dr. Peng Lu lupeng@hku.hk.

For any technical issues, please contact me via email zhaol@connect.hku.hk.

### Citation

Zhao, L., Luo, Z., Han, Y. et al. Learning aggressive animal locomotion skills for quadrupedal robots solely from monocular videos. npj Robot 3, 32 (2025). https://doi.org/10.1038/s44182-025-00048-x
