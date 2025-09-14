# FreeDNA
Code for ICCV2025 "FreeDNA: Endowing Domain Adaptation of Diffusion-Based Dense Prediction with Training-Free Domain Noise Alignment"

# Abstract
Domain Adaptation(DA) for dense prediction tasks is an important topic, which enhances the dense prediction model's performance when tested on its unseen domain. Recently, with the development of Diffusion-based Dense Prediction (DDP) models, the exploration of DA designs tailored to this framework is worth exploring, since the diffusion model is effective in modeling the distribution transformation that comprises domain information. In this work, we propose a training-free mechanism for DDP frameworks, endowing them with DA capabilities. Our motivation arises from the observation that the exposure bias (e.g., noise statistics bias) in diffusion brings domain shift, and different domains in conditions of DDP models can also be effectively captured by the noise prediction statistics. Based on this, we propose a training-free Domain Noise Alignment (DNA) approach, which alleviates the variations of noise statistics to domain changes during the diffusion sampling process, thereby achieving domain adaptation. Specifically, when the source domain is available, we directly adopt the DNA method to achieve domain adaptation by aligning the noise statistics of the target domain with those of the source domain. For the more challenging source-free DA, inspired by the observation that regions closer to the source domain exhibit higher confidence meeting variations of sampling noise, we utilize the statistics from the high-confidence regions progressively to guide the noise statistic adjustment during the sampling process. Notably, our method demonstrates the effectiveness of enhancing the DA capability of DDP models across four common dense prediction tasks.

## Installation
Our code is based on [Marigold](https://github.com/prs-eth/Marigold). Please follow the installation instructions in this repository.

## Evaluation
Our work is evaluated on NuScenes and RobotCar_Night. Please download them [here](https://drive.google.com/drive/folders/1n2WsaGtB-tRiPyee-vAYF6Cd7EZr4RGe). You can also follow [STEPS](https://github.com/ucaszyp/STEPS) to get the two datasets.

You can get the predictions with the following instructions:
```
python run.py
```
The predictions (.npy) will be saved at the output dir.

You can translate the npy predictions to images with [npy2img.py](https://github.com/xuhang07/FreeDNA/blob/main/Marigold/npy2img.py).

You can evaluate the predictions will the following instructions:
```
python eval_robotcar.py
```

### More Applications
We have also implemented our method on [DiffBIR](https://github.com/XPixelGroup/DiffBIR) and [OpenDDVM](https://github.com/DQiaole/FlowDiffusion_pytorch). Please follow the documentation in their respective folders or the procedures provided by their authors for specific details.
### Acknowledgement

Many parts of this code are adapted from:

- [Marigold](https://github.com/prs-eth/Marigold)
- [STEPS](https://github.com/ucaszyp/STEPS)
- [ADM-ES](https://github.com/forever208/ADM-ES)
- [DiffBIR](https://github.com/XPixelGroup/DiffBIR)
- [OpenDDVM](https://github.com/DQiaole/FlowDiffusion_pytorch)

We thank the authors for sharing codes for their great works.
