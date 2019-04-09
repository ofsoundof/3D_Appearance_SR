# How to Run the Code
## Prepare pretrained model
1. Download our pretrained model for 3D appearance SR from [google drive](https://drive.google.com/file/d/1TaBua-A0DT0jc4x_I4HVFicKOndzSBxU/view?usp=sharing). The pretrained models of NLR and NHR in the paper are included.

2. Download the pretrained EDSR model from [EDSR project page](https://github.com/thstkdgus35/EDSR-PyTorch).

3. Put the pretrained model at [./experiment](./experiment).

## Prepare dataset
1. Download the [texture map](https://drive.google.com/file/d/18rHsefdYNSEG7QMwzaS8iFHIdLOB2eND/view?usp=sharing) of the proposed 3D appearance dataset.

## Train and test
1. Please refer to [demo.sh](./code/scripts/demo.sh) for the training and testing demo script. In a batch system, you can also use [qsub_NLR.sh](./code/scripts/qsub_NLR.sh).
2. Remember to change the log directory `--dir` and data directory `--dir_data`.

# Pretrained model
The pretrained model for NLR and NHR in the paper is available at [google drive](https://drive.google.com/drive/folders/1_MjdHD8GHrZv37p9_vd2sgUB1T787RhC?usp=sharing).

# BibTeX
If you find our work useful in your research or publication, please cite our work:

Yawei Li , Vagia Tsiminaki, Radu Timofte, Marc Pollefeys, and Luc van Gool, "**3D Appearance Super-Resolution with Deep Learning**", In In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2019. 

    @inproceedings{li2019_3dappearance,
      title={3D Appearance Super-Resolution with Deep Learning},
      author={Li, Yawei and Tsiminaki, Vagia and Timofte, Radu and Pollefeys, Marc and Van Gool, Luc},
      booktitle={In Proceedings of the IEEE International Conference on Computer Vision},
      year={2019}
    }
