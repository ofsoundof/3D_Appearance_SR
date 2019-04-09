# Quick
## Prepare pretrained model
1. Download our pretrained model for 3D appearance SR from [google drive](https://drive.google.com/drive/folders/1_MjdHD8GHrZv37p9_vd2sgUB1T787RhC?usp=sharing). The pretrained models of NLR and NHR in the paper are included.

2. Download the pretrained EDSR model from [EDSR project page](https://github.com/thstkdgus35/EDSR-PyTorch).

3. Put the pretrained model at ./experiment

## Prepare dataset
For fast use of the code, please refer to [demo.sh](./code/scripts/demo.sh). In a batch system, you can also use [qsub_NLR.sh](./code/scripts/qsub_NLR.sh).

# Pretrained model
The pretrained model for NLR and NHR in the paper is available at [google drive](https://drive.google.com/drive/folders/1_MjdHD8GHrZv37p9_vd2sgUB1T787RhC?usp=sharing).

# BibTeX
If you find our work useful in your research or publication, please cite our work:

Yawei Li , Vagia Tsiminaki, Radu Timofte, Marc Pollefeys, and Luc van Gool, "**3D Appearance Super-Resolution with Deep Learning**", In In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2019. 

    @InProceedings{li2018carn,
        author = {Li, Yawei and Agustsson, Eirikur and Gu, Shuhang and Timofte, Radu and Van Gool, Luc},
        title = {CARN: Convolutional Anchored Regression Network for Fast and Accurate Single Image Super-Resolution},
        booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
        month = {September},
        year = {2018}
    }
