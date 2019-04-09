This is the official website of our work 3D Appearance Super-Resolution with Deep Learning published on CVPR2019.

We provided 3DASR, a 3D appearance SR dataset that captures both synthetic and real scenes with a large variety of texture characteristics. The dataset contains ground truth HR texture maps and LR texture maps of scaling factors ×2, ×3, and ×4. The 3D mesh, multi-view images, projection matrices, and normal maps are also provided. We introduced a deep learning-based SR framework in the multi-view setting. We showed that 2D deep learning-based SR techniques can successfully be adapted to the new texture domain by introducing the geometric information via normal maps.

# Quick start
1. Download pretrained model and texture map dataset.
2. Put pretrained model at [./experiment/](./experiment).
3. `cd ./code/script`
   
   `CUDA_VISIBLE_DEVICES=xx python ../main.py --model FINETUNE --submodel NLR --save Test/NLR_first --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../../experiment/model/NLR/model_x2_split1.pt  --data_train texture --data_test texture --model_one one --subset . --normal_lr lr --input_res lr --chop --reset --save_results --print_model --test_only`
   
   Use `--ext sep_reset` for the first run.
   
   Be sure to change log directory `--dir` and data directory `--dir_data`.
   
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
