#!/bin/bash
# Called by qsub_NLR.sh, qsub_NHR.sh and qsub_HRST_CNN.sh
# Submit to GPU
# We use the batch system Grid Engine in our lab. The qsub command is used to submit the jobs. If you do not use that, just use the command in demo.sh.


########################################################################################
#MODEL=EDSR
#SUBMODEL=EDSR
N_BLOCK=32
N_FEATS=256
#N_PATCH=192
#SCALE=4
#EPOCH=100
#MODEL_ONE=one
#SUBSET=.

if [ $MODEL == 'SR' ]; then
	CHECKPOINT="${SUBMODEL}_X${SCALE}_F${N_FEATS}B${N_BLOCK}P${N_PATCH}E${EPOCH}"
elif [ $MODEL == 'FINETUNE' ]; then
	# NLR-Sub, NLR, NHR, HRST-CNN
    CHECKPOINT="${SUBMODEL}_X${SCALE}_F${N_FEATS}B${N_BLOCK}P${N_PATCH}E${EPOCH}_${NORMAL}_${MODEL_ONE}_Input"
else
	# EDSR-FT
	CHECKPOINT="${MODEL}_X${SCALE}_F${N_FEATS}B${N_BLOCK}P${N_PATCH}E${EPOCH}_${MODEL_ONE}_WO_NORMAL"
fi

if [ $MODEL == 'FINETUNE' ]; then
	LR=0.0001
else
	# learning rate used for EDSR-FT
	LR=0.000025
fi

if [ ! ${SUBSET} == '.' ]; then
    CHECKPOINT=${CHECKPOINT}_${SUBSET}
fi
export CHECKPOINT=${CHECKPOINT}

echo $LR
echo $CHECKPOINT
qsub -N $CHECKPOINT ./vpython.sh main.py --model $MODEL --submodel $SUBMODEL --save $CHECKPOINT --scale $SCALE --patch_size $N_PATCH --n_resblocks $N_BLOCK --n_feats $N_FEATS --epochs $EPOCH --res_scale 0.1 --print_every 100 --lr $LR --lr_decay 100 --batch_size 8 --pre_train ../experiment/model/EDSR_model/EDSR_x${SCALE}.pt --test_every 200 --data_train texture_hr --data_test texture_hr --model_one ${MODEL_ONE} --subset ${SUBSET} --normal_lr $NORMAL --input_res ${INPUT_RES}

# --ext sep_reset: used to reset the dataset (generate .npy files).
# --data_train: used to judge how to test model, namely, whether to use cross validation. For all of the models trained with texture or texture_hr, cross validation should be used. Only testing EDSR do not used cross validation.
# --model_one: used to split the dataset.
# --subset: used to choose from the Subset. Five options: . (means the combination of all of the subsets), ETH3D, MiddleBury, Collection, SyB3R.
# --normal_lr: use hr or lr normal map
# --input_res: HRST_CNN should use the HR input texture map





