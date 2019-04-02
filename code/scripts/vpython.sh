#!/bin/bash
#$ -l gpu=1
#$ -l h_rt=120:00:00
#$ -l h_vmem=40G
#$ -l h="biwirender15"
#$ -cwd
#$ -V
#$ -j y
#$ -o logs/

echo "$@"
echo "Reserved GPU: $SGE_GPU"
export CUDA_VISIBLE_DEVICES=$SGE_GPU
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u $@




