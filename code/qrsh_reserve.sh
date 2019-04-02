#!/bin/bash

qrsh -l gpu=1 -l h_vmem=40G -l h_rt=24:00:00 -l h="biwirender15"
