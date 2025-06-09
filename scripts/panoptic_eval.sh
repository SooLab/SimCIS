#!/bin/bash
t=${1:-11}
python train_continual.py --dist-url auto \
  --eval-only --num-gpus 4 \
  --config-file configs/ade20k/panoptic-segmentation/100-5.yaml \
  CONT.TASK ${t} CONT.VQ_NUMBER 0 \
  OUTPUT_DIR ./output/ps/100-5_607/step${t} \