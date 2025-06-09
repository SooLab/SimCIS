
python train_continual.py --num-gpus 4 --resume --config-file configs/ade20k/panoptic-segmentation/100-50.yaml \
CONT.TASK 2 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 50000 CONT.COLLECT_QUERY_MODE False OUTPUT_DIR ./output/ps/100-50/step2 \
