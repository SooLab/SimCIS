# python train_continual.py --num-gpus 4 --config-file configs/ade20k/panoptic-segmentation/100-50.yaml \
# CONT.TASK 1 SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 160000 OUTPUT_DIR ./output/ps/100-50/step1

python train_continual.py --dist-url auto --num-gpus 1 --config-file configs/ade20k/semantic-segmentation/100-50.yaml \
CONT.TASK 2 SOLVER.BASE_LR 0.0 SOLVER.MAX_ITER 1900 CONT.KD_TEMPERATURE2 10.0 CONT.COLLECT_QUERY_MODE True OUTPUT_DIR output/ss/100-50/step2 \

python train_continual.py --num-gpus 4 --resume --config-file configs/ade20k/semantic-segmentation/100-50.yaml \
CONT.TASK 2 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 50000 CONT.KD_TEMPERATURE2 10.0 CONT.LIB_SIZE 80 CONT.COLLECT_QUERY_MODE False OUTPUT_DIR output/ss/100-50/step2 \
