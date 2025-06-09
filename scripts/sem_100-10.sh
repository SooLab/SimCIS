#!/bin/bash
itratioin=(214 287 344 535 356)

# Iterate through task t
for t in 2 3 4 5 6; do
    index=$((t - 2))  # Array index starts from 0, so t=3 corresponds to index 2 in the array

    # Ensure index is within valid range
    if [ $index -ge 0 ] && [ $index -lt ${#itratioin[@]} ]; then
        iter=${itratioin[$index]}
        
        python train_continual.py --dist-url auto --num-gpus 1 --config-file configs/ade20k/semantic-segmentation/100-10.yaml \
            CONT.TASK ${t} SOLVER.BASE_LR 0.0 SOLVER.MAX_ITER $iter CONT.COLLECT_QUERY_MODE True OUTPUT_DIR ./output/newss/100-10/step${t}

        python train_continual.py --dist-url auto --num-gpus 4 --config-file configs/ade20k/semantic-segmentation/100-10.yaml \
            CONT.TASK ${t} SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 10000 CONT.COLLECT_QUERY_MODE False OUTPUT_DIR ./output/newss/100-10/step${t}
    else
        echo "Index $index out of range for itratioin array"
    fi
done
