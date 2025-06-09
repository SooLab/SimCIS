#!/bin/bash

# Define an array containing all iteration counts FOR ABOUT ONE EPOCH
itratioin=(142 75 154 150 143 234 225 367 154 237 )

# Iterate through task t
for t in 2 3 4 5 6 7 8 9 10 11; do
    # Calculate index
    index=$((t - 2))  # Array index starts from 0, so t=3 corresponds to index 2 in the array

    # Ensure index is within valid range
    if [ $index -ge 0 ] && [ $index -lt ${#itratioin[@]} ]; then
        iter=${itratioin[$index]}
        
        python train_continual.py --dist-url auto --num-gpus 1 --config-file configs/ade20k/semantic-segmentation/100-5.yaml \
            CONT.TASK ${t} SOLVER.BASE_LR 0.0 SOLVER.MAX_ITER $iter CONT.COLLECT_QUERY_MODE True OUTPUT_DIR ./output/ss/100-5/step${t}

        python train_continual.py --dist-url auto --num-gpus 4 --config-file configs/ade20k/semantic-segmentation/100-5.yaml \
            CONT.TASK ${t} CONT.KL_ALL True SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 CONT.LIB_SIZE 80 CONT.COLLECT_QUERY_MODE False OUTPUT_DIR ./output/ss/100-5/step${t}
    else
        echo "Index $index out of range for itratioin array"
    fi
done
