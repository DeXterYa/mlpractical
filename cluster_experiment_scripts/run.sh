#!/bin/sh




for (( i = 1; i <= 8; i++ ))
do

    for (( j = 1 ; j <= 3; j++ ))
    do
        sbatch training_template_experiment_${i}_${j}.sh
    done


done