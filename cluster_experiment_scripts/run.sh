#!/bin/sh

for i in {1..13}
do
  sbatch cifar10_arg_parsing_densenet_se_$i.sh
done


for (( i = 1; i <= 4; i++ ))
do

    for (( j = 1 ; j <= 3; j++ ))
    do
        sbatch training_template_experiment_${i}_${j}.sh
    done


done