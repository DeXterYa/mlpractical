#!/bin/sh

cd ..
ls
for (( i = 1; i <= 4; i++ ))
do

    for (( j = 1 ; j <= 3; j++ ))
    do

       rsync -ua --progress s1891076@mlp.inf.ed.ac.uk:/home/s1891076/mlpractical/experiment_${i}_${j}/result_outputs/summary.csv /afs/inf.ed.ac.uk/user/s18/s1891076/

    done


done


rsync -ua --progress s1891076@mlp.inf.ed.ac.uk:/home/s1891076/mlpractical/cifar10_test_exp_1_1/result_outputs/summary.csv /afs/inf.ed.ac.uk/user/s18/s1891076


