#!/bin/sh

cd ..
ls
for (( i = 1; i <= 4; i++ ))
do

    for (( j = 1 ; j <= 3; j++ ))
    do

          cd cifar10_test_exp_${i}_$j
          cd result_outputs
          echo "---------------"
          echo "result_${i}_$j"
          echo "validation"
          awk -F, '{if (a[2] < $3 && NR!=1) a[2] = $3}END{print a[2]}' summary.csv

          echo  "train"
          awk -F, '{if (a[1] < $1 && NR!=1) a[1] = $1}END{print a[1]}' summary.csv
          cat test_summary.csv
          cd ..
          cd ..
    done


done


rsync -ua --progress s1891076@mlp.inf.ed.ac.uk:/home/s1891076/mlpractical/cifar10_test_exp_1_1/result_outputs/summary.csv /afs/inf.ed.ac.uk/user/s18/s1891076


