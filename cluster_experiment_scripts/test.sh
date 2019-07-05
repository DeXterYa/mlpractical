#!/bin/bash
cd ..
ls
cd cifar10_test_exp/
cd result_outputs/



for i in {1..4}
do

    for j in {1..3}
    do

          echo -n "result_${i}_${j}"
    done


done