#!/bin/sh

for i in {1..9}
do
  sbatch cifar10_arg_parsing_densenet_se_$i.sh
done
