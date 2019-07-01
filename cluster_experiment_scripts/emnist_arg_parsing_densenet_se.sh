#!/bin/sh

cd ..
export DATASET_DIR="data/"
# Activate the relevant virtual environment:

python train_evaluate_emnist_classification_system.py --batch_size 100 --continue_from_epoch 1 --seed 0 \
                                                      --image_num_channels 3 --image_height 32 --image_width 32 \
                                                      --num_epochs 100 --experiment_name 'cifar10_test_exp' \
                                                      --growth_rate 12 --block_config "(8, 8, 8)" --compression 0.5 \
                                                      --num_init_feature 24 --bn_size 4 --drop_rate 0 --avgpool_size 8 \
                                                      --attention_pooling_type 'avg_pool' --dataset_name 'cifar10'\
                                                      --attention_network_type 'fcc_network' --attention_pooling_size 1\
                                                      --num_attention_layers 2\
                                                      --use_gpu "True" --gpu_id "0" --weight_decay_coefficient 0.