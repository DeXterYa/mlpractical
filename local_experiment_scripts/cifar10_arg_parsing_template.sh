#!/bin/sh

cd ..
export DATASET_DIR="data/"
# Activate the relevant virtual environment:

python train_evaluate_emnist_classification_system.py --batch_size 50 --continue_from_epoch -1 --seed 0 \
                                                      --image_num_channels 3 --image_height 32 --image_width 32 \
                                                      --num_epochs 100 --experiment_name "MultiTaskMixedImage_cifar10_SE_1" \
                                                      --growth_rate 24 --block_config "(5, 5, 5, 5)" --compression 0.5 \
                                                      --num_init_feature 24 --bn_size 8 --drop_rate 0.2 --avgpool_size 8 \
                                                      --attention_pooling_type "avg_pool" --dataset_name "cifar10_mixed"\
                                                      --attention_network_type "fcc_network" --attention_pooling_size 1\
                                                      --num_attention_layers 3  --num_attention_filters 50\
                                                      --conv_bn_relu_type_bottleneck "SqueezeExciteConv2dNormLeakyReLU" \
                                                      --conv_bn_relu_type_processing "Conv2dNormLeakyReLU" \
                                                      --num_images_per_input 4\
                                                      --use_gpu "True" --gpu_id "0" --weight_decay_coefficient 0.0