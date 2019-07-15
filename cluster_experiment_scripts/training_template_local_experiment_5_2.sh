cd ..

python train_evaluate_emnist_classification_system.py --batch_size 50 --continue_from_epoch -1 --seed 1 \
                                                      --image_num_channels 3 --image_height 32 --image_width 32 \
                                                      --num_epochs 100 --experiment_name "experiment_5_2" \
                                                      --growth_rate 24 --block_config "(5, 5, 5, 5)" --compression 0.5 \
                                                      --num_init_feature 24 --bn_size 8 --drop_rate 0.2 --avgpool_size 8 \
                                                      --attention_pooling_type "max_pool" --dataset_name "cifar100"\
                                                      --attention_network_type "relational_network" --attention_pooling_size 7\
                                                      --num_attention_layers 4  --num_attention_filters 50\
                                                      --conv_bn_relu_type_bottleneck "WeightAttentionalConv2dNormLeakyReLU" \
                                                      --conv_bn_relu_type_processing "Conv2dNormLeakyReLU" \
                                                      --use_gpu "True" --gpu_id "0" --weight_decay_coefficient 0.00001